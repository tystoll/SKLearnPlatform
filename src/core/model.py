"""
src/core/model.py
=================
Machine-learning layer on top of a Strategy.

Responsibilities:
  - Build sklearn ColumnTransformer preprocessor (keyed to Strategy feature groups)
  - Build sklearn Pipeline (preprocessor + optional PCA + estimator)
  - Time-ordered train / val / test split
  - Three execution modes (called manually after construction):
      fit()          — baseline single model run
      grid_search()  — GridSearchCV with TSCV
      fit_optuna()   — Optuna Bayesian search with TSCV + pruning

Does NOT touch disk — delegate save/load to Data.
Does NOT render or summarize — delegate to Analyzer.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import optuna
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.base import clone
from sklearn.metrics import (
    mean_squared_error, accuracy_score, f1_score,
    balanced_accuracy_score, check_scoring,
)

from .strategy import Strategy
from ..registry.strategies import (
    get_model,
    get_modelgs,
    config_to_scoring,
    config_to_param_grid,
    build_model_from_config,
    ALL_MODELS,
    ALL_STRATEGIES,
    ALL_MODELSGS,
)

# ============================================================================
# LOGGING
# ============================================================================
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  [Model]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ============================================================================
# CUSTOM SCORING FUNCTIONS
# ============================================================================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def directional_accuracy(y_true, y_pred):
    return np.mean(np.sign(np.asarray(y_true)) == np.sign(np.asarray(y_pred)))

def sharpe_scorer(y_true, y_pred):
    returns = np.asarray(y_pred) * np.asarray(y_true)
    return returns.mean() / (returns.std() + 1e-10)

# ============================================================================
# DEFAULTS
# ============================================================================
DEFAULT_STATE      = 42
DEFAULT_MODEL_NAME = "HGBC"
DEFAULT_MODEL_TYPE = "classifier"
DEFAULT_TARGET     = "y_class_3"
DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC   = 0.15
DEFAULT_DROPNA     = True
# PCA
DEFAULT_USE_PCA        = False
DEFAULT_PCA_COMPONENTS = 0.95
# TSCV
DEFAULT_TSCV_SPLITS         = 3
DEFAULT_TSCV_MAX_TRAIN_SIZE = None
DEFAULT_TSCV_TEST_SIZE      = 500
DEFAULT_TSCV_GAP            = 50
# GridSearch
DEFAULT_N_JOBS             = -1
DEFAULT_VERBOSE            = 2
DEFAULT_RETURN_TRAIN_SCORE = True
DEFAULT_ERROR_SCORE        = np.nan

# ============================================================================
# RESULT CONTAINERS
# ============================================================================
@dataclass
class SplitPack:
    X_train:  pd.DataFrame
    y_train:  pd.Series
    df_train: pd.DataFrame
    X_val:    pd.DataFrame
    y_val:    pd.Series
    df_val:   pd.DataFrame
    X_test:   pd.DataFrame
    y_test:   pd.Series
    df_test:  pd.DataFrame
    meta:     dict

@dataclass
class FitResults:
    estimator:    Any
    val_scores:   Dict[str, float]
    test_scores:  Dict[str, float]
    train_scores: Dict[str, float]

@dataclass
class GridSearchResults:
    cv_results:             pd.DataFrame
    best_params_per_metric: Dict[str, Dict]
    best_scores_per_metric: Dict[str, float]
    grid_search:            GridSearchCV
    best_estimator:         Optional[Any] = None

@dataclass
class OptunaResults:
    study:          Any
    best_params:    Dict
    best_value:     float
    trials_df:      pd.DataFrame
    best_estimator: Optional[Any] = None

# ============================================================================
# MODEL CLASS
# ============================================================================
class Model:
    """
    Machine-learning layer on top of a Strategy.

    build() wires everything and runs time_split() — ready to run.
    Then call one of the three execution modes:

        m = Model()                     # default Strategy + HGBC
        m.fit()                         # baseline single run
        m.grid_search()                 # GridSearchCV
        m.fit_optuna()                  # Optuna Bayesian search

    Pass a pre-built Strategy to share data across models without reloading:

        s = Strategy(strategy_="classic_swing")
        m1 = Model(strategy_=s, model_name_="HGBC")
        m2 = Model(strategy_=s, model_name_="RFC")
    """

    def __init__(self,
                 # Strategy
                 strategy_:    Strategy = None,
                 # Model
                 model_name_:  str      = None,
                 model_type_:  str      = None,
                 # Target + splits
                 target_:      str      = None,
                 train_frac_:  float    = None,
                 val_frac_:    float    = None,
                 dropna_:      bool     = None,
                 # Optional pre-built overrides
                 preprocessor_: ColumnTransformer = None,
                 model_:        Any               = None,
                 estimator_:    Pipeline          = None,
    ):
        # Result containers — always reset on construction
        self.fit_results:         Optional[FitResults]         = None
        self.grid_search_results: Optional[GridSearchResults]  = None
        self.optuna_results:      Optional[OptunaResults]      = None
        self._fitted_models:      Dict[str, Any]               = {}

        self.build(
            strategy=strategy_,
            model_name=model_name_,
            model_type=model_type_,
            target=target_,
            train_frac=train_frac_,
            val_frac=val_frac_,
            dropna=dropna_,
            preprocessor=preprocessor_,
            model=model_,
            estimator=estimator_,
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self,
              # Strategy
              strategy:    Strategy          = None,
              # Model
              model_name:  str               = None,
              model_type:  str               = None,
              # Target + splits
              target:      str               = None,
              train_frac:  float             = None,
              val_frac:    float             = None,
              dropna:      bool              = None,
              # Optional pre-built overrides
              preprocessor: ColumnTransformer = None,
              model:        Any               = None,
              estimator:    Pipeline          = None,
    ) -> 'Model':
        # -- Strategy --------------------------------------------------------
        if strategy is not None:
            self._strategy = strategy
        elif not hasattr(self, '_strategy') or self._strategy is None:
            self._strategy = Strategy()   # default Data + default strategy
        # (no patch branch — Strategy is a whole object, replace or keep)

        # -- Model identity --------------------------------------------------
        if model_type == "regressor" and model_name is None:
            model_name = "HGBR"
        self._model_name = model_name  if model_name  is not None else getattr(self, '_model_name', DEFAULT_MODEL_NAME)
        self._model_type = model_type  if model_type  is not None else getattr(self, '_model_type', DEFAULT_MODEL_TYPE)
        self._target     = target      if target      is not None else getattr(self, '_target',     DEFAULT_TARGET)
        self._train_frac = train_frac  if train_frac  is not None else getattr(self, '_train_frac', DEFAULT_TRAIN_FRAC)
        self._val_frac   = val_frac    if val_frac    is not None else getattr(self, '_val_frac',   DEFAULT_VAL_FRAC)
        self._dropna     = dropna      if dropna      is not None else getattr(self, '_dropna',     DEFAULT_DROPNA)

        # -- sklearn objects -------------------------------------------------
        model_cfg          = get_model(self._model_name)
        self._model        = model        if model        is not None else build_model_from_config(model_cfg)
        self._preprocessor = preprocessor if preprocessor is not None else None
        self._estimator    = estimator    if estimator    is not None else None
        self.build_preprocessor()
        self.build_estimator()

        # -- Splits ----------------------------------------------------------
        self._splits = self.time_split(
            target_col=self._target,
            train_frac=self._train_frac,
            val_frac=self._val_frac,
            dropna=self._dropna,
        )
        logger.info(
            "Model.build done — model=%s  train=%d  val=%d  test=%d  features=%d",
            self._model_name,
            self._splits.meta["n_train"],
            self._splits.meta["n_val"],
            self._splits.meta["n_test"],
            self._splits.meta["feature_count"],
        )
        return self

    # ------------------------------------------------------------------
    # Preprocessor + estimator
    # ------------------------------------------------------------------
    def build_preprocessor(self) -> None:
        """Build ColumnTransformer keyed to Strategy's feature groups."""
        fg = self._strategy._feature_groups
        scaler_map = {
            "price":     StandardScaler(),
            "bounded":   MinMaxScaler(feature_range=(0, 1)),
            "delta":     StandardScaler(),
            "rate":      RobustScaler(),
            "composite": RobustScaler(),
        }
        transformers = [
            (
                group,
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler",  scaler),
                ]),
                fg[group],
            )
            for group, scaler in scaler_map.items()
            if fg.get(group)
        ]
        self._preprocessor = ColumnTransformer(
            transformers=transformers, remainder="drop"
        )

    def build_estimator(self) -> None:
        """Build sklearn Pipeline: preprocessor [+ PCA] + estimator."""
        steps = [("preprocess", self._preprocessor)]
        if DEFAULT_USE_PCA:
            self._pca = PCA(
                n_components=DEFAULT_PCA_COMPONENTS, random_state=DEFAULT_STATE
            )
            steps.append(("pca", self._pca))
        steps.append((self._model_type, self._model))
        self._estimator = Pipeline(steps=steps)

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------
    def time_split(self,
                   target_col: str   = None,
                   train_frac: float = None,
                   val_frac:   float = None,
                   dropna:     bool  = None,
    ) -> SplitPack:
        """Time-ordered train / val / test split from Strategy's labeled df."""
        df_src = self._strategy._data._df
        feat   = self._strategy._feature_columns

        if df_src is None:
            raise ValueError("Strategy has no data.")
        if not feat:
            raise ValueError("No features. Strategy.build_feature_groups() must run first.")

        target = target_col if target_col is not None else self._target
        tfrac  = train_frac if train_frac is not None else self._train_frac
        vfrac  = val_frac   if val_frac   is not None else self._val_frac
        dn     = dropna     if dropna     is not None else self._dropna

        if not (0 < tfrac < 1):
            raise ValueError("train_frac must be in (0, 1).")
        if tfrac + vfrac >= 1:
            raise ValueError("train_frac + val_frac must be < 1.")

        df = df_src.sort_index().copy()
        if dn:
            df = df.dropna(subset=feat + [target])

        n         = len(df)
        train_end = int(n * tfrac)
        val_end   = int(n * (tfrac + vfrac))

        df_train = df.iloc[:train_end]
        df_val   = df.iloc[train_end:val_end]
        df_test  = df.iloc[val_end:]

        sp = SplitPack(
            X_train=df_train[feat], y_train=df_train[target], df_train=df_train,
            X_val=df_val[feat],     y_val=df_val[target],     df_val=df_val,
            X_test=df_test[feat],   y_test=df_test[target],   df_test=df_test,
            meta={
                "n_total":       n,
                "n_train":       len(df_train),
                "n_val":         len(df_val),
                "n_test":        len(df_test),
                "train_frac":    tfrac,
                "val_frac":      vfrac,
                "feature_count": len(feat),
                "target":        target,
            },
        )
        self._splits = sp
        return sp

    # ------------------------------------------------------------------
    # Execution mode 1 — Baseline fit
    # ------------------------------------------------------------------
    def fit(self,
            splits: SplitPack = None,
            refit_on_trainval: bool = False,
    ) -> FitResults:
        """
        Baseline single model run.

        Fits the estimator on train split, scores on train / val / test.
        If refit_on_trainval=True, refits on train+val before scoring test.

        Returns FitResults with fitted estimator and score dicts.
        """
        sp = splits if splits is not None else self._splits
        if sp is None:
            raise ValueError("No splits. Call time_split() first.")

        logger.info("Fit: model=%s  train=%d  val=%d  test=%d",
                    self._model_name, len(sp.X_train), len(sp.X_val), len(sp.X_test))

        est = clone(self._estimator)
        est.fit(sp.X_train, sp.y_train)

        def _score(est, X, y) -> Dict[str, float]:
            y_pred = est.predict(X)
            if self._model_type == "classifier":
                return {
                    "accuracy":          accuracy_score(y, y_pred),
                    "f1_macro":          f1_score(y, y_pred, average="macro", zero_division=0),
                    "f1_weighted":       f1_score(y, y_pred, average="weighted", zero_division=0),
                    "balanced_accuracy": balanced_accuracy_score(y, y_pred),
                }
            else:
                return {
                    "rmse": rmse(y, y_pred),
                    "mae":  float(np.mean(np.abs(y - y_pred))),
                    "dir_acc": directional_accuracy(y, y_pred),
                }

        train_scores = _score(est, sp.X_train, sp.y_train)
        val_scores   = _score(est, sp.X_val,   sp.y_val)

        if refit_on_trainval:
            X_tv = pd.concat([sp.X_train, sp.X_val])
            y_tv = pd.concat([sp.y_train, sp.y_val])
            est  = clone(self._estimator)
            est.fit(X_tv, y_tv)

        test_scores = _score(est, sp.X_test, sp.y_test)

        logger.info("Fit complete — val=%s  test=%s", val_scores, test_scores)

        self.fit_results = FitResults(
            estimator=est,
            val_scores=val_scores,
            test_scores=test_scores,
            train_scores=train_scores,
        )
        return self.fit_results

    # ------------------------------------------------------------------
    # Execution mode 2 — Grid Search
    # ------------------------------------------------------------------
    def grid_search(self, splits: SplitPack = None) -> GridSearchResults:
        """
        GridSearchCV with TSCV on training split.

        Param grid, scoring, and refit metric are read from models_full.json.
        Returns GridSearchResults with cv_results DataFrame and best params/scores
        per metric.
        """
        sp = splits if splits is not None else self._splits
        if sp is None:
            raise ValueError("No splits. Call time_split() first.")

        # Load GS config on demand — not built at construction time
        modelgs_cfg        = get_modelgs(self._model_name)
        param_grid         = config_to_param_grid(modelgs_cfg)
        scoring, refit_met = config_to_scoring(modelgs_cfg)
        tscv = TimeSeriesSplit(
            n_splits=DEFAULT_TSCV_SPLITS,
            gap=DEFAULT_TSCV_GAP,
            max_train_size=DEFAULT_TSCV_MAX_TRAIN_SIZE,
            test_size=DEFAULT_TSCV_TEST_SIZE,
        )

        logger.info(
            "Grid search: model=%s  features=%d  samples=%d  folds=%d  combos=%d",
            self._model_name, sp.X_train.shape[1], sp.X_train.shape[0],
            tscv.n_splits, self._count_param_combinations(param_grid),
        )

        gs = GridSearchCV(
            estimator=self._estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=tscv,
            refit=refit_met,
            n_jobs=DEFAULT_N_JOBS,
            verbose=DEFAULT_VERBOSE,
            return_train_score=DEFAULT_RETURN_TRAIN_SCORE,
            error_score=DEFAULT_ERROR_SCORE,
        )
        gs.fit(sp.X_train, sp.y_train)

        cv_df        = pd.DataFrame(gs.cv_results_)
        bppm:  Dict  = {}
        bspm:  Dict  = {}
        metric_names = scoring.keys() if isinstance(scoring, dict) else [scoring]
        for m in metric_names:
            col = f"mean_test_{m}"
            if col in cv_df.columns:
                idx     = cv_df[col].idxmax()
                bppm[m] = cv_df.loc[idx, "params"]
                bspm[m] = cv_df.loc[idx, col]

        self.grid_search_results = GridSearchResults(
            cv_results=cv_df,
            best_params_per_metric=bppm,
            best_scores_per_metric=bspm,
            grid_search=gs,
            best_estimator=gs.best_estimator_ if refit_met else None,
        )
        logger.info("Grid search complete. Best scores: %s", bspm)
        return self.grid_search_results

    # ------------------------------------------------------------------
    # Execution mode 3 — Optuna
    # ------------------------------------------------------------------
    def fit_optuna(self,
                   splits:         SplitPack = None,
                   n_trials:       int   = 100,
                   scoring:        str   = "f1_macro",
                   direction:      str   = "maximize",
                   n_warmup_steps: int   = 5,
                   refit:          bool  = True,
                   verbose:        bool  = False,
    ) -> OptunaResults:
        """
        Optuna Bayesian hyperparameter search with walk-forward TSCV + pruning.

        The search space is built from the model's param grid in models_full.json —
        continuous params are sampled as floats/ints, list params become categoricals.
        Returns OptunaResults with study, best_params, best_value, trials_df,
        and optionally refitted best_estimator.
        """
        sp = splits if splits is not None else self._splits
        if sp is None:
            raise ValueError("No splits. Call time_split() first.")

        X, y      = sp.X_train, sp.y_train
        step_name = self._model_type

        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Load TSCV + search space on demand — not built at construction time
        tscv = TimeSeriesSplit(
            n_splits=DEFAULT_TSCV_SPLITS,
            gap=DEFAULT_TSCV_GAP,
            max_train_size=DEFAULT_TSCV_MAX_TRAIN_SIZE,
            test_size=DEFAULT_TSCV_TEST_SIZE,
        )
        modelgs_cfg = get_modelgs(self._model_name)
        raw_params  = modelgs_cfg.get("params", {})
        # Flatten list-of-dicts to single dict for Optuna (union of all keys)
        if isinstance(raw_params, list):
            flat = {}
            for d in raw_params:
                flat.update(d)
            raw_params = flat

        def _make_suggest(trial, key, values):
            """Auto-detect suggest type from the values list."""
            prefixed = f"{step_name}__{key}"
            # Filter out non-numeric / None for range detection
            numeric = [v for v in values if isinstance(v, (int, float)) and v is not None]
            if not numeric or len(values) <= 3:
                return trial.suggest_categorical(prefixed, values)
            all_int = all(isinstance(v, int) for v in numeric)
            if all_int:
                return trial.suggest_int(prefixed, int(min(numeric)), int(max(numeric)))
            return trial.suggest_float(prefixed, float(min(numeric)), float(max(numeric)))

        def objective(trial):
            params = {}
            for key, values in raw_params.items():
                if key in ("random_state",):        # skip fixed params
                    continue
                if not isinstance(values, list) or len(values) == 0:
                    continue
                params[f"{step_name}__{key}"] = _make_suggest(trial, key, values)

            pipe        = clone(self._estimator).set_params(**params)
            scorer      = check_scoring(pipe, scoring=scoring)
            fold_scores = []

            for step_idx, (tr_idx, val_idx) in enumerate(tscv.split(X)):
                pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                fold_scores.append(scorer(pipe, X.iloc[val_idx], y.iloc[val_idx]))
                trial.report(np.mean(fold_scores), step=step_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(fold_scores)

        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=DEFAULT_STATE),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps),
        )
        logger.info("Optuna: model=%s  trials=%d  scoring=%s", self._model_name, n_trials, scoring)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_estimator = None
        if refit:
            logger.info("Refitting best model on %d samples...", len(X))
            best_estimator = clone(self._estimator).set_params(**study.best_params)
            best_estimator.fit(X, y)

        self.optuna_results = OptunaResults(
            study=study,
            best_params=study.best_params,
            best_value=study.best_value,
            trials_df=study.trials_dataframe(),
            best_estimator=best_estimator,
        )
        logger.info("Optuna complete. Best %s: %.4f", scoring, study.best_value)
        return self.optuna_results

    # ------------------------------------------------------------------
    # Post-run helpers
    # ------------------------------------------------------------------
    def get_best_model(self,
                       metric: str,
                       refit:  bool = True) -> Pipeline:
        """
        Clone estimator with best params for a given metric and optionally
        refit on train split. Caches result — calling twice returns cached.
        """
        if self.grid_search_results is None:
            raise ValueError("Run grid_search() first.")
        if metric not in self.grid_search_results.best_params_per_metric:
            raise ValueError(
                f"Metric '{metric}' not found. "
                f"Available: {list(self.grid_search_results.best_params_per_metric)}"
            )
        if metric in self._fitted_models:
            return self._fitted_models[metric]

        best_params = self.grid_search_results.best_params_per_metric[metric]
        model = clone(self._estimator).set_params(**best_params)
        if refit and self._splits is not None:
            model.fit(self._splits.X_train, self._splits.y_train)
        self._fitted_models[metric] = model
        return model

    # ------------------------------------------------------------------
    # Results export
    # ------------------------------------------------------------------
    def to_results_df(self, run_id: str = None) -> pd.DataFrame:
        """
        Normalize whichever execution mode ran into a consistent long DataFrame.

        Schema
        ------
        run_id      : str   — caller-supplied or auto-generated timestamp
        timestamp   : str   — ISO format
        model       : str
        strategy    : str
        mode        : str   — "fit" | "grid_search" | "optuna"
        metric      : str
        fold        : str   — fold index (int) for CV modes, "train"/"val"/"test" for fit
        score       : float
        params      : str   — JSON-serialised best params for that metric (or {} for fit)

        Raises ValueError if no execution mode has been run yet.
        """
        import json
        from datetime import datetime

        ts     = datetime.now().isoformat()
        rid    = run_id if run_id is not None else ts
        model  = self._model_name
        strat  = self._strategy._strategy
        rows: List[Dict] = []

        # ── fit() ───────────────────────────────────────────────────────────
        if self.fit_results is not None:
            fr     = self.fit_results
            splits = {"train": fr.train_scores,
                      "val":   fr.val_scores,
                      "test":  fr.test_scores}
            for fold_name, score_dict in splits.items():
                for metric, score in score_dict.items():
                    rows.append({
                        "run_id":    rid,
                        "timestamp": ts,
                        "model":     model,
                        "strategy":  strat,
                        "mode":      "fit",
                        "metric":    metric,
                        "fold":      fold_name,
                        "score":     score,
                        "params":    "{}",
                    })

        # ── grid_search() ───────────────────────────────────────────────────
        if self.grid_search_results is not None:
            cv_df = self.grid_search_results.cv_results

            for metric, best_params in self.grid_search_results.best_params_per_metric.items():
                # Find the row in cv_results that matches best_params for this metric
                mean_col = f"mean_test_{metric}"
                if mean_col not in cv_df.columns:
                    continue
                best_idx = cv_df[mean_col].idxmax()

                # Expand per-fold scores for the best config
                fold = 0
                while True:
                    fold_col = f"split{fold}_test_{metric}"
                    if fold_col not in cv_df.columns:
                        break
                    rows.append({
                        "run_id":    rid,
                        "timestamp": ts,
                        "model":     model,
                        "strategy":  strat,
                        "mode":      "grid_search",
                        "metric":    metric,
                        "fold":      fold,
                        "score":     cv_df.loc[best_idx, fold_col],
                        "params":    json.dumps(
                            {k: str(v) for k, v in best_params.items()},
                            sort_keys=True,
                        ),
                    })
                    fold += 1

        # ── fit_optuna() ────────────────────────────────────────────────────
        if self.optuna_results is not None:
            otr        = self.optuna_results
            best_trial = otr.study.best_trial

            # intermediate_values holds {step_idx: mean_score_up_to_that_fold}
            # Re-derive per-fold scores: score[fold] = mean[fold]*( fold+1) - mean[fold-1]*fold
            iv         = best_trial.intermediate_values   # {0: s0, 1: s1, ...}
            fold_scores: Dict[int, float] = {}
            for step in sorted(iv):
                cumulative = iv[step] * (step + 1)
                prev       = iv[step - 1] * step if step > 0 else 0.0
                fold_scores[step] = cumulative - prev

            # Optuna runs a single scoring string — use it as the metric name
            # (stored as the study's direction label isn't available, so we use
            #  the value from fit_optuna's scoring arg if accessible, else "score")
            metric_name = getattr(otr.study, "_metric_name", "score")

            params_str = json.dumps(
                {k: str(v) for k, v in otr.best_params.items()}, sort_keys=True
            )
            for fold_idx, score in fold_scores.items():
                rows.append({
                    "run_id":    rid,
                    "timestamp": ts,
                    "model":     model,
                    "strategy":  strat,
                    "mode":      "optuna",
                    "metric":    metric_name,
                    "fold":      fold_idx,
                    "score":     score,
                    "params":    params_str,
                })

        if not rows:
            raise ValueError(
                "No results to export. Run fit(), grid_search(), or fit_optuna() first."
            )

        return pd.DataFrame(rows, columns=[
            "run_id", "timestamp", "model", "strategy",
            "mode", "metric", "fold", "score", "params",
        ])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _count_param_combinations(self, param_grid) -> int:
        if isinstance(param_grid, list):
            return sum(self._count_param_combinations(pg) for pg in param_grid)
        count = 1
        for values in param_grid.values():
            count *= len(values) if isinstance(values, list) else 1
        return count

    def __repr__(self) -> str:
        n_train = self._splits.meta["n_train"] if self._splits else "?"
        n_feat  = self._splits.meta["feature_count"] if self._splits else "?"
        return (
            f"Model(name={self._model_name!r}, type={self._model_type!r}, "
            f"target={self._target!r}, train={n_train}, features={n_feat})"
        )
