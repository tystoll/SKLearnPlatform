"""
src/core/analyzer.py
====================
Consumes the normalized results DataFrame from Model.to_results_df()
and produces structured summaries per execution mode.

Responsibilities:
  - Route to the right summary based on "mode" column
  - Surface overfitting gap, fold stability, generalization delta
  - Print clean human-readable tables
  - Return structured dicts for downstream use (logging, Analyzer chaining)

Does NOT touch disk — delegate persistence to Data or calling scripts.
Does NOT do ML — that lives in Model.

Usage
-----
    # From a Model that has been run
    m = Model()
    m.fit()
    az = Analyzer(m.to_results_df())
    az.summarize()

    # Or pass the df directly
    df = m.to_results_df(run_id="run_001")
    az = Analyzer(df)
    az.summarize()

    # Compare multiple runs
    az = Analyzer(pd.concat([df1, df2, df3]))
    az.compare_runs()
"""
from __future__ import annotations
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

# ============================================================================
# LOGGING
# ============================================================================
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  [Analyzer]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ============================================================================
# FORMATTING HELPERS
# ============================================================================
_W  = 80          # total line width
_SEP  = "=" * _W
_SEP2 = "-" * _W
_SEP3 = "·" * _W

def _header(text: str, char: str = "=") -> str:
    return f"\n{char * _W}\n  {text}\n{char * _W}"

def _subheader(text: str) -> str:
    return f"\n  {text}\n  {'─' * (len(text) + 2)}"

def _fmt(val, decimals: int = 4) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "   n/a  "
    return f"{val:>{decimals + 6}.{decimals}f}"

def _stability_flag(std: float, threshold: float = 0.03) -> str:
    """Simple flag: HIGH std means unstable folds."""
    if std > threshold:    return "⚠ unstable"
    if std > threshold / 2: return "~ moderate"
    return "✓ stable"

def _overfit_flag(train: float, val: float, threshold: float = 0.05) -> str:
    gap = train - val
    if gap > threshold:  return f"⚠ overfit  (+{gap:.4f})"
    if gap < -threshold: return f"⚠ underfit ({gap:.4f})"
    return f"✓ ok       ({gap:+.4f})"

# ============================================================================
# ANALYZER CLASS
# ============================================================================
class Analyzer:
    """
    Consumes Model.to_results_df() and produces mode-aware summaries.

    All public methods return a structured dict so results can be
    chained, logged, or compared programmatically.
    """

    def __init__(self, results: pd.DataFrame):
        if results is None or results.empty:
            raise ValueError("results DataFrame is empty or None.")
        required = {"run_id", "model", "strategy", "mode", "metric", "fold", "score", "params"}
        missing  = required - set(results.columns)
        if missing:
            raise ValueError(f"results DataFrame missing columns: {missing}")
        self._df = results.copy()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def summarize(self) -> Dict[str, Any]:
        """
        Auto-route to the right summary based on mode(s) present in the df.
        Prints and returns a dict of summaries keyed by mode.
        """
        modes   = self._df["mode"].unique()
        results = {}

        for mode in modes:
            sub = self._df[self._df["mode"] == mode]
            if mode == "fit":
                results["fit"] = self._summarize_fit(sub)
            elif mode == "grid_search":
                results["grid_search"] = self._summarize_grid_search(sub)
            elif mode == "optuna":
                results["optuna"] = self._summarize_optuna(sub)
            else:
                logger.warning("Unknown mode '%s' — skipping.", mode)

        return results

    # ------------------------------------------------------------------
    # Fit summary
    # ------------------------------------------------------------------
    def _summarize_fit(self, df: pd.DataFrame) -> Dict:
        """
        Baseline fit summary.

        Shows train / val / test scores per metric, overfitting gap,
        and val→test generalization delta.
        """
        model    = df["model"].iloc[0]
        strategy = df["strategy"].iloc[0]
        run_id   = df["run_id"].iloc[0]

        # Pivot: index=metric, columns=fold (train/val/test)
        pivot = (df.pivot_table(index="metric", columns="fold",
                                values="score", aggfunc="first")
                   .reindex(columns=["train", "val", "test"]))

        print(_header(f"FIT  |  model={model}  strategy={strategy}  run={run_id}"))

        print(_subheader("Scores"))
        col_w = 12
        header_row = (f"  {'metric':<22}"
                      f"{'train':>{col_w}}"
                      f"{'val':>{col_w}}"
                      f"{'test':>{col_w}}"
                      f"  {'overfit check':<20}"
                      f"  {'val→test Δ'}")
        print(header_row)
        print("  " + "─" * (_W - 2))

        summary: Dict[str, Dict] = {}
        for metric in pivot.index:
            row   = pivot.loc[metric]
            train = row.get("train", np.nan)
            val   = row.get("val",   np.nan)
            test  = row.get("test",  np.nan)

            of_flag    = _overfit_flag(train, val) if not np.isnan(train) and not np.isnan(val) else ""
            delta      = test - val if not np.isnan(test) and not np.isnan(val) else np.nan
            delta_str  = f"{delta:+.4f}" if not np.isnan(delta) else "n/a"

            print(f"  {metric:<22}"
                  f"{_fmt(train)}"
                  f"{_fmt(val)}"
                  f"{_fmt(test)}"
                  f"  {of_flag:<20}"
                  f"  {delta_str}")

            summary[metric] = {
                "train": train, "val": val, "test": test,
                "overfit_gap": train - val if not np.isnan(train) and not np.isnan(val) else None,
                "val_test_delta": delta,
            }

        print()
        print(_subheader("Headline"))
        best_metric = max(
            summary,
            key=lambda m: summary[m]["val"] if summary[m]["val"] is not None else -np.inf,
        )
        bv = summary[best_metric]
        print(f"  Best val metric   : {best_metric} = {bv['val']:.4f}")
        print(f"  Matching test     : {bv['test']:.4f}")
        print(f"  Overfit gap       : {bv['overfit_gap']:+.4f}")
        print()

        return {"mode": "fit", "model": model, "strategy": strategy,
                "run_id": run_id, "metrics": summary}

    # ------------------------------------------------------------------
    # Grid search summary
    # ------------------------------------------------------------------
    def _summarize_grid_search(self, df: pd.DataFrame) -> Dict:
        """
        Grid search summary.

        For each metric's best config: mean ± std across folds,
        stability flag, and best params (truncated for readability).
        """
        model    = df["model"].iloc[0]
        strategy = df["strategy"].iloc[0]
        run_id   = df["run_id"].iloc[0]

        metrics = df["metric"].unique()

        print(_header(f"GRID SEARCH  |  model={model}  strategy={strategy}  run={run_id}"))

        summary: Dict[str, Dict] = {}

        for metric in metrics:
            sub    = df[df["metric"] == metric]
            scores = sub["score"].dropna().values
            mean   = float(np.mean(scores))
            std    = float(np.std(scores))
            mn     = float(np.min(scores))
            mx     = float(np.max(scores))
            n_fold = len(scores)
            stab   = _stability_flag(std)

            # Params — parse from first row (all rows for same metric share best params)
            try:
                params = json.loads(sub["params"].iloc[0])
            except Exception:
                params = {}

            print(_subheader(f"Metric: {metric}"))
            print(f"  Folds    : {n_fold}")
            print(f"  Mean     : {mean:.4f}")
            print(f"  Std      : {std:.4f}    {stab}")
            print(f"  Min      : {mn:.4f}")
            print(f"  Max      : {mx:.4f}")
            print(f"  Range    : {mx - mn:.4f}")

            if params:
                print(f"\n  Best params for {metric}:")
                for k, v in params.items():
                    # Strip pipeline prefix for readability (e.g. "classifier__max_depth")
                    short_k = k.split("__", 1)[-1] if "__" in k else k
                    print(f"    {short_k:<30} {v}")

            summary[metric] = {
                "mean": mean, "std": std, "min": mn, "max": mx,
                "n_folds": n_fold, "stability": stab, "best_params": params,
            }

        print()
        print(_subheader("Headline"))
        best_metric = max(summary, key=lambda m: summary[m]["mean"])
        bm = summary[best_metric]
        print(f"  Best metric       : {best_metric}")
        print(f"  Mean ± std        : {bm['mean']:.4f} ± {bm['std']:.4f}")
        print(f"  Stability         : {bm['stability']}")
        print()

        return {"mode": "grid_search", "model": model, "strategy": strategy,
                "run_id": run_id, "metrics": summary}

    # ------------------------------------------------------------------
    # Optuna summary
    # ------------------------------------------------------------------
    def _summarize_optuna(self, df: pd.DataFrame) -> Dict:
        """
        Optuna summary.

        Per-fold scores for best trial: mean ± std, stability,
        best params, and convergence note.
        """
        model    = df["model"].iloc[0]
        strategy = df["strategy"].iloc[0]
        run_id   = df["run_id"].iloc[0]

        metrics = df["metric"].unique()

        print(_header(f"OPTUNA  |  model={model}  strategy={strategy}  run={run_id}"))

        summary: Dict[str, Dict] = {}

        for metric in metrics:
            sub    = df[df["metric"] == metric].sort_values("fold")
            scores = sub["score"].dropna().values
            mean   = float(np.mean(scores))
            std    = float(np.std(scores))
            mn     = float(np.min(scores))
            mx     = float(np.max(scores))
            n_fold = len(scores)
            stab   = _stability_flag(std)

            try:
                params = json.loads(sub["params"].iloc[0])
            except Exception:
                params = {}

            print(_subheader(f"Metric: {metric}"))
            print(f"  Folds completed  : {n_fold}")
            print(f"  Mean             : {mean:.4f}")
            print(f"  Std              : {std:.4f}    {stab}")
            print(f"  Min              : {mn:.4f}")
            print(f"  Max              : {mx:.4f}")

            # Per-fold breakdown
            print(f"\n  Per-fold scores:")
            for _, row in sub.iterrows():
                bar = "█" * int(row["score"] * 40) if row["score"] > 0 else ""
                print(f"    fold {int(row['fold']):>2}  {row['score']:.4f}  {bar}")

            if params:
                print(f"\n  Best params:")
                for k, v in params.items():
                    short_k = k.split("__", 1)[-1] if "__" in k else k
                    print(f"    {short_k:<30} {v}")

            summary[metric] = {
                "mean": mean, "std": std, "min": mn, "max": mx,
                "n_folds": n_fold, "stability": stab, "best_params": params,
            }

        print()
        print(_subheader("Headline"))
        best_metric = max(summary, key=lambda m: summary[m]["mean"])
        bm = summary[best_metric]
        print(f"  Best metric       : {best_metric}")
        print(f"  Mean ± std        : {bm['mean']:.4f} ± {bm['std']:.4f}")
        print(f"  Stability         : {bm['stability']}")
        print()

        return {"mode": "optuna", "model": model, "strategy": strategy,
                "run_id": run_id, "metrics": summary}

    # ------------------------------------------------------------------
    # Multi-run comparison
    # ------------------------------------------------------------------
    def compare_runs(self,
                     metric:   str   = None,
                     mode:     str   = None,
                     top_n:    int   = 10,
    ) -> pd.DataFrame:
        """
        Compare multiple runs in one table.

        Filters by mode and metric if provided, otherwise uses the most
        common mode and best available metric. Returns a sorted DataFrame
        and prints a ranked table.

        Useful for comparing:
          - Same model across strategies
          - Same strategy across models
          - GS vs Optuna on the same setup
        """
        df = self._df.copy()

        # Filter mode
        if mode is not None:
            df = df[df["mode"] == mode]
        else:
            mode = df["mode"].value_counts().idxmax()
            df   = df[df["mode"] == mode]

        # Filter metric
        if metric is None:
            metric = df["metric"].value_counts().idxmax()
        df = df[df["metric"] == metric]

        if df.empty:
            print(f"No results for mode='{mode}' metric='{metric}'")
            return pd.DataFrame()

        # Aggregate: mean score per run
        agg = (df.groupby(["run_id", "model", "strategy"])["score"]
                 .agg(mean_score="mean", std_score="std", n_folds="count")
                 .reset_index()
                 .sort_values("mean_score", ascending=False)
                 .head(top_n)
                 .reset_index(drop=True))
        agg["rank"] = agg.index + 1

        print(_header(f"COMPARE RUNS  |  mode={mode}  metric={metric}  top={top_n}"))
        print(f"\n  {'rank':>4}  {'model':<16}  {'strategy':<24}  "
              f"{'mean':>8}  {'std':>8}  {'folds':>6}  stability")
        print("  " + "─" * (_W - 2))
        for _, row in agg.iterrows():
            stab = _stability_flag(row["std_score"])
            print(f"  {int(row['rank']):>4}  {row['model']:<16}  {row['strategy']:<24}  "
                  f"{row['mean_score']:>8.4f}  {row['std_score']:>8.4f}  "
                  f"{int(row['n_folds']):>6}  {stab}")
        print()

        return agg

    def __repr__(self) -> str:
        modes   = self._df["mode"].unique().tolist()
        runs    = self._df["run_id"].nunique()
        metrics = self._df["metric"].nunique()
        return (f"Analyzer(runs={runs}, modes={modes}, "
                f"metrics={metrics}, rows={len(self._df)})")
