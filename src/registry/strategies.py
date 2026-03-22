"""
strategies.py - Loader and helper functions for strategies.json

strategies.json is the source of truth. This module provides:
  - load_strategies()         : load all strategies from JSON
  - get_strategy(name)        : fetch one by name
  - save_strategy(name, cfg)  : add/update a strategy and persist
  - delete_strategy(name)     : remove a strategy and persist
  - list_strategies()         : print a summary table

Usage:
    from strategies import get_strategy, load_strategies, DAILY_STRATEGIES

    strat = get_strategy("classic_swing")
    pipeline.add_indicators(strat)

    all_strats = load_strategies()
    for name, strat in DAILY_STRATEGIES.items():
        run_strategy(strat, csv_file)
"""
import json
import os
from typing import Dict
from collections.abc import Mapping
# ============================================================================
# Model Registry #############################################################
# ============================================================================
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    Ridge,
    ElasticNet,
    HuberRegressor,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
SKLEARN_MODELS = {
    "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
    "RandomForestClassifier": RandomForestClassifier,
    "RandomForestRegressor": RandomForestRegressor,
    "ExtraTreesClassifier": ExtraTreesClassifier,
    "LogisticRegression": LogisticRegression,
    "RidgeClassifier": RidgeClassifier,
    "Ridge": Ridge,
    "ElasticNet": ElasticNet,
    "HuberRegressor": HuberRegressor,
    "SVC": SVC,
    "MLPClassifier": MLPClassifier,
    "MLPRegressor": MLPRegressor,
}
# ============================================================================
# PATH #######################################################################
# ============================================================================
_DIR  = os.path.dirname(os.path.abspath(__file__))
_PATH = os.path.join(_DIR, "strategies.json")
_PATHMODEL = os.path.join(_DIR, "models.json")
_PATHGS = os.path.join(_DIR, "models_full.json")
# ============================================================================
# Config to param_grid=model_full.json Grid Search Helper ####################
# ============================================================================
def prefix_param_grid(step_name: str, grid: Mapping) -> dict:
    """
    Turns {"max_depth":[3,5]} into {"classifier__max_depth":[3,5]}.
    Works for GridSearchCV param_grid dicts.
    """
    return {f"{step_name}__{k}": v for k, v in grid.items()}
def config_to_param_grid(model_config: dict) -> dict:
    step = model_config["step_name"]
    grids = model_config.get("param_grid") or model_config.get("params") or {}
    # allow dict OR list-of-dicts
    if isinstance(grids, dict):
        grids = [grids]
    out = []
    for grid in grids:
        # fixups
        if "hidden_layer_sizes" in grid:
            hls = grid["hidden_layer_sizes"]
            if isinstance(hls, list):
                try:
                    grid["hidden_layer_sizes"] = [tuple(v) for v in hls]
                except:
                    grid["hidden_layer_sizes"] = [tuple(hls)]
        normalized = {k: (v if isinstance(v, (list, tuple)) else [v]) for k, v in grid.items()}
        out.append(prefix_param_grid(step, normalized))
    return out if len(out) > 1 else out[0]
# ============================================================================
# Build from config=model.json Model Build Helper ############################
# ============================================================================
def build_model_from_config(cfg: dict):
    """
    Build an sklearn estimator from JSON config.
    """
    class_name = cfg["sklearn_class"]
    params = cfg.get("params", {})

    if class_name not in SKLEARN_MODELS:
        raise ValueError(f"Unknown sklearn model: {class_name}")
        

    ModelClass = SKLEARN_MODELS[class_name]
    return ModelClass(**params)
# ============================================================================
# Config to scoring=model_full.json Scoring Helper ###########################
# ============================================================================
from collections.abc import Mapping
from sklearn.metrics import get_scorer
def config_to_scoring(model_config: dict):
    """
    Returns (scoring, refit)
    - scoring: str | dict[str, scorer]
    - refit: bool | str
    """
    scoring_cfg = model_config.get("scoring")
    # Default: single metric
    if scoring_cfg is None:
        return "f1_macro", "f1_macro"  # pick your default
    # Allow a single string in JSON: "scoring": "f1_macro"
    if isinstance(scoring_cfg, str):
        return scoring_cfg, scoring_cfg
    # Dict in JSON: {"F1_macro":"f1_macro", "Accuracy":"accuracy", ...}
    if isinstance(scoring_cfg, Mapping):
        scoring = {}
        for label, scorer_name in scoring_cfg.items():
            # Option A: pass strings directly (GridSearchCV accepts dict of strings)
            scoring[label] = scorer_name
            # Option B (more explicit): convert to scorer callables
            # scoring[label] = get_scorer(scorer_name)
        # Choose which metric to refit on:
        refit = model_config.get("refit") or ("F1_macro" if "F1_macro" in scoring else next(iter(scoring)))
        return scoring, refit
    raise TypeError(f"Unsupported scoring format: {type(scoring_cfg)}")
# ============================================================================
# CORE LOAD ##################################################################
# ============================================================================
# Strategy Loading ###########################################################
def load_strategies() -> Dict[str, dict]:
    """Load and return all strategies from strategies.json."""
    with open(_PATH) as f:
        return json.load(f)
def get_strategy(name: str) -> dict:
    """Fetch a single strategy by key name. Raises KeyError if not found."""
    strategies = load_strategies()
    if name not in strategies:
        raise KeyError(
            f"Strategy '{name}' not found. Available: {list(strategies.keys())}"
        )
    return strategies[name]
# Normal Model Loading #######################################################
def load_models() -> Dict[str, dict]:
    """Load and return all strategies from strategies.json."""
    with open(_PATHMODEL) as f:
        return json.load(f)
def get_model(name: str) -> dict:
    """Fetch a single strategy by key name. Raises KeyError if not found."""
    models = load_models()
    if name not in models:
        raise KeyError(
            f"Model '{name}' not found. Available: {list(models.keys())}"
        )
    return models[name]
# Grid Search Param Grid Loading #############################################
def load_modelsgs() -> Dict[str, dict]:
    """Load and return all strategies from strategies.json."""
    with open(_PATHGS) as f:
        return json.load(f)
def get_modelgs(name: str) -> dict:
    """Fetch a single strategy by key name. Raises KeyError if not found."""
    modelsgs = load_modelsgs()
    if name not in modelsgs:
        raise KeyError(
            f"Model '{name}' not found. Available: {list(modelsgs.keys())}"
        )
    return modelsgs[name]
# ============================================================================
# CORE EDIT ##################################################################
# ============================================================================
def save_strategy(name: str, config: dict) -> None:
    """
    Add or overwrite a strategy in strategies.json.

    Example - create a new one from scratch:
        save_strategy("my_rsi_only", {
            "name": "my_rsi_only",
            "description": "Just RSI at three periods",
            "indicators": [
                {"name": "rsi", "period": 7},
                {"name": "rsi", "period": 14},
                {"name": "rsi", "period": 21},
            ],
            "horizon": 5,
            "buy_thresh": 0.02,
            "sell_thresh": 0.02,
        })

    Example - clone and tweak an existing one:
        base = get_strategy("classic_swing")
        base["name"] = "classic_swing_fast"
        base["indicators"].append({"name": "rsi", "period": 2})
        base["horizon"] = 3
        save_strategy("classic_swing_fast", base)
    """
    strategies = load_strategies()
    config["name"] = name          # keep name field in sync with key
    strategies[name] = config
    _write(strategies)
    print(f"[strategies] saved '{name}'")
def delete_strategy(name: str) -> None:
    """Remove a strategy from strategies.json."""
    strategies = load_strategies()
    if name not in strategies:
        raise KeyError(f"Strategy '{name}' not found.")
    del strategies[name]
    _write(strategies)
    print(f"[strategies] deleted '{name}'")
def _write(strategies: dict) -> None:
    with open(_PATH, "w") as f:
        json.dump(strategies, f, indent=2)
# ============================================================================
# CORE SAVE ##################################################################
# ============================================================================
# Make versions for models?
def list_strategies() -> None:
    """Print a summary table of all strategies."""
    strategies = load_strategies()
    print(f"\n{'NAME':<28} {'INDICATORS':>10} {'HORIZON':>8} {'BUY%':>7} {'SELL%':>7}  DESCRIPTION")
    print("-" * 90)
    for name, s in strategies.items():
        print(
            f"{name:<28} {len(s.get('indicators', [])):>10} "
            f"{s.get('horizon', '?'):>8} "
            f"{s.get('buy_thresh', 0)*100:>6.2f}% "
            f"{s.get('sell_thresh', 0)*100:>6.2f}%  "
            f"{s.get('description', '')}"
        )
    print()
# ============================================================================
# INTERNAL List Helpers ######################################################
# ============================================================================
def _by_keys(*keys) -> Dict[str, dict]:
    all_s = load_strategies()
    return {k: all_s[k] for k in keys if k in all_s}
def _by_keysm(*keys) -> Dict[str, dict]:
    all_s = load_models()
    return {k: all_s[k] for k in keys if k in all_s}
def _by_keysgs(*keys) -> Dict[str, dict]:
    all_s = load_modelsgs()
    return {k: all_s[k] for k in keys if k in all_s}
# ============================================================================
# CONVENIENCE COLLECTIONS ####################################################
# ============================================================================
ALL_STRATEGIES = load_strategies()
ALL_MODELS = load_models()
ALL_MODELSGS = load_modelsgs()
INTRADAY_STRATEGIES = _by_keys(
            "default",
            "intraday_default",
            "intraday_fast_momentum"
)
DAILY_STRATEGIES    = _by_keys(
            "daily_comprehensive",
            "classic_swing",
            "fib_swing",
            "fast_momentum",
            "trend_follow"
)
CLASSIFICATION_MODELS = _by_keysm(
            "HGBC",
            "RFC",
            "LOGREG",
            "RIDGE_CLF",
            "ET",
            "SVC_MODEL",
            "MLP_CLF"
)
REGRESSION_MODELS = _by_keysm(
            "HGBR",
            "RFR",
            "RIDGE_REG",
            "ENET_REG",
            "HUBER_REG",
            "MLP_REG"
)
SMALL_MODELS = _by_keysgs(
            "HGBC_SMALL",
            "RFC_SMALL"
)