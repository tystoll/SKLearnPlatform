"""
scripts/run_optuna.py
=====================
Demo: Optuna Bayesian search run.

    python -m scripts.run_optuna

Builds Strategy + Model, runs Optuna with TSCV + pruning,
prints Analyzer summary with per-fold bar chart.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.strategy  import Strategy
from src.core.model     import Model
from src.core.analyzer  import Analyzer
from scripts._demo_config import (
    STRATEGY, MODEL_NAME, MODEL_TYPE, TARGET, TRAIN_FRAC, VAL_FRAC,
    OPTUNA_TRIALS, OPTUNA_SCORING,
)

def main():
    print("\n╔" + "═" * 78 + "╗")
    print("║" + "  OPTUNA SEARCH DEMO".center(78) + "║")
    print("╚" + "═" * 78 + "╝\n")

    # ── 1. Build Strategy ────────────────────────────────────────────────────
    print(f"[1/3]  Building Strategy  (strategy={STRATEGY!r})")
    s = Strategy(strategy_=STRATEGY)
    print(f"       {s}\n")

    # ── 2. Build Model ───────────────────────────────────────────────────────
    print(f"[2/3]  Building Model  (model={MODEL_NAME!r}  target={TARGET!r})")
    m = Model(
        strategy_=s,
        model_name_=MODEL_NAME,
        model_type_=MODEL_TYPE,
        target_=TARGET,
        train_frac_=TRAIN_FRAC,
        val_frac_=VAL_FRAC,
    )
    print(f"       {m}\n")

    # ── 3. Optuna + Analyze ──────────────────────────────────────────────────
    print(f"[3/3]  Running Optuna  (trials={OPTUNA_TRIALS}  scoring={OPTUNA_SCORING!r}) ...\n")
    m.fit_optuna(
        n_trials=OPTUNA_TRIALS,
        scoring=OPTUNA_SCORING,
        refit=True,
        verbose=True,
    )

    az = Analyzer(m.to_results_df(run_id=f"optuna_{MODEL_NAME}_{STRATEGY}"))
    az.summarize()

    # ── Print study stats ────────────────────────────────────────────────────
    study = m.optuna_results.study
    print(f"  Total trials    : {len(study.trials)}")
    print(f"  Pruned trials   : {sum(1 for t in study.trials if str(t.state) == 'TrialState.PRUNED')}")
    print(f"  Best value      : {study.best_value:.4f}")
    print(f"  Best params     :")
    for k, v in study.best_params.items():
        short_k = k.split("__", 1)[-1] if "__" in k else k
        print(f"    {short_k:<30} {v}")
    print()


if __name__ == "__main__":
    main()
