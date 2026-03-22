"""
scripts/run_grid.py
===================
Demo: GridSearchCV run.

    python -m scripts.run_grid

Builds Strategy + Model, runs GridSearchCV with TSCV,
prints Analyzer summary of best params and fold stability per metric.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.strategy  import Strategy
from src.core.model     import Model
from src.core.analyzer  import Analyzer
from scripts._demo_config import (
    STRATEGY, MODEL_NAME, MODEL_TYPE, TARGET, TRAIN_FRAC, VAL_FRAC,
)

def main():
    print("\n╔" + "═" * 78 + "╗")
    print("║" + "  GRID SEARCH DEMO".center(78) + "║")
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

    # ── 3. Grid Search + Analyze ─────────────────────────────────────────────
    print("[3/3]  Running grid search ...\n")
    m.grid_search()

    az = Analyzer(m.to_results_df(run_id=f"gs_{MODEL_NAME}_{STRATEGY}"))
    az.summarize()

    # ── Optional: score best model on val+test ───────────────────────────────
    print("\n  Scoring best model (F1_macro) on val + test ...")
    from sklearn.metrics import accuracy_score, f1_score
    best = m.get_best_model(metric="F1_macro")
    sp   = m._splits
    val_f1  = f1_score(sp.y_val,  best.predict(sp.X_val),  average="macro", zero_division=0)
    test_f1 = f1_score(sp.y_test, best.predict(sp.X_test), average="macro", zero_division=0)
    print(f"  Val  F1 macro : {val_f1:.4f}")
    print(f"  Test F1 macro : {test_f1:.4f}\n")


if __name__ == "__main__":
    main()
