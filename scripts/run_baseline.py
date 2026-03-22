"""
scripts/run_baseline.py
=======================
Demo: baseline single fit run.

    python -m scripts.run_baseline

Builds Strategy + Model, fits on train, scores train/val/test,
prints Analyzer summary.
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
    print("║" + "  BASELINE FIT DEMO".center(78) + "║")
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

    # ── 3. Fit + Analyze ─────────────────────────────────────────────────────
    print("[3/3]  Running baseline fit ...\n")
    m.fit()

    az = Analyzer(m.to_results_df(run_id=f"baseline_{MODEL_NAME}_{STRATEGY}"))
    az.summarize()


if __name__ == "__main__":
    main()
