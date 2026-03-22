"""
scripts/_demo_config.py
=======================
Shared config for all demo scripts.
Edit these values to change what runs.
 
Model quick-reference
---------------------
HGBC_SMALL   8 combos  ×  3 folds =    24 fits  (~1-2 min grid search)
RFC_SMALL    6 combos  ×  3 folds =    18 fits  (~1-2 min grid search)
HGBC      2592 combos  ×  3 folds =  7776 fits  (~2-4 hrs grid search)
RFC       ~900 combos  ×  3 folds = ~2700 fits  (~1-2 hrs grid search)
"""
 
# ── Strategy ─────────────────────────────────────────────────────────────────
STRATEGY   = "classic_swing"   # any key from strategies.json
 
# ── Model ─────────────────────────────────────────────────────────────────────
# Use HGBC_SMALL / RFC_SMALL for quick runs (~1-2 min)
# Use HGBC / RFC for full production runs (hours)
MODEL_NAME = "HGBC_SMALL"      # any key from models.json / models_full.json
MODEL_TYPE = "classifier"      # "classifier" or "regressor"
 
# ── Target ───────────────────────────────────────────────────────────────────
TARGET     = "y_class_3"       # y_class_3 | y_up | y_fwd_ret | y_fwd_logret
 
# ── Splits ───────────────────────────────────────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
 
# ── Optuna ───────────────────────────────────────────────────────────────────
OPTUNA_TRIALS  = 50            # 50 trials ~5-10 min; 200 trials ~30-60 min
OPTUNA_SCORING = "f1_macro"
 