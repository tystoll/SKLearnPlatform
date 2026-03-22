# Quant Research Platform

An end-to-end algorithmic trading research platform built in Python. Ingests multi-timeframe market data for a configurable watchlist via `yfinance`, processes it through a modular feature engineering pipeline with 20+ technical indicators and structural market features, and feeds it into a clean three-class ML stack — with structured result logging and Analyzer output for every run.

**Status:** `src/core/` complete. All three run scripts (`run_baseline`, `run_grid`, `run_optuna`) are confirmed working end-to-end. Next up: `src/features/` split, regime detection, multi-horizon labeling.

---

## What We Built (Refactor — March 2026)

We completed a full restructure from the old monolithic `DataPipeline` / `ModelPipeline` classes into a clean four-class architecture in `src/core/`. The pipeline runs end-to-end from raw CSV → trained model → Analyzer output on all three execution modes.

### What changed

| Before | After |
|---|---|
| `DataPipeline` (monolith) | `Data` (I/O only) + `Strategy` (features + labels) |
| `ModelPipeline` (monolith) | `Model` (preprocessor + fit/GS/Optuna) |
| No result normalization | `Analyzer` consumes `to_results_df()` from any mode |
| Hardcoded params | All configs live in `strategies.json` / `models.json` / `models_full.json` |
| Scripts didn't run | All three demo scripts working end-to-end |

### Scripts confirmed working

```bash
python -m scripts.run_baseline   # ✓ baseline fit, train/val/test scores
python -m scripts.run_grid       # ✓ GridSearchCV, fold stability, best params
python -m scripts.run_optuna     # ✓ Optuna Bayesian search, per-fold bar chart
```

---

## What It Does

### Data (`src/core/data.py`)

- Downloads OHLCV data for a full watchlist (25+ tickers) across four timeframes: `1m` (7d), `1h` (max), `1d` (max), `1mo` (max) — with rate-limit-aware sleep between requests
- Saves and loads from a consistent CSV schema: `{TICKER}_{period}_{interval}_raw.csv`
- Persists and loads fitted sklearn models via `joblib`
- Pure I/O — no indicators, no labels, no ML

### Strategy (`src/core/strategy.py`)

- Applies indicator sets defined in `strategies.json`: SMA, EMA, TEMA, RSI, momentum, ROC, volume ROC, PVP, ATR, and more
- Builds structural market features: swing highs/lows, HH/HL/LH/LL classification, break-of-structure (BOS) signals, failed BOS, liquidity sweeps, range compression, inside bars, bar anatomy
- Buckets all features into typed groups (price / bounded / delta / rate / composite) for scaler-aware preprocessing
- Generates classification labels (`y_class_3`) based on configurable forward-return thresholds and horizon windows read directly from strategy config

### Model (`src/core/model.py`)

- Builds a sklearn `Pipeline`: imputation → per-group scaling (Standard / MinMax / Robust) → optional PCA → estimator
- Time-ordered train / val / test split, no data leakage
- Three execution modes called explicitly after construction:
  - `fit()` — baseline single model run, scores train / val / test
  - `grid_search()` — GridSearchCV with TSCV, param grid from `models_full.json`
  - `fit_optuna()` — Optuna Bayesian search with TSCV + pruning, search space from `models_full.json`
- `to_results_df()` normalizes all three modes into one consistent long DataFrame for Analyzer

### Analyzer (`src/core/analyzer.py`)

- Consumes `Model.to_results_df()` — same interface regardless of run mode
- `fit()` summary: train / val / test per metric, overfit gap, val→test delta
- `grid_search()` summary: mean ± std per fold per metric, stability flag, best params stripped of pipeline prefix
- `optuna()` summary: per-fold bar chart, mean ± std, best params, trial stats
- `compare_runs()`: ranked table across multiple runs — same model vs strategies, or same strategy vs models

### Registry (`src/registry/`)

- `strategies.json` — indicator sets, horizon, buy/sell thresholds, fully decoupled from code
- `models.json` — production model configs (single param set, ready to instantiate)
- `models_full.json` — full grid search configs with param grids and scoring definitions
- `strategies.py` — loader helpers: `get_strategy()`, `get_model()`, `get_modelgs()`, `config_to_param_grid()`, `config_to_scoring()`, `build_model_from_config()`

---

## Project Structure

```
stocks/
├── .git
├── .venv
├── configs/
│   ├── run/
│   │   ├── baseline.yaml           # todo
│   │   ├── grid.yaml               # todo
│   │   └── optuna.yaml             # todo
│   ├── mlflow.yaml                 # todo
│   ├── database.py                 # database info loader - pulls from .env
│   └── engine.py                   # database connector through sqlalchemy
│
├── data/                           # .gitignore
│   └── raw/
│       └── StockDailyPrice.csv     # download from CapstoneDatabase
│
├── mlruns/                         # local MLflow tracking store
│
├── notebooks/
│   ├── analysis/
│   └── experiments/
│
├── runs/
│   ├── analyzer/                   # summaries / charts / tables
│   ├── gs/                         # JSON/CSV outputs from grid search
│   ├── models/                     # best-model joblibs
│   └── optuna/                     # Optuna trial exports
│
├── scripts/                        # ✓ done
│   ├── _demo_config.py             # ✓ shared config for all demo scripts
│   ├── run_baseline.py             # ✓ baseline fit run — working
│   ├── run_grid.py                 # ✓ GridSearchCV run — working
│   ├── run_optuna.py               # ✓ Optuna run — working
│   └── summarize_runs.py           # todo
│
├── server/                         # CSV exports from CapstoneDatabase
│
├── src/
│   ├── core/                       # ✓ done — all four classes operational
│   │   ├── data.py                 # ✓ Data class: I/O only
│   │   ├── strategy.py             # ✓ Strategy class: indicators, feature groups, labels
│   │   ├── model.py                # ✓ Model class: preprocessor, splits, fit/GS/Optuna
│   │   └── analyzer.py             # ✓ Analyzer class: per-mode result summaries
│   │
│   ├── features/                   # todo — split out from strategy.py
│   │   ├── indicators.py           # sma/ema/rsi/roc/etc pure funcs
│   │   ├── micro.py                # accel, jerk, ER, vol change, compression
│   │   ├── structure.py            # swing, BoS, range, anatomy, structure state
│   │   └── labeling.py             # forward-return targets / class targets
│   │
│   ├── registry/                   # ✓ done
│   │   ├── models.json             # production model configs
│   │   ├── models_full.json        # full grid / experimental configs
│   │   ├── strategies.json         # strategy definitions
│   │   ├── strategies.py           # loader/helpers around JSON registries
│   │   └── tickers.txt             # tickers to download data for
│   │
│   ├── tracking/                   # todo
│   │   ├── mlflow_logger.py
│   │   ├── artifacts.py
│   │   └── naming.py
│   │
│   ├── training/                   # todo
│   │   ├── preprocess.py
│   │   ├── search.py
│   │   └── scoring.py
│   │
│   ├── utils/                      # todo
│   │   ├── paths.py
│   │   ├── logging.py
│   │   └── io.py
│   │
│   └── __init__.py
│
├── tests/                          # todo
│   ├── test_data.py
│   ├── test_strategy.py
│   ├── test_model.py
│   └── test_labeling.py
│
├── .env                            # database credentials - .gitignore
├── .gitignore
├── README.md
├── requirements.txt
└── project_structure.txt
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Git

### Setup

```bash
git clone https://github.com/tystoll/SKLearnPlatform.git
cd SKLearnPlatform
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # fill in DB credentials if using SQL features
```

Make sure these `__init__.py` files exist (all empty):
```
src/__init__.py
src/core/__init__.py
src/registry/__init__.py
scripts/__init__.py
```

### Download Data

```python
from src.core.strategy import generate_data_all_times

generate_data_all_times(ticker="AAPL")   # downloads 1m / 1h / 1d / 1mo
```

### Run the Demo Scripts

All three scripts share `scripts/_demo_config.py` — edit `STRATEGY`, `MODEL_NAME`, and `OPTUNA_TRIALS` there.

```bash
# Baseline fit — train/val/test scores, overfit check
# Runtime: ~5-10 seconds
python -m scripts.run_baseline

# GridSearchCV — best params per metric, fold stability
# Runtime: HGBC_SMALL ~1-2 min  |  HGBC full ~2-4 hrs
python -m scripts.run_grid

# Optuna Bayesian search — per-fold bar chart, convergence stats
# Runtime: 50 trials ~5-10 min  |  200 trials ~30-60 min
python -m scripts.run_optuna
```

### Grid Search Runtime Reference

| Model | Combos | × 3 folds | Approx runtime |
|-------|--------|-----------|----------------|
| `HGBC_SMALL` | 8 | 24 fits | 1–2 min |
| `RFC_SMALL` | 6 | 18 fits | 1–2 min |
| `HGBC` | 2,592 | 7,776 fits | 2–4 hrs |
| `RFC` | ~900 | ~2,700 fits | 1–2 hrs |

Use `HGBC_SMALL` / `RFC_SMALL` for development and sanity checks. Switch to the full models when you're ready for a production run.

### Build a Pipeline Manually

```python
from src.core.strategy import Strategy
from src.core.model    import Model
from src.core.analyzer import Analyzer

# Strategy builds end-to-end automatically
s = Strategy(strategy_="classic_swing")

# Model wires preprocessor + splits, ready to run
m = Model(strategy_=s, model_name_="HGBC_SMALL")

# Pick your execution mode
m.fit()           # baseline
m.grid_search()   # GridSearchCV
m.fit_optuna()    # Optuna

# Analyze whichever ran
az = Analyzer(m.to_results_df(run_id="my_run_001"))
az.summarize()
```

### Share a Strategy Across Models (no data reload)

```python
s = Strategy(strategy_="classic_swing")

m1 = Model(strategy_=s, model_name_="HGBC_SMALL")
m2 = Model(strategy_=s, model_name_="RFC_SMALL")

m1.grid_search()
m2.grid_search()

# Compare both in one Analyzer
import pandas as pd
az = Analyzer(pd.concat([
    m1.to_results_df(run_id="hgbc_swing"),
    m2.to_results_df(run_id="rfc_swing"),
]))
az.compare_runs(metric="F1_macro")
```

---

## `_demo_config.py` Reference

Edit `scripts/_demo_config.py` to control every demo script from one place:

```python
STRATEGY      = "classic_swing"   # any key from strategies.json
MODEL_NAME    = "HGBC_SMALL"      # any key from models.json / models_full.json
MODEL_TYPE    = "classifier"      # "classifier" or "regressor"
TARGET        = "y_class_3"       # y_class_3 | y_up | y_fwd_ret | y_fwd_logret
TRAIN_FRAC    = 0.70
VAL_FRAC      = 0.15
OPTUNA_TRIALS = 50                # 50 trials ~5-10 min
OPTUNA_SCORING = "f1_macro"
```

Available strategies: `default`, `intraday_default`, `intraday_fast_momentum`, `daily_comprehensive`, `classic_swing`, `fib_swing`, `fast_momentum`, `trend_follow`

Available quick models: `HGBC_SMALL`, `RFC_SMALL`

Available full models: `HGBC`, `RFC`, `LOGREG`, `RIDGE_CLF`, `ET`, `SVC_MODEL`, `MLP_CLF`, `HGBR`, `RFR`, `RIDGE_REG`, `ENET_REG`, `HUBER_REG`, `MLP_REG`

---

## Stack

| Layer | Tools |
|---|---|
| Data ingestion | `yfinance` |
| Feature engineering | `pandas`, `numpy` |
| ML models | `scikit-learn` |
| Hyperparameter tuning | `Optuna`, `GridSearchCV` |
| Cross-validation | `TimeSeriesSplit` |
| Model persistence | `joblib` |

---

## Roadmap

- [ ] **`src/features/` split** — extract indicators, micro features, structure, and labeling out of `strategy.py` into dedicated modules
- [ ] **Market regime detection** — classify current macro/volatility regime to condition model selection and signal interpretation
- [ ] **Multi-horizon labeling** — predict across multiple forward windows simultaneously; evaluate which horizon fits which strategy best
- [ ] **Multi-timeframe feature fusion** — merge 1m / 1h / 1d features into unified input vectors at inference time
- [ ] **SQL integration** — persist processed features and results to a structured DB via the existing `configs/engine.py` layer
- [ ] **MLflow tracking** — `src/tracking/` wrappers for experiment logging
- [ ] **Test suite** — `tests/` coverage for Data, Strategy, Model, and labeling
- [ ] **YAML run configs** — `configs/run/baseline.yaml`, `grid.yaml`, `optuna.yaml`
