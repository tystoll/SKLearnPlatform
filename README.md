# Quant ML Research Framework

End-to-end machine learning framework for systematic trading research.

This project builds market datasets, generates structured features, trains multiple models, and evaluates strategy performance across different market conditions.

---

## 🚀 Quick Start

```bash
git clone https://github.com/yourusername/ml-quant-research-framework.git
cd ml-quant-research-framework

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt

python -m main.gs1
📊 What This Project Does
Downloads market data using yfinance
Builds feature-rich datasets using configurable strategies
Applies technical + structural indicators
Trains multiple ML models (sklearn pipelines)
Runs time-series aware cross-validation
Performs grid search + Optuna optimization
Logs results for every strategy × model combination
Saves best models for ensemble development
🧠 Core Architecture
Data Pipeline

Handles:

Data ingestion (multi-ticker support)
Indicator generation (EMA, RSI, ROC, ATR, etc.)
Structural features (trend, volatility, compression)
Feature grouping (price, bounded, delta, rate, composite)
from src.DataPipeline import DataPipeline

data = DataPipeline()
data.build()
Strategy Registry

Strategies are fully configurable via JSON.

Each strategy defines:

Indicator set
Prediction horizon
Buy/sell thresholds
from src.strategies import get_strategy

strategy = get_strategy("classic_swing")
Model Pipeline

Builds full sklearn pipelines:

Imputation → Scaling → (Optional PCA) → Model
TimeSeriesSplit (no data leakage)
Multi-metric evaluation
Grid Search + Optuna support
from src.ModelPipeline import ModelPipeline

mp = ModelPipeline(model_name_="HGBC", strategy_="default")
mp.grid_search()
Experiment Runners
Single Strategy Benchmark
python -m main.gs1
Runs ONE strategy across ALL models
Uses shared dataset splits for fair comparison
Outputs:
logs/
models/
results_summary.csv
Full Strategy × Model Sweep
python -m main.gs
Runs ALL strategies × ALL models
Generates full performance matrix
📈 Example Output

After running:

python -m main.gs1
results/
  ├── logs/
  │   ├── default_HGBC.json
  │   ├── default_RFC.json
  │   └── ...
  ├── models/
  │   ├── default_HGBC_f1.joblib
  │   └── ...
  ├── results_summary.csv
  └── summary.txt
⚙️ Features
JSON-driven architecture (strategies + models)
Time-series cross-validation (TSCV)
Feature engineering pipeline with grouped preprocessing
Multi-model benchmarking framework
Structured experiment logging

Designed for:

Regime-aware modeling
Ensemble learning
Strategy comparison at scale
🧪 Models Supported
HistGradientBoosting (HGBC)
Random Forest (RFC)
Logistic Regression
Extra Trees
Support Vector Machine (SVC)
Multi-Layer Perceptron (MLP)
📦 Requirements
pandas
yfinance
python-dotenv
joblib
optuna
scikit-learn
🧱 Project Structure
stocks/
├── configs/
├── data/
├── notebooks/
├── runs/
├── scripts/
├── server/
├── src/
│   ├── core/
│   ├── features/
│   ├── registry/
│   ├── tracking/
│   ├── training/
│   └── utils/
├── tests/
├── README.md
├── requirements.txt
└── .env
🧭 Roadmap
 Market regime classification (trend / chop / volatility)
 Multi-horizon prediction targets
 Ensemble model stacking
 Multi-timeframe feature fusion
 Live trading integration (IBKR / TradingView)
⚠️ Disclaimer

This project is for research purposes only.
Not financial advice.

👤 Author

Built as part of a quantitative research and ML trading system.


---

If you want next level beyond this, I’d go:

👉 add a **results screenshot + “best model so far” section**  
👉 or turn this into a **portfolio page / recruiter version**

Just say the word 👍