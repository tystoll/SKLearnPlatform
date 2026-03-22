"""
src/core/data.py
================
Thin I/O wrapper for a single ticker + timeframe.

Responsibilities:
  - Download raw OHLCV from Yahoo Finance
  - Persist / load raw CSV to disk
  - Save / load fitted sklearn models via joblib
  - Expose a clean _df for Strategy to consume

Does NOT do indicators, feature engineering, labels, or any ML.
All of that lives in Strategy and Model.
"""
from __future__ import annotations
import logging
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from typing import Any, Optional

# src/core/data.py  →  parents[0]=core/  parents[1]=src/  parents[2]=stocks/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data"
MODEL_DIR    = PROJECT_ROOT / "models"
SERVER_DIR   = PROJECT_ROOT / "server"

# Default CSV to load when no path is given.
# Swap this out once data/ is populated from the pipeline.
SERVER_CSV_DEFAULT = SERVER_DIR / "AAPL_daily_features.csv"

# ============================================================================
# LOGGING
# ============================================================================
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  [Data]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_TICKER   = "AAPL"
DEFAULT_PERIOD   = "max"
DEFAULT_INTERVAL = "1d"

TICKER_LIST_TEST = ["AAPL", "SPY", "QQQ"]

TICKER_LIST = [
    "AAPL", "SPY",  "QQQ",  "NVDA", "BBAI", "TSLA", "ONDS",
    "PLUG", "OPEN", "AAL",  "RIVN", "BMNR", "AVGO", "K",
    "IREN", "CLSK", "INTC", "CIFR", "PFE",  "MARA", "DNN",
    "ORCL", "F",    "SOFI", "APLD", "WULF", "SNAP", "NU",
]

TIMEFRAMES = [
    ("7d",  "1m"),
    ("max", "1h"),
    ("max", "1d"),
    ("max", "1mo"),
]

# ============================================================================
# DATA CLASS
# ============================================================================
class Data:
    """
    Thin I/O wrapper for a single ticker + timeframe.

    Usage
    -----
    # Download and save raw data
    d = Data(ticker_="NVDA", period_="max", interval_="1d")
    d.download_from_yahoo().save_to_csv()

    # Load existing CSV (defaults to server/AAPL_daily_features.csv for now)
    d = Data()
    d.load_from_csv()

    # Load a specific file
    d.load_from_csv("server/NVDA_daily_features.csv")
    """

    def __init__(self,
                 ticker_:   str = None,
                 period_:   str = None,
                 interval_: str = None,
    ):
        self._ticker   = ticker_   if ticker_   is not None else DEFAULT_TICKER
        self._period   = period_   if period_   is not None else DEFAULT_PERIOD
        self._interval = interval_ if interval_ is not None else DEFAULT_INTERVAL
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Properties / paths
    # ------------------------------------------------------------------
    @property
    def ticker(self) -> str:
        return self._ticker

    @property
    def period(self) -> str:
        return self._period

    @property
    def interval(self) -> str:
        return self._interval

    @property
    def csv_path(self) -> Path:
        """Canonical raw CSV: data/<TICKER>_<PERIOD>_<INTERVAL>_raw.csv"""
        return DATA_DIR / f"{self._ticker}_{self._period}_{self._interval}_raw.csv"

    def model_path(self, name: str) -> Path:
        """Canonical model artifact: models/<name>.joblib"""
        return MODEL_DIR / f"{name}.joblib"

    # ------------------------------------------------------------------
    # Yahoo Finance
    # ------------------------------------------------------------------
    def download_from_yahoo(self,
                            ticker:   str = None,
                            period:   str = None,
                            interval: str = None,
    ) -> 'Data':
        """Download OHLCV from Yahoo Finance → self._df. Returns self."""
        t = ticker   if ticker   is not None else self._ticker
        p = period   if period   is not None else self._period
        i = interval if interval is not None else self._interval

        logger.info("Downloading %s  period=%s  interval=%s", t, p, i)
        df = yf.download(
            tickers=t,
            period=p,
            interval=i,
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            raise ValueError(
                f"yfinance returned empty DataFrame for "
                f"ticker='{t}'  period='{p}'  interval='{i}'. "
                "Check ticker symbol and that the market was open."
            )

        # yfinance sometimes returns MultiIndex columns even for a single ticker
        if isinstance(df.columns, pd.MultiIndex):
            sym = df.columns.get_level_values(1).unique()[0]
            df = df.xs(sym, level=1, axis=1)

        df.columns = [c.replace(" ", "_") for c in df.columns]
        df = df.reset_index(drop=False)

        # yfinance returns 'Datetime' for intraday intervals, 'Date' for daily+
        date_src = next((c for c in ("Datetime", "Date") if c in df.columns), None)
        if date_src is None:
            raise ValueError(f"No date column found. Got: {list(df.columns)}")

        df["Ticker"] = t.upper()
        df["Date"] = (
            pd.to_datetime(df[date_src], utc=True)
            .dt.tz_localize(None)
            .dt.floor("min")
        )
        df = df.sort_values("Date", ascending=False).reset_index(drop=True)
        if date_src != "Date":
            df.drop(columns=[date_src], inplace=True)

        self._df = df
        logger.info("Downloaded %d rows × %d cols  (%s)", len(df), len(df.columns), t)
        return self

    # ------------------------------------------------------------------
    # CSV I/O
    # ------------------------------------------------------------------
    def save_to_csv(self, path: str = None) -> 'Data':
        """Write self._df to CSV. Defaults to canonical csv_path."""
        dest = Path(path) if path is not None else self.csv_path
        if self._df is None:
            raise ValueError("No data to save. Call download_from_yahoo() first.")
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(dest, index=False)
        logger.info("Saved %d rows → %s", len(self._df), dest)
        return self

    def load_from_csv(self, path: str = None) -> 'Data':
        """
        Read CSV into self._df.

        Default path is SERVER_CSV_DEFAULT (server/AAPL_daily_features.csv).
        Pass an explicit path to load any other file.
        """
        src = Path(path) if path is not None else SERVER_CSV_DEFAULT
        if not src.exists():
            raise FileNotFoundError(
                f"CSV not found: {src}\n"
                f"Hint: run download_from_yahoo().save_to_csv() first, "
                f"or pass an explicit path to load_from_csv()."
            )
        df = pd.read_csv(src, parse_dates=["Date"], date_format="%Y-%m-%d %H:%M")
        if df.empty:
            raise ValueError(f"CSV '{src}' loaded as empty DataFrame.")
        self._df = df
        logger.info("Loaded %d rows × %d cols  ← %s", len(df), len(df.columns), src)
        return self

    # ------------------------------------------------------------------
    # Model I/O  (joblib)
    # ------------------------------------------------------------------
    def save_model(self, model: Any, name: str, path: str = None) -> Path:
        """Persist a fitted sklearn estimator. Returns the save path."""
        dest = Path(path) if path is not None else self.model_path(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, dest)
        logger.info("Model saved → %s", dest)
        return dest

    def load_model(self, name: str, path: str = None) -> Any:
        """Load a persisted sklearn estimator."""
        src = Path(path) if path is not None else self.model_path(name)
        if not src.exists():
            raise FileNotFoundError(f"Model not found: {src}")
        model = joblib.load(src)
        logger.info("Model loaded ← %s", src)
        return model

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        rows = len(self._df) if self._df is not None else 0
        return (
            f"Data(ticker={self._ticker!r}, period={self._period!r}, "
            f"interval={self._interval!r}, rows={rows})"
        )
