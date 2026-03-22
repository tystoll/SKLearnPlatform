"""
src/core/strategy.py
====================
Feature engineering pipeline on top of a Data object.

Responsibilities:
  - Apply indicator sets defined in strategies.json
  - Build feature groups (price / bounded / delta / rate / composite)
  - Add forward-looking labels for supervised learning
  - Expose clean feature columns + labeled DataFrame for Model

Does NOT touch disk (delegate to Data).
Does NOT do ML (delegate to Model).

Module-level helpers
--------------------
generate_data_all()          — download every ticker x every timeframe
generate_data_all_times()    — download every timeframe for one ticker
generate_data_all_strategy() — download one timeframe for every ticker
"""
from __future__ import annotations
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .data import Data, TICKER_LIST, TICKER_LIST_TEST, TIMEFRAMES, DEFAULT_TICKER
from ..registry.strategies import get_strategy, ALL_STRATEGIES

# ============================================================================
# LOGGING
# ============================================================================
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  [Strategy]  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_STRATEGY  = "default"
DEFAULT_CSV_PATH  = None  # None -> Data uses SERVER_CSV_DEFAULT
# Label defaults - overridden by strategy config values at build time
DAILY_HORIZON        = 5
DAILY_BUY_THRESH     = 0.05
DAILY_SELL_THRESH    = 0.05
INTRADAY_BUY_THRESH  = 0.001
INTRADAY_SELL_THRESH = 0.001

# ============================================================================
# MODULE-LEVEL GENERATE HELPERS
# ============================================================================

def generate_data_all(ticker_list: List[str] = None,
                      sleep_seconds: int = 60) -> None:
    """Download + save all timeframes for every ticker in the list."""
    tickers = ticker_list if ticker_list is not None else TICKER_LIST
    for idx, tic in enumerate(tickers, start=1):
        logger.info("[%d/%d] %s: generating all timeframes", idx, len(tickers), tic)
        try:
            generate_data_all_times(ticker=tic)
        except Exception as e:
            logger.error("%s failed: %s", tic, e)
        if sleep_seconds > 0 and idx < len(tickers):
            time.sleep(sleep_seconds)


def generate_data_all_times(ticker: str = None,
                             sleep_seconds: int = 15) -> None:
    """Download + save all 4 canonical timeframes for a single ticker."""
    tic = ticker if ticker is not None else DEFAULT_TICKER
    for period, interval in TIMEFRAMES:
        logger.info("%s: period=%s  interval=%s", tic, period, interval)
        try:
            d = Data(ticker_=tic, period_=period, interval_=interval)
            d.download_from_yahoo()
            d.save_to_csv()
        except Exception as e:
            logger.error("%s  %s  %s  failed: %s", tic, period, interval, e)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def generate_data_all_strategy(period: str,
                                interval: str,
                                ticker_list: List[str] = None,
                                sleep_seconds: int = 15) -> None:
    """Download + save one timeframe for every ticker in the list."""
    tickers = ticker_list if ticker_list is not None else TICKER_LIST
    for idx, tic in enumerate(tickers, start=1):
        logger.info("[%d/%d] %s: period=%s  interval=%s",
                    idx, len(tickers), tic, period, interval)
        try:
            d = Data(ticker_=tic, period_=period, interval_=interval)
            d.download_from_yahoo()
            d.save_to_csv()
        except Exception as e:
            logger.error("%s  %s  %s  failed: %s", tic, period, interval, e)
        if sleep_seconds > 0 and idx < len(tickers):
            time.sleep(sleep_seconds)

# ============================================================================
# STRATEGY CLASS
# ============================================================================

class Strategy:
    """
    Feature engineering pipeline on top of a Data object.

    __init__ accepts all config options then immediately calls build(),
    which wires each param (passed value OR getattr fallback to default)
    and runs the full pipeline end-to-end.

    build() can be called again at any time to reconfigure without
    losing existing state on fields that are not re-passed.

    Full pipeline order (executed inside build):
        1. data.load_from_csv()
        2. add_indicators()
        3. build_feature_groups()
        4. add_labels()

    Usage
    -----
    # Fully automatic with defaults
    s = Strategy()

    # Specific strategy
    s = Strategy(strategy_="classic_swing")

    # Provide a pre-loaded Data object
    d = Data(ticker_="NVDA")
    d.load_from_csv("server/NVDA_daily_features.csv")
    s = Strategy(data_=d, strategy_="classic_swing")

    # Override label thresholds
    s = Strategy(strategy_="default", buy_thresh_=0.02, sell_thresh_=0.02)
    """

    def __init__(self,
                 # Data
                 data_:        Data  = None,
                 ticker_:      str   = None,
                 period_:      str   = None,
                 interval_:    str   = None,
                 csv_path_:    str   = None,
                 # Strategy
                 strategy_:    str   = None,
                 # Labels - optional overrides (otherwise read from strategy config)
                 horizon_:     int   = None,
                 buy_thresh_:  float = None,
                 sell_thresh_: float = None,
    ):
        self._feature_columns: List[str] = []
        self._feature_groups: Dict[str, List[str]] = {
            "price": [], "bounded": [], "delta": [], "rate": [], "composite": [],
        }
        self.build(
            # Data
            data=data_,
            ticker=ticker_,
            period=period_,
            interval=interval_,
            csv_path=csv_path_,
            # Strategy
            strategy=strategy_,
            # Labels
            horizon=horizon_,
            buy_thresh=buy_thresh_,
            sell_thresh=sell_thresh_,
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self,
              # Data
              data:        Data  = None,
              ticker:      str   = None,
              period:      str   = None,
              interval:    str   = None,
              csv_path:    str   = None,
              # Strategy
              strategy:    str   = None,
              # Labels
              horizon:     int   = None,
              buy_thresh:  float = None,
              sell_thresh: float = None,
    ) -> 'Strategy':
        # -- Data ------------------------------------------------------------
        if data is not None:
            self._data = data
        elif not hasattr(self, '_data') or self._data is None:
            self._data = Data(ticker_=ticker, period_=period, interval_=interval)
        else:
            # Patch individual fields on existing Data if re-calling build
            if ticker   is not None: self._data._ticker   = ticker
            if period   is not None: self._data._period   = period
            if interval is not None: self._data._interval = interval

        # -- Strategy --------------------------------------------------------
        self._strategy = strategy  if strategy  is not None else getattr(self, '_strategy', DEFAULT_STRATEGY)
        self._csv_path = csv_path  if csv_path  is not None else getattr(self, '_csv_path', DEFAULT_CSV_PATH)

        # -- Labels - prefer explicit override, then strategy config, then module default
        strat_cfg         = get_strategy(self._strategy)
        self._horizon     = horizon     if horizon     is not None else getattr(self, '_horizon',     strat_cfg.get("horizon",     DAILY_HORIZON))
        self._buy_thresh  = buy_thresh  if buy_thresh  is not None else getattr(self, '_buy_thresh',  strat_cfg.get("buy_thresh",  DAILY_BUY_THRESH))
        self._sell_thresh = sell_thresh if sell_thresh is not None else getattr(self, '_sell_thresh', strat_cfg.get("sell_thresh", DAILY_SELL_THRESH))

        # -- Execute pipeline ------------------------------------------------
        self._data.load_from_csv(self._csv_path)
        self.add_indicators()
        self.build_feature_groups()
        self.add_labels(
            horizon=self._horizon,
            buy_thresh=self._buy_thresh,
            sell_thresh=self._sell_thresh,
        )

        logger.info(
            "Strategy.build done - rows=%d  features=%d",
            len(self._data._df), len(self._feature_columns),
        )
        return self

    # ------------------------------------------------------------------
    # Convenience pass-through
    # ------------------------------------------------------------------
    @property
    def df(self) -> Optional[pd.DataFrame]:
        return self._data._df

    @df.setter
    def df(self, value: pd.DataFrame) -> None:
        self._data._df = value

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------
    def add_indicators(self, strategy_: str = None) -> 'Strategy':
        """Apply indicator set from strategy config -> self._data._df."""
        if self._data._df is None:
            raise ValueError(
                "No data loaded. Call data.load_from_csv() or "
                "data.download_from_yahoo() first."
            )

        strategy = strategy_ if strategy_ is not None else self._strategy
        strat    = get_strategy(strategy)

        if isinstance(strat, dict) and "indicators" in strat:
            indicator_list = strat["indicators"]
        elif isinstance(strat, list):
            indicator_list = strat
        else:
            raise ValueError(
                f"strategy must be a list of indicator dicts or a strategy dict "
                f"with an 'indicators' key. Got: {type(strat)}"
            )

        logger.info("Adding %d indicator(s) [strategy=%s] to %d rows",
                    len(indicator_list), strategy, len(self._data._df))
        df = self._data._df.copy()

        DISPATCH = {
            "sma":               lambda d, s: add_sma(d, period=s["period"]),
            "ema":               lambda d, s: add_ema(d, period=s["period"]),
            "tema":              lambda d, s: add_tema(d, period=s["period"]),
            "rsi":               lambda d, s: add_rsi(d, period=s["period"]),
            "momentum":          lambda d, s: add_momentum(d, period=s["period"]),
            "roc":               lambda d, s: add_roc(d, period=s["period"]),
            "volumeroc":         lambda d, s: add_volume_roc(d, period=s["period"]),
            "pvp":               lambda d, s: add_pvp(
                d,
                price_roc_col=s.get("price_roc_col",
                    f"ROC_{s.get('roc_period', s.get('period', 14))}"),
                volume_col=s.get("volume_col", "Volume"),
                period=s.get("pvp_period", s.get("period", 14)),
            ),
            "atr":               lambda d, s: add_atr(
                d, period=s.get("period", 14), col_name=s.get("col_name"),
            ),
            "accel_jerk":        lambda d, s: add_acceleration_and_jerk(
                d, price_col=s.get("price_col", "Close"),
                log_returns=s.get("log_returns", True),
            ),
            "er":                lambda d, s: add_efficiency_ratio(
                d, price_col=s.get("price_col", "Close"), period=s["period"],
            ),
            "er_two":            lambda d, s: add_efficiency_ratio_two_horizons(
                d, price_col=s.get("price_col", "Close"),
                short=s.get("short", 10), long=s.get("long", 20),
            ),
            "range_compression": lambda d, s: add_range_compression_ratio(
                d, high_col=s.get("high_col", "High"),
                low_col=s.get("low_col", "Low"), period=s.get("period", 20),
            ),
            "vol_change":        lambda d, s: add_volatility_change_features(
                d, price_col=s.get("price_col", "Close"),
                vol_window=s.get("vol_window", 20),
                change_lag=s.get("change_lag", 5),
                long_window=s.get("long_window", 50),
            ),
            "dist_extremes":     lambda d, s: add_dist_to_extremes_atr_scaled(
                d, high_col=s.get("high_col", "High"),
                low_col=s.get("low_col", "Low"),
                close_col=s.get("close_col", "Close"),
                lookback=s.get("lookback", 20),
                atr_period=s.get("atr_period", 14),
            ),
            "time_of_day":       lambda d, s: add_time_of_day_features(
                d, dt_col=s.get("dt_col", "Date"),
                tz=s.get("tz", "America/New_York"),
                add_day_of_week_onehot=s.get("add_day_of_week_onehot", True),
                add_session_flags=s.get("add_session_flags", True),
            ),
        }

        for spec in indicator_list:
            name = spec["name"].lower().strip()
            if name not in DISPATCH:
                raise ValueError(
                    f"Unknown indicator '{name}'. Supported: {sorted(DISPATCH.keys())}"
                )
            logger.debug("  + %s", spec)
            df = DISPATCH[name](df, spec)

        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if inf_count:
            logger.warning("Replaced %d Inf value(s) with NaN", inf_count)
        df = df.replace([np.inf, -np.inf], np.nan)

        if "Date" in df.columns:
            df = df.sort_values("Date", ascending=False).reset_index(drop=True)

        self._data._df = df
        logger.info("Indicators done - DataFrame now %d cols", len(df.columns))
        return self

    def add_tier1_micro_features(self,
                                  price_col: str = "Close",
                                  high_col:  str = "High",
                                  low_col:   str = "Low",
    ) -> 'Strategy':
        """Accel/jerk, ER (2 horizons), range compression,
        volatility change, ATR-scaled distance to extremes."""
        logger.info("Adding Tier-1 micro features to %d rows", len(self._data._df))
        df = self._data._df.copy()
        df = add_acceleration_and_jerk(df, price_col=price_col, log_returns=True)
        df = add_efficiency_ratio_two_horizons(df, price_col=price_col, short=10, long=20)
        df = add_range_compression_ratio(df, high_col=high_col, low_col=low_col, period=20)
        df = add_volatility_change_features(df, price_col=price_col, vol_window=20,
                                            change_lag=5, long_window=50)
        df = add_dist_to_extremes_atr_scaled(df, high_col=high_col, low_col=low_col,
                                             close_col=price_col, lookback=20, atr_period=14)
        self._data._df = df
        logger.info("Tier-1 micro features done - DataFrame now %d cols", len(df.columns))
        return self

    def add_market_structure_features(self,
                                       price_col:       str   = "Close",
                                       high_col:        str   = "High",
                                       low_col:         str   = "Low",
                                       open_col:        str   = "Open",
                                       atr_period:      int   = 14,
                                       swing_left:      int   = 2,
                                       swing_right:     int   = 2,
                                       min_swing_atr:   float = 0.0,
                                       range_n:         int   = 50,
                                       inside_n:        int   = 20,
                                       bos_fail_window: int   = 10,
    ) -> 'Strategy':
        """Full structure pipeline: ATR -> swings -> BoS/fails/sweeps
        -> range/compression -> anatomy -> structure state."""
        logger.info("Adding market structure features to %d rows", len(self._data._df))
        out = self._data._df.copy()
        out = add_atr(out, high_col=high_col, low_col=low_col, close_col=price_col,
                      period=atr_period, col_name="ATR")
        cfg = SwingConfig(left=swing_left, right=swing_right,
                          atr_col="ATR", min_swing_atr=min_swing_atr)
        out = detect_fractal_swings(out, high_col=high_col, low_col=low_col, cfg=cfg)
        out = add_last_prev_swings(out, high_col=high_col, low_col=low_col,
                                   close_col=price_col, atr_col="ATR")
        out = add_bos_and_failures(out, high_col=high_col, low_col=low_col,
                                   close_col=price_col, atr_col="ATR",
                                   fail_window=bos_fail_window)
        out = add_range_and_compression(out, high_col=high_col, low_col=low_col,
                                        close_col=price_col, atr_col="ATR",
                                        n=range_n, inside_n=inside_n)
        out = add_bar_anatomy(out, open_col=open_col, high_col=high_col,
                              low_col=low_col, close_col=price_col)
        out = add_structure_state(out, lookback_swings=6)
        self._data._df = out
        logger.info("Market structure done - DataFrame now %d cols", len(out.columns))
        return self

    # ------------------------------------------------------------------
    # Feature groups
    # ------------------------------------------------------------------
    def build_feature_groups(self,
                              use_price:     bool = True,
                              use_bounded:   bool = True,
                              use_delta:     bool = True,
                              use_rate:      bool = True,
                              use_composite: bool = True,
    ) -> 'Strategy':
        """Bucket numeric columns into feature groups and populate _feature_columns."""
        if self._data._df is None:
            raise ValueError("No data. Load data first.")

        self._feature_groups = {
            "price": [], "bounded": [], "delta": [], "rate": [], "composite": []
        }

        for col in self._data._df.columns:
            if use_price and col.startswith((
                "SMA_", "EMA_", "TEMA_", "WMA_",
                "swing_high", "swing_low",
                "last_swing_high", "prev_swing_high",
                "last_swing_low",  "prev_swing_low",
                "range_high", "range_low",
            )):
                self._feature_groups["price"].append(col)

            elif use_bounded and col.startswith((
                "RSI_", "STOCH_", "WILLR_", "ER_",
                "dist_to_last_high_atr", "dist_to_last_low_atr",
                "dist_to_range_high_atr", "dist_to_range_low_atr",
                "close_position_in_bar", "upper_wick_ratio", "lower_wick_ratio",
                "body_ratio", "rejection_wick_ratio",
                "range_compression_ratio", "overlap_ratio", "trend_consistency",
            )):
                self._feature_groups["bounded"].append(col)

            elif use_delta and col.startswith((
                "Momentum_", "Delta_", "r1", "accel", "jerk",
                "ER_slope", "vol_change",
                "swing_high_delta", "swing_low_delta", "swing_range",
                "range_width", "bos_distance_atr",
            )):
                self._feature_groups["delta"].append(col)

            elif use_rate and (
                col.startswith((
                    "ROC_", "VolumeROC_", "range_comp_ratio_20",
                    "std_r", "vol_ratio", "ATR",
                    "dist_to_high_20_atr", "dist_to_low_20_atr",
                ))
                or col.endswith(("_ret", "_logret"))
            ):
                self._feature_groups["rate"].append(col)

            elif use_composite and col.startswith((
                "PVP_", "Composite_",
                "swing_high_idx", "swing_low_idx",
                "is_HH", "is_HL", "is_LH", "is_LL",
                "bos_up", "bos_down",
                "failed_bos_up", "failed_bos_down",
                "liquidity_sweep_high", "liquidity_sweep_low",
                "inside_bar_count", "structure_trend", "structure_age",
                "bars_since_swing_high", "bars_since_swing_low",
            )):
                self._feature_groups["composite"].append(col)

        self._feature_columns = [
            c for cols in self._feature_groups.values() for c in cols
        ]

        logger.info(
            "Feature groups: price=%d  bounded=%d  delta=%d  rate=%d  composite=%d  total=%d",
            len(self._feature_groups["price"]),
            len(self._feature_groups["bounded"]),
            len(self._feature_groups["delta"]),
            len(self._feature_groups["rate"]),
            len(self._feature_groups["composite"]),
            len(self._feature_columns),
        )

        _skip = {
            "Open", "High", "Low", "Close", "Volume", "Date", "Ticker",
            "range_hl", "rolling_max_high_20", "rolling_min_low_20",
            "Close_fwd", "y_fwd_ret", "y_fwd_logret", "y_up", "y_class_3",
        }
        unbucketed = [
            c for c in self._data._df.select_dtypes(include=[np.number]).columns
            if c not in self._feature_columns and c not in _skip
        ]
        if unbucketed:
            logger.warning(
                "%d numeric col(s) not assigned to any feature group: %s",
                len(unbucketed), unbucketed,
            )

        return self

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------
    def add_labels(self,
                   horizon:     int   = None,
                   buy_thresh:  float = None,
                   sell_thresh: float = None,
                   price_col:   str   = "Close",
                   date_col:    str   = "Date",
                   ticker_col:  str   = "Ticker",
    ) -> 'Strategy':
        """
        Add forward-looking labels for supervised learning.

        Creates
        -------
        Close_fwd    : forward close price
        y_fwd_ret    : forward simple return
        y_fwd_logret : forward log return
        y_up         : binary 1/0
        y_class_3    : 3-class  1=buy  0=hold  -1=sell
        """
        if self._data._df is None:
            raise ValueError("No data loaded.")

        h  = horizon     if horizon     is not None else self._horizon
        bt = buy_thresh  if buy_thresh  is not None else self._buy_thresh
        st = sell_thresh if sell_thresh is not None else self._sell_thresh

        df = self._data._df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values([ticker_col, date_col]).reset_index(drop=True)

        df["Close_fwd"]    = df.groupby(ticker_col)[price_col].shift(-h)
        df["y_fwd_ret"]    = (df["Close_fwd"] / df[price_col]) - 1.0
        df["y_fwd_logret"] = np.log(df["Close_fwd"] / df[price_col])
        df["y_up"]         = (df["y_fwd_ret"] > 0).astype(int)
        df["y_class_3"]    = np.select(
            [df["y_fwd_ret"] >= bt, df["y_fwd_ret"] <= -st],
            [1, -1],
            default=0,
        ).astype(int)

        df = df.dropna(
            subset=["Close_fwd", "y_fwd_ret", "y_fwd_logret"]
        ).reset_index(drop=True)

        dist = df["y_class_3"].value_counts().sort_index().to_dict()
        logger.info(
            "Labels added - horizon=%d  buy_thresh=%.4f  sell_thresh=%.4f  dist=%s",
            h, bt, st, dist,
        )
        self._data._df = df
        return self

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------
    def get_features(self, dropna: bool = False) -> pd.DataFrame:
        """Return the feature DataFrame (X)."""
        if not self._feature_columns:
            raise ValueError("No features. Run build_feature_groups() first.")
        df = self._data._df.copy()
        if dropna:
            df = df.dropna(subset=self._feature_columns)
        return df[self._feature_columns]

    def get_labels(self, col: str = "y_class_3") -> pd.Series:
        """Return a label column (y)."""
        if col not in self._data._df.columns:
            raise ValueError(
                f"Label column '{col}' not found. Run add_labels() first. "
                f"Available: {[c for c in self._data._df.columns if c.startswith('y_')]}"
            )
        return self._data._df[col]

    def __repr__(self) -> str:
        rows = len(self._data._df) if self._data._df is not None else 0
        return (
            f"Strategy(strategy={self._strategy!r}, "
            f"data={self._data!r}, "
            f"features={len(self._feature_columns)}, rows={rows})"
        )

# ============================================================================
# INDICATOR FUNCTIONS  (pure - no side effects)
# ============================================================================

def add_sma(df: pd.DataFrame, price_col: str = "Close", period: int = 20,
            min_periods: Optional[int] = None,
            col_name: Optional[str] = None) -> pd.DataFrame:
    if min_periods is None: min_periods = period
    if col_name is None:    col_name = f"SMA_{period}"
    df[col_name] = df[price_col].rolling(window=period, min_periods=min_periods).mean()
    return df

def add_ema(df: pd.DataFrame, price_col: str = "Close", period: int = 15,
            min_periods: Optional[int] = None, col_name: Optional[str] = None,
            adjust: bool = False) -> pd.DataFrame:
    if min_periods is None: min_periods = period
    if col_name is None:    col_name = f"EMA_{period}"
    df[col_name] = df[price_col].ewm(span=period, adjust=adjust, min_periods=min_periods).mean()
    return df

def add_tema(df: pd.DataFrame, price_col: str = "Close", period: int = 15,
             min_periods: Optional[int] = None, col_name: Optional[str] = None,
             adjust: bool = False) -> pd.DataFrame:
    if min_periods is None: min_periods = period
    if col_name is None:    col_name = f"TEMA_{period}"
    ema1 = df[price_col].ewm(span=period, adjust=adjust, min_periods=min_periods).mean()
    ema2 = ema1.ewm(span=period, adjust=adjust, min_periods=min_periods).mean()
    ema3 = ema2.ewm(span=period, adjust=adjust, min_periods=min_periods).mean()
    df[col_name] = 3 * ema1 - 3 * ema2 + ema3
    return df

def add_rsi(df: pd.DataFrame, price_col: str = "Close", period: int = 14,
            min_periods: Optional[int] = None,
            col_name: Optional[str] = None) -> pd.DataFrame:
    if min_periods is None: min_periods = period
    if col_name is None:    col_name = f"RSI_{period}"
    delta    = df[price_col].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=min_periods).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=min_periods).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[col_name] = 100 - (100 / (1 + rs))
    return df

def add_momentum(df: pd.DataFrame, price_col: str = "Close", period: int = 14,
                 col_name: Optional[str] = None) -> pd.DataFrame:
    if col_name is None: col_name = f"Momentum_{period}"
    df[col_name] = df[price_col] - df[price_col].shift(period)
    return df

def add_roc(df: pd.DataFrame, price_col: str = "Close", period: int = 14,
            col_name: Optional[str] = None, pct: bool = True) -> pd.DataFrame:
    if col_name is None: col_name = f"ROC_{period}"
    roc = df[price_col] / df[price_col].shift(period) - 1.0
    df[col_name] = roc * 100.0 if pct else roc
    return df

def add_volume_roc(df: pd.DataFrame, volume_col: str = "Volume", period: int = 14,
                   col_name: Optional[str] = None, pct: bool = True) -> pd.DataFrame:
    if col_name is None: col_name = f"VolumeROC_{period}"
    vol = df[volume_col]
    roc = vol / vol.shift(period) - 1.0
    df[col_name] = roc * 100.0 if pct else roc
    return df

def add_pvp(df: pd.DataFrame, price_roc_col: str, volume_col: str = "Volume",
            period: int = 14, min_periods: Optional[int] = None,
            col_name: Optional[str] = None) -> pd.DataFrame:
    if min_periods is None: min_periods = period
    if col_name is None:    col_name = f"PVP_{period}"
    rel_vol = df[volume_col] / df[volume_col].rolling(period, min_periods=min_periods).mean()
    df[col_name] = df[price_roc_col] * rel_vol
    return df

def _get_dt_series(df: pd.DataFrame, dt_col: Optional[str] = None,
                   tz: str = "America/New_York") -> pd.Series:
    dt = (pd.to_datetime(df[dt_col], errors="coerce") if dt_col
          else pd.to_datetime(df.index, errors="coerce"))
    return (dt.dt.tz_convert(tz) if getattr(dt.dt, "tz", None)
            else dt.dt.tz_localize(tz))

def _true_range(df: pd.DataFrame, high_col: str,
                low_col: str, close_col: str) -> pd.Series:
    pc = df[close_col].shift(1)
    return pd.concat([
        (df[high_col] - df[low_col]).abs(),
        (df[high_col] - pc).abs(),
        (df[low_col]  - pc).abs(),
    ], axis=1).max(axis=1)

def add_atr(df: pd.DataFrame, high_col: str = "High", low_col: str = "Low",
            close_col: str = "Close", period: int = 14,
            min_periods: Optional[int] = None,
            col_name: Optional[str] = None) -> pd.DataFrame:
    if min_periods is None: min_periods = period
    if col_name is None:    col_name = f"ATR_{period}"
    tr = _true_range(df, high_col=high_col, low_col=low_col, close_col=close_col)
    df[col_name] = tr.rolling(period, min_periods=min_periods).mean()
    return df

def add_acceleration_and_jerk(df: pd.DataFrame, price_col: str = "Close",
                               log_returns: bool = True, eps: float = 1e-12,
                               r1_col: str = "r1", accel_col: str = "accel",
                               jerk_col: str = "jerk") -> pd.DataFrame:
    p = df[price_col]
    if log_returns:
        df[r1_col] = np.log((p + eps) / (p.shift(1) + eps))
    else:
        df[r1_col] = (p / (p.shift(1) + eps)) - 1.0
    df[accel_col] = df[r1_col] - df[r1_col].shift(1)
    df[jerk_col]  = df[accel_col] - df[accel_col].shift(1)
    return df

def add_efficiency_ratio(df: pd.DataFrame, price_col: str = "Close",
                          period: int = 20, min_periods: Optional[int] = None,
                          col_name: Optional[str] = None,
                          eps: float = 1e-12) -> pd.DataFrame:
    if min_periods is None: min_periods = period
    if col_name is None:    col_name = f"ER_{period}"
    change     = (df[price_col] - df[price_col].shift(period)).abs()
    volatility = df[price_col].diff().abs().rolling(period, min_periods=min_periods).sum()
    df[col_name] = change / (volatility + eps)
    return df

def add_efficiency_ratio_two_horizons(df: pd.DataFrame, price_col: str = "Close",
                                       short: int = 10, long: int = 20,
                                       eps: float = 1e-12, add_slope: bool = True,
                                       slope_col: str = "ER_slope") -> pd.DataFrame:
    df = add_efficiency_ratio(df, price_col=price_col, period=short, eps=eps, col_name=f"ER_{short}")
    df = add_efficiency_ratio(df, price_col=price_col, period=long,  eps=eps, col_name=f"ER_{long}")
    if add_slope:
        df[slope_col] = df[f"ER_{short}"] - df[f"ER_{long}"]
    return df

def add_range_compression_ratio(df: pd.DataFrame, high_col: str = "High",
                                 low_col: str = "Low", period: int = 20,
                                 min_periods: Optional[int] = None,
                                 range_col: str = "range_hl",
                                 ratio_col: Optional[str] = None,
                                 eps: float = 1e-12) -> pd.DataFrame:
    if min_periods is None: min_periods = period
    if ratio_col is None:   ratio_col = f"range_comp_ratio_{period}"
    df[range_col] = (df[high_col] - df[low_col]).abs()
    roll_mean     = df[range_col].rolling(period, min_periods=min_periods).mean()
    df[ratio_col] = df[range_col] / (roll_mean + eps)
    return df

def add_volatility_change_features(df: pd.DataFrame, price_col: str = "Close",
                                    vol_window: int = 20, change_lag: int = 5,
                                    long_window: int = 50, log_returns: bool = True,
                                    eps: float = 1e-12,
                                    std_col: Optional[str] = None,
                                    std_long_col: Optional[str] = None,
                                    vol_change_col: str = "vol_change",
                                    vol_ratio_col:  str = "vol_ratio") -> pd.DataFrame:
    if std_col is None:      std_col = f"std_r{vol_window}"
    if std_long_col is None: std_long_col = f"std_r{long_window}"
    p = df[price_col]
    r = (np.log((p + eps) / (p.shift(1) + eps)) if log_returns
         else (p / (p.shift(1) + eps)) - 1.0)
    df[std_col]        = r.rolling(vol_window,  min_periods=vol_window).std()
    df[std_long_col]   = r.rolling(long_window, min_periods=long_window).std()
    df[vol_change_col] = df[std_col] - df[std_col].shift(change_lag)
    df[vol_ratio_col]  = df[std_col] / (df[std_long_col] + eps)
    return df

def add_dist_to_extremes_atr_scaled(df: pd.DataFrame, high_col: str = "High",
                                     low_col: str = "Low", close_col: str = "Close",
                                     lookback: int = 20, atr_period: int = 14,
                                     atr_col: Optional[str] = None,
                                     dist_high_col: Optional[str] = None,
                                     dist_low_col: Optional[str] = None,
                                     eps: float = 1e-12) -> pd.DataFrame:
    if atr_col is None:       atr_col = f"ATR_{atr_period}"
    if dist_high_col is None: dist_high_col = f"dist_to_high_{lookback}_atr"
    if dist_low_col is None:  dist_low_col  = f"dist_to_low_{lookback}_atr"
    df = add_atr(df, high_col=high_col, low_col=low_col, close_col=close_col,
                 period=atr_period, col_name=atr_col)
    roll_max = df[high_col].rolling(lookback, min_periods=lookback).max()
    roll_min = df[low_col].rolling(lookback,  min_periods=lookback).min()
    df[f"rolling_max_high_{lookback}"] = roll_max
    df[f"rolling_min_low_{lookback}"]  = roll_min
    df[dist_high_col] = (df[close_col] - roll_max) / (df[atr_col] + eps)
    df[dist_low_col]  = (df[close_col] - roll_min) / (df[atr_col] + eps)
    return df

def add_time_of_day_features(df: pd.DataFrame, dt_col: Optional[str] = None,
                              tz: str = "America/New_York",
                              add_day_of_week_onehot: bool = True,
                              add_session_flags: bool = True,
                              include_extended: bool = False,
                              session:    Tuple[str, str] = ("09:30", "16:00"),
                              premarket:  Tuple[str, str] = ("04:00", "09:30"),
                              afterhours: Tuple[str, str] = ("16:00", "20:00"),
                              prefix: str = "") -> pd.DataFrame:
    dt  = _get_dt_series(df, dt_col=dt_col, tz=tz)
    mod = dt.dt.hour * 60 + dt.dt.minute
    rad = 2.0 * np.pi * (mod / 1440.0)
    df[f"{prefix}sin_minute_of_day"] = np.sin(rad)
    df[f"{prefix}cos_minute_of_day"] = np.cos(rad)
    if add_day_of_week_onehot:
        dow = dt.dt.dayofweek
        for i in range(5):
            df[f"{prefix}dow_{i}"] = (dow == i).astype(int)
    if add_session_flags:
        tstr = dt.dt.strftime("%H:%M")
        def _between(t, s, e): return (t >= s) & (t < e)
        rs, re = session
        df[f"{prefix}is_rth"]        = _between(tstr, rs, re).astype(int)
        df[f"{prefix}is_opening"]    = _between(tstr, rs, "10:00").astype(int)
        df[f"{prefix}is_lunch"]      = _between(tstr, "11:30", "13:30").astype(int)
        df[f"{prefix}is_power_hour"] = _between(tstr, "15:00", re).astype(int)
        if include_extended:
            df[f"{prefix}is_premarket"]  = _between(tstr, *premarket).astype(int)
            df[f"{prefix}is_afterhours"] = _between(tstr, *afterhours).astype(int)
    return df

# ============================================================================
# SWING DETECTION & MARKET STRUCTURE
# ============================================================================

@dataclass
class SwingConfig:
    left:          int   = 2
    right:         int   = 2
    atr_col:       str   = "ATR"
    min_swing_atr: float = 0.0


def detect_fractal_swings(df: pd.DataFrame, high_col: str = "High",
                           low_col: str = "Low",
                           cfg: SwingConfig = SwingConfig()) -> pd.DataFrame:
    df  = df.copy()
    L, R = cfg.left, cfg.right
    h, l = df[high_col], df[low_col]
    win  = L + R + 1
    roll_max = h.rolling(win, center=True, min_periods=win).max()
    roll_min = l.rolling(win, center=True, min_periods=win).min()
    shi = (h == roll_max) & (~(h == roll_max).shift(1).fillna(False))
    sli = (l == roll_min) & (~(l == roll_min).shift(1).fillna(False))
    df["swing_high_idx"] = shi
    df["swing_low_idx"]  = sli
    df["swing_high"]     = np.where(shi, h, np.nan)
    df["swing_low"]      = np.where(sli, l, np.nan)
    if cfg.min_swing_atr > 0:
        if cfg.atr_col not in df.columns:
            raise ValueError(f"ATR column '{cfg.atr_col}' not found. Call add_atr first.")
        atr = df[cfg.atr_col]
        last_high = last_low = np.nan
        keep_high = np.zeros(len(df), dtype=bool)
        keep_low  = np.zeros(len(df), dtype=bool)
        for i in range(len(df)):
            a = atr.iat[i]
            if np.isnan(a) or a == 0:
                if shi.iat[i]: keep_high[i] = True
                if sli.iat[i]: keep_low[i]  = True
                continue
            if shi.iat[i]:
                p = h.iat[i]
                if np.isnan(last_low) or (p - last_low) >= cfg.min_swing_atr * a:
                    keep_high[i] = True;  last_high = p
            if sli.iat[i]:
                p = l.iat[i]
                if np.isnan(last_high) or (last_high - p) >= cfg.min_swing_atr * a:
                    keep_low[i] = True;  last_low = p
        df["swing_high_idx"] = keep_high
        df["swing_low_idx"]  = keep_low
        df["swing_high"]     = np.where(keep_high, h, np.nan)
        df["swing_low"]      = np.where(keep_low,  l, np.nan)
    return df


def add_last_prev_swings(df: pd.DataFrame, high_col: str = "High",
                          low_col: str = "Low", close_col: str = "Close",
                          atr_col: str = "ATR") -> pd.DataFrame:
    df = df.copy()
    sh, sl = df["swing_high"], df["swing_low"]
    df["last_swing_high"] = sh.ffill()
    df["prev_swing_high"] = sh.where(sh.notna()).shift(1).ffill()
    df["last_swing_low"]  = sl.ffill()
    df["prev_swing_low"]  = sl.where(sl.notna()).shift(1).ffill()
    idx = np.arange(len(df))
    lhi = pd.Series(np.where(sh.notna(), idx, np.nan), index=df.index).ffill()
    lli = pd.Series(np.where(sl.notna(), idx, np.nan), index=df.index).ffill()
    df["bars_since_swing_high"] = idx - lhi.values
    df["bars_since_swing_low"]  = idx - lli.values
    if atr_col in df.columns:
        atr = df[atr_col].replace(0, np.nan)
        df["dist_to_last_high_atr"] = (df[close_col] - df["last_swing_high"]) / atr
        df["dist_to_last_low_atr"]  = (df[close_col] - df["last_swing_low"])  / atr
    else:
        df["dist_to_last_high_atr"] = df["dist_to_last_low_atr"] = np.nan
    df["is_HH"] = (df["last_swing_high"] > df["prev_swing_high"]) & sh.notna()
    df["is_LH"] = (df["last_swing_high"] < df["prev_swing_high"]) & sh.notna()
    df["is_HL"] = (df["last_swing_low"]  > df["prev_swing_low"])  & sl.notna()
    df["is_LL"] = (df["last_swing_low"]  < df["prev_swing_low"])  & sl.notna()
    df["swing_high_delta"] = df["last_swing_high"] - df["prev_swing_high"]
    df["swing_low_delta"]  = df["last_swing_low"]  - df["prev_swing_low"]
    df["swing_range"]      = df["last_swing_high"] - df["last_swing_low"]
    if atr_col in df.columns:
        df["swing_range_atr"] = df["swing_range"] / df[atr_col].replace(0, np.nan)
    else:
        df["swing_range_atr"] = np.nan
    return df


def add_bos_and_failures(df: pd.DataFrame, high_col: str = "High",
                          low_col: str = "Low", close_col: str = "Close",
                          atr_col: str = "ATR", fail_window: int = 10) -> pd.DataFrame:
    df = df.copy()
    if "last_swing_high" not in df.columns:
        raise ValueError("Run add_last_prev_swings() first.")
    lh, ll = df["last_swing_high"], df["last_swing_low"]
    df["bos_up"]               = df[close_col] > lh
    df["bos_down"]             = df[close_col] < ll
    df["liquidity_sweep_high"] = (df[high_col] > lh) & (df[close_col] <= lh)
    df["liquidity_sweep_low"]  = (df[low_col]  < ll) & (df[close_col] >= ll)
    if atr_col in df.columns:
        atr    = df[atr_col].replace(0, np.nan)
        broken = np.where(df["bos_up"], lh, np.where(df["bos_down"], ll, np.nan))
        df["bos_distance_atr"] = (df[close_col] - broken) / atr
    else:
        df["bos_distance_atr"] = np.nan
    bos_up_ev   = df["bos_up"]   & (~df["bos_up"].shift(1).fillna(False))
    bos_down_ev = df["bos_down"] & (~df["bos_down"].shift(1).fillna(False))
    fup = np.zeros(len(df), dtype=bool)
    fdn = np.zeros(len(df), dtype=bool)
    close = df[close_col].values
    lhv, llv = lh.values, ll.values
    for i in np.where(bos_up_ev.values)[0]:
        j = min(len(df) - 1, i + fail_window)
        if np.any(close[i+1:j+1] <= lhv[i]): fup[i] = True
    for i in np.where(bos_down_ev.values)[0]:
        j = min(len(df) - 1, i + fail_window)
        if np.any(close[i+1:j+1] >= llv[i]): fdn[i] = True
    df["failed_bos_up"]   = fup
    df["failed_bos_down"] = fdn
    return df


def add_range_and_compression(df: pd.DataFrame, high_col: str = "High",
                               low_col: str = "Low", close_col: str = "Close",
                               atr_col: str = "ATR",
                               n: int = 50, inside_n: int = 20) -> pd.DataFrame:
    df    = df.copy()
    rh    = df[high_col].rolling(n, min_periods=n).max()
    rl    = df[low_col].rolling(n,  min_periods=n).min()
    width = rh - rl
    df["range_high"]  = rh
    df["range_low"]   = rl
    df["range_width"] = width
    if atr_col in df.columns:
        atr = df[atr_col].replace(0, np.nan)
        df["range_width_atr"]        = width / atr
        df["dist_to_range_high_atr"] = (df[close_col] - rh) / atr
        df["dist_to_range_low_atr"]  = (df[close_col] - rl) / atr
    else:
        df["range_width_atr"] = df["dist_to_range_high_atr"] = df["dist_to_range_low_atr"] = np.nan
    df["range_compression_ratio"] = width / width.rolling(n, min_periods=n).mean()
    inside  = (df[high_col] <= df[high_col].shift(1)) & (df[low_col] >= df[low_col].shift(1))
    overlap = (df[low_col]  <= df[high_col].shift(1)) & (df[high_col] >= df[low_col].shift(1))
    df["inside_bar_count"] = inside.rolling(inside_n,  min_periods=inside_n).sum()
    df["overlap_ratio"]    = overlap.rolling(inside_n, min_periods=inside_n).mean()
    return df


def add_bar_anatomy(df: pd.DataFrame, open_col: str = "Open",
                    high_col: str = "High", low_col: str = "Low",
                    close_col: str = "Close") -> pd.DataFrame:
    df = df.copy()
    o, h, l, c = df[open_col], df[high_col], df[low_col], df[close_col]
    rng        = (h - l).replace(0, np.nan)
    body       = (c - o).abs()
    upper_wick = h - np.maximum(o, c)
    lower_wick = np.minimum(o, c) - l
    df["close_position_in_bar"] = (c - l) / rng
    df["upper_wick_ratio"]      = upper_wick / rng
    df["lower_wick_ratio"]      = lower_wick / rng
    df["body_ratio"]            = body / rng
    df["rejection_wick_ratio"]  = np.maximum(upper_wick, lower_wick) / rng
    return df


def add_structure_state(df: pd.DataFrame, lookback_swings: int = 6) -> pd.DataFrame:
    df  = df.copy()
    event       = df["swing_high"].notna() | df["swing_low"].notna()
    up_evidence = (df["is_HH"] | df["is_HL"]) & event
    dn_evidence = (df["is_LL"] | df["is_LH"]) & event
    state_event = np.where(up_evidence & ~dn_evidence, 1,
                   np.where(dn_evidence & ~up_evidence, -1, 0))
    state = (pd.Series(np.where(event, state_event, np.nan), index=df.index)
             .ffill().fillna(0).astype(int))
    df["structure_trend"] = state
    consistency = np.full(len(df), np.nan)
    event_idx   = np.where(event.values)[0]
    for k_i, i in enumerate(event_idx):
        window = event_idx[max(0, k_i - lookback_swings + 1): k_i + 1]
        cur    = state.iat[i]
        if cur == 0:
            consistency[i] = 0.0
        else:
            w     = state_event[window]
            valid = (w != 0)
            consistency[i] = (w[valid] == cur).mean() if valid.sum() > 0 else 0.0
    df["trend_consistency"] = pd.Series(consistency, index=df.index).ffill().fillna(0.0)
    flip = ((df["structure_trend"] != df["structure_trend"].shift(1)) &
            (df["structure_trend"] != 0))
    last_flip = pd.Series(
        np.where(flip.fillna(False), np.arange(len(df)), np.nan),
        index=df.index,
    ).ffill()
    df["structure_age"] = np.arange(len(df)) - last_flip.values
    return df

# ============================================================================
# MAIN  (smoke tests)
# ============================================================================
if __name__ == "__main__":
    RUN_CODE = 0

    match RUN_CODE:
        case 0:
            # No args - full defaults, auto end-to-end
            s = Strategy()
            print(s)
            print(f"\nLabel distribution:\n{s._data._df['y_class_3'].value_counts().sort_index()}")
        case 1:
            # Specific strategy
            s = Strategy(strategy_="classic_swing")
            print(s)
        case 2:
            # Pre-built Data passed in
            d = Data()
            d.load_from_csv()
            s = Strategy(data_=d, strategy_="fib_swing")
            print(s)
        case 3:
            # Override label thresholds
            s = Strategy(strategy_="default", buy_thresh_=0.02, sell_thresh_=0.02)
            print(f"\nLabel distribution:\n{s._data._df['y_class_3'].value_counts().sort_index()}")
        case 4:
            # Every registered strategy
            for name in ALL_STRATEGIES:
                s = Strategy(strategy_=name)
                print(s)
        case 5:
            generate_data_all_times()
        case 6:
            generate_data_all()
        case 7:
            for period, interval in TIMEFRAMES:
                generate_data_all_strategy(period=period, interval=interval)
