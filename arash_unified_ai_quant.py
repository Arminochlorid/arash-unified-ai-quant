from __future__ import annotations

"""
ARASH UNIFIED AI QUANT SYSTEM
=============================

Purpose
-------
A research + decision engine that keeps the user's existing trading systems as
separate setup families and lets ML learn *when each family works* instead of
flattening them into one OR-combined signal.

Implemented source families
---------------------------
- ARASH_MR_LONG / ARASH_MR_SHORT
- ARASH_TREND_LONG / ARASH_TREND_SHORT
- QUANT_CROSS_LONG / QUANT_CROSS_SHORT
- EMA_BUY / EMA_SELL
- QTE_SCALPER_LONG / QTE_SCALPER_SHORT

Context/features
----------------
- ArashArrow: session VWAP/sigma/Z, ADR, Stoch RSI, Kaufman ER, normalized
  VWAP slope, Z velocity/acceleration, band expansion, acceptance, rejection,
  reversal/continuation pressure, trend score, regime and mode.
- Quant engine: momentum Z, mean-reversion Z, vol Z, quant alpha Z,
  distribution Z, volatility event flags, VWAP distance, lag correlation.
- BOS/CHOCH: causal pivot confirmation, BOS/CHOCH events, inducement flags,
  active demand/supply zones, zone distances, mitigation/break events.
- Buy/Sell EMA timing and QTE scalper timing.
- External timing features: input columns named timing_*, signal_*, ext_*.

AI design
---------
- Candidate setups remain separate rows with a setup_family field.
- Every event is labelled from the NEXT BAR OPEN using a causal triple barrier.
- Labels include first-touch result, realized R, MFE/MAE and label_end_time.
- Walk-forward training is purged: an event can only be in training when its
  label_end_time is earlier than the OOS test start.
- One ML classifier estimates P(positive R); one regressor estimates expected R.
- Setup family, symbol, asset class, timeframe and deterministic market regime
  are categorical model inputs.
- Decision layer picks the highest-EV candidate only if thresholds are met.

This module does not claim an edge without real historical / forward data.
Synthetic tests verify mechanics only.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =============================================================================
# CONFIG
# =============================================================================

@dataclass(frozen=True)
class FeatureConfig:
    adr_length: int = 14
    rsi_length: int = 14
    stoch_length: int = 14
    k_smooth: int = 3
    d_smooth: int = 3
    momentum_lookback: int = 4
    reversal_lookback: int = 10
    er_length: int = 20
    er_trend_threshold: float = 0.55
    er_mean_threshold: float = 0.35
    slope_length: int = 8
    slope_trend_threshold: float = 0.25
    slope_flat_threshold: float = 0.15
    mr_start_z: float = 2.0
    mr_hot_score: float = 65.0
    mr_go_score: float = 80.0
    trend_hot_score: float = 65.0
    trend_go_score: float = 80.0
    trend_entry_min_z: float = 0.30
    trend_entry_max_z: float = 1.80

    # V3 pressure block
    velocity_length: int = 3
    band_expansion_length: int = 5
    acceptance_length: int = 12
    acceptance_z: float = 2.0
    velocity_scale: float = 0.20
    acceleration_scale: float = 0.15
    expansion_scale: float = 0.08
    rev_watch: float = 45.0
    rev_go: float = 65.0
    cont_warning: float = 50.0
    cont_block: float = 65.0

    # Quant engine
    quant_length: int = 20
    smile_length: int = 50
    exec_length: int = 30

    # Buy/Sell
    ema_fast_len: int = 5
    ema_slow_len: int = 13
    buy_sell_atr_len: int = 14
    buy_sell_atr_mult: float = 0.5
    buy_sell_rr: float = 3.0
    buy_sell_confirm_candle: bool = True

    # QTE scalper, matching pasted defaults
    qte_ema_period: int = 10
    qte_min_atr: float = 1.0
    qte_touch_type: str = "Body Touch"  # Body Touch | Wick Touch | No Touch

    # Structure
    bos_lookbacks: Tuple[int, ...] = (1, 2, 3, 5, 11, 15, 20)

    # Shared ATR
    atr_length: int = 14


@dataclass(frozen=True)
class OutcomeConfig:
    horizon_bars: int = 20
    stop_atr: float = 1.0
    target_atr: float = 1.5
    ambiguous_policy: str = "exclude"  # exclude | stop_first


@dataclass(frozen=True)
class AIConfig:
    train_days: int = 180
    test_days: int = 30
    min_train_events: int = 500
    min_probability: float = 0.58
    min_expected_r: float = 0.10
    max_iter: int = 250
    learning_rate: float = 0.05
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 30
    l2_regularization: float = 1.0
    random_state: int = 42


@dataclass(frozen=True)
class ExecutionConfig:
    point_value: float = 1.0
    tick_size: float = 0.01
    commission_per_side: float = 0.0
    slippage_ticks: float = 0.0
    initial_capital: float = 50_000.0

    @property
    def cost_per_side(self) -> float:
        return float(self.commission_per_side + self.slippage_ticks * self.tick_size * self.point_value)


# =============================================================================
# PINE-LIKE PRIMITIVES
# =============================================================================

def _validate_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    req = ["open", "high", "low", "close"]
    missing = [c for c in req if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}")
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be DatetimeIndex")
    if raw.index.has_duplicates:
        raise ValueError("DatetimeIndex contains duplicates")
    df = raw.sort_index().copy()
    for c in req:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[req].isna().any().any():
        raise ValueError("OHLC contains NaN/non-numeric values")
    if (df["high"] < df[["open", "close"]].max(axis=1)).any():
        raise ValueError("high below open/close")
    if (df["low"] > df[["open", "close"]].min(axis=1)).any():
        raise ValueError("low above open/close")
    if (df["high"] < df["low"]).any():
        raise ValueError("high below low")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    return df


def _pine_sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _pine_ema(s: pd.Series, n: int) -> pd.Series:
    """Pine ta.ema style recursive EMA seeded with first finite source value."""
    x = pd.to_numeric(s, errors="coerce").to_numpy(float)
    out = np.full(len(x), np.nan)
    alpha = 2.0 / (n + 1.0)
    prev = np.nan
    for i, v in enumerate(x):
        if not np.isfinite(v):
            out[i] = prev
            continue
        if not np.isfinite(prev):
            prev = v
        else:
            prev = alpha * v + (1.0 - alpha) * prev
        out[i] = prev
    return pd.Series(out, index=s.index)


def _pine_rma(s: pd.Series, n: int) -> pd.Series:
    """Pine ta.rma: SMA seed after n values, then alpha=1/n recursion."""
    x = pd.to_numeric(s, errors="coerce").to_numpy(float)
    out = np.full(len(x), np.nan)
    alpha = 1.0 / n
    window: List[float] = []
    prev = np.nan
    for i, v in enumerate(x):
        if not np.isfinite(prev):
            if np.isfinite(v):
                window.append(float(v))
            if len(window) == n:
                prev = float(np.mean(window))
                out[i] = prev
        else:
            if np.isfinite(v):
                prev = alpha * v + (1.0 - alpha) * prev
            out[i] = prev
    return pd.Series(out, index=s.index)


def _pine_rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    au = _pine_rma(up, n)
    ad = _pine_rma(dn, n)
    rs = au / ad.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi[(ad == 0) & (au > 0)] = 100.0
    rsi[(au == 0) & (ad > 0)] = 0.0
    rsi[(au == 0) & (ad == 0)] = 50.0
    return rsi


def _pine_atr(df: pd.DataFrame, n: int) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - pc).abs(),
        (df["low"] - pc).abs(),
    ], axis=1).max(axis=1)
    return _pine_rma(tr, n)


def _safe_div(a: pd.Series, b: pd.Series, fill=np.nan) -> pd.Series:
    out = a.astype(float) / b.astype(float).replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill)


def _pine_stdev(s: pd.Series, n: int) -> pd.Series:
    # Pine ta.stdev defaults to biased=true => population standard deviation.
    return s.rolling(n, min_periods=n).std(ddof=0)


def _bars_since(cond: pd.Series) -> pd.Series:
    out = np.full(len(cond), np.nan)
    last = None
    arr = cond.fillna(False).to_numpy(bool)
    for i, v in enumerate(arr):
        if v:
            last = i
            out[i] = 0
        elif last is not None:
            out[i] = i - last
    return pd.Series(out, index=cond.index)


def _cross_over(a: pd.Series, b: pd.Series | float) -> pd.Series:
    bb = pd.Series(float(b), index=a.index) if np.isscalar(b) else b
    return (a > bb) & (a.shift(1) <= bb.shift(1))


def _cross_under(a: pd.Series, b: pd.Series | float) -> pd.Series:
    bb = pd.Series(float(b), index=a.index) if np.isscalar(b) else b
    return (a < bb) & (a.shift(1) >= bb.shift(1))


def _session_vwap_sigma(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Anchored daily VWAP and volume-weighted population sigma of hlc3."""
    src = (df["high"] + df["low"] + df["close"]) / 3.0
    day = pd.Series(df.index.normalize(), index=df.index)
    if "volume" in df.columns and (df["volume"] > 0).any():
        w = df["volume"].clip(lower=0.0).astype(float)
    else:
        # Without volume a parity claim to Pine VWAP is impossible. Equal weights
        # keep the feature usable for research, and parity audit marks the issue.
        w = pd.Series(1.0, index=df.index)
    cw = w.groupby(day).cumsum()
    cwx = (src * w).groupby(day).cumsum()
    cwx2 = (src.pow(2) * w).groupby(day).cumsum()
    vwap = cwx / cw.replace(0.0, np.nan)
    var = (cwx2 / cw.replace(0.0, np.nan) - vwap.pow(2)).clip(lower=0.0)
    sigma = np.sqrt(var)
    return vwap, sigma, day


def _directional_adr(df: pd.DataFrame, z: pd.Series, day: pd.Series, n: int) -> Tuple[pd.Series, pd.Series]:
    daily = df.groupby(day).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"))
    daily["range"] = daily["high"] - daily["low"]
    # Pine array uses prior completed days only.
    daily["adr"] = daily["range"].shift(1).rolling(n, min_periods=1).mean()
    adr = day.map(daily["adr"]).astype(float)
    dopen = df["open"].groupby(day).transform("first")
    dhigh = df["high"].groupby(day).cummax()
    dlow = df["low"].groupby(day).cummin()
    directional = pd.Series(np.where(z >= 0, dhigh - dopen, dopen - dlow), index=df.index)
    adr_pct = _safe_div(directional, adr, fill=np.nan) * 100.0
    return adr, adr_pct


# =============================================================================
# ARASHARROW PORT
# =============================================================================

def arasharrow_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    vwap, sigma, day = _session_vwap_sigma(df)
    z = _safe_div(df["close"] - vwap, sigma, fill=np.nan)
    abs_z = z.abs()
    adr, adr_pct = _directional_adr(df, z, day, cfg.adr_length)

    rsi = _pine_rsi(df["close"], cfg.rsi_length)
    low_rsi = rsi.rolling(cfg.stoch_length, min_periods=cfg.stoch_length).min()
    high_rsi = rsi.rolling(cfg.stoch_length, min_periods=cfg.stoch_length).max()
    stoch_raw = 100.0 * _safe_div(rsi - low_rsi, high_rsi - low_rsi, fill=0.5)
    k = _pine_sma(stoch_raw, cfg.k_smooth)
    d = _pine_sma(k, cfg.d_smooth)
    cross_up = _cross_over(k, d)
    cross_dn = _cross_under(k, d)
    bs_up = _bars_since(cross_up)
    bs_dn = _bars_since(cross_dn)
    recent_up = bs_up.le(cfg.momentum_lookback)
    recent_dn = bs_dn.le(cfg.momentum_lookback)

    er_dir = (df["close"] - df["close"].shift(cfg.er_length)).abs()
    er_noise = df["close"].diff().abs().rolling(cfg.er_length, min_periods=cfg.er_length).sum()
    er = _safe_div(er_dir, er_noise, fill=np.nan).clip(0.0, 1.0)
    slope = _safe_div(vwap - vwap.shift(cfg.slope_length), sigma, fill=np.nan)
    abs_slope = slope.abs()
    trend_regime = (er >= cfg.er_trend_threshold) & (abs_slope >= cfg.slope_trend_threshold)
    mean_regime = (er <= cfg.er_mean_threshold) | (abs_slope <= cfg.slope_flat_threshold)

    # Original ArashArrow scoring
    mr_band = pd.Series(0.0, index=df.index)
    m = abs_z >= cfg.mr_start_z
    mr_band.loc[m] = (10.0 + (abs_z.loc[m] - cfg.mr_start_z) * 30.0).clip(0, 40)
    mr_adr = (((adr_pct - 50.0) / 50.0) * 20.0).clip(0, 20).fillna(0.0)
    mr_mom = pd.Series(0.0, index=df.index)
    longside = z < 0
    shortside = z > 0
    mr_mom += ((longside & (k <= 10)).astype(float) * 12.0)
    mr_mom += ((longside & (k > 10) & (k <= 20)).astype(float) * 7.0)
    mr_mom += ((longside & recent_up).astype(float) * 8.0)
    mr_mom += ((shortside & (k >= 90)).astype(float) * 12.0)
    mr_mom += ((shortside & (k < 90) & (k >= 80)).astype(float) * 7.0)
    mr_mom += ((shortside & recent_dn).astype(float) * 8.0)
    mr_mom = mr_mom.clip(0, 20)
    mr_regime = pd.Series(0.0, index=df.index)
    mr_regime += np.where(er <= cfg.er_mean_threshold, 10.0, np.where(er < cfg.er_trend_threshold, 5.0, 0.0))
    mr_regime += np.where(abs_slope <= cfg.slope_flat_threshold, 10.0, np.where(abs_slope < cfg.slope_trend_threshold, 5.0, 0.0))
    mr_score = (mr_band + mr_adr + mr_mom + mr_regime).clip(0, 100)
    mr_score = mr_score.where(abs_z >= cfg.mr_start_z, 0.0)

    trend_er = ((er - cfg.er_trend_threshold) / (1.0 - cfg.er_trend_threshold) * 30.0).clip(0, 30).fillna(0.0)
    trend_slope = ((abs_slope - cfg.slope_trend_threshold) / 0.75 * 30.0).clip(0, 30).fillna(0.0)
    trend_long_aligned = (z > 0) & (slope > 0)
    trend_short_aligned = (z < 0) & (slope < 0)
    trend_location = pd.Series(0.0, index=df.index)
    trend_location.loc[(abs_z >= cfg.trend_entry_min_z) & (abs_z <= cfg.trend_entry_max_z)] = 20.0
    trend_location.loc[(abs_z > cfg.trend_entry_max_z) & (abs_z <= 2.20)] = 8.0
    trend_mom = pd.Series(0.0, index=df.index)
    trend_mom.loc[trend_long_aligned & (k > d) & (k >= 55)] = 20.0
    trend_mom.loc[trend_long_aligned & (k > d) & (k < 55)] = 10.0
    trend_mom.loc[trend_short_aligned & (k < d) & (k <= 45)] = 20.0
    trend_mom.loc[trend_short_aligned & (k < d) & (k > 45)] = 10.0
    trend_score = (trend_er + trend_slope + trend_location + trend_mom).clip(0, 100)
    trend_score = trend_score.where(trend_regime & (trend_long_aligned | trend_short_aligned), 0.0)

    # V3 REV / CONT pressure block
    raw_zvel = z - z.shift(1)
    zvel = _pine_sma(raw_zvel.fillna(0.0), cfg.velocity_length).fillna(0.0)
    zacc = zvel - zvel.shift(1).fillna(zvel)
    dvel = pd.Series(np.where(z >= 0, zvel, -zvel), index=df.index)
    dacc = pd.Series(np.where(z >= 0, zacc, -zacc), index=df.index)
    sigchg = _safe_div(sigma - sigma.shift(1), sigma.shift(1), fill=0.0)
    expansion = _pine_sma(sigchg, cfg.band_expansion_length).fillna(0.0)
    acceptance = (abs_z >= cfg.acceptance_z).astype(float).rolling(cfg.acceptance_length, min_periods=1).mean() * 100.0

    cr = (df["high"] - df["low"]).replace(0.0, np.nan)
    upper_w = (df["high"] - df[["open", "close"]].max(axis=1)).clip(lower=0)
    lower_w = (df[["open", "close"]].min(axis=1) - df["low"]).clip(lower=0)
    upper_pct = _safe_div(upper_w, cr, fill=0.0) * 100
    lower_pct = _safe_div(lower_w, cr, fill=0.0) * 100
    close_from_high = _safe_div(df["high"] - df["close"], cr, fill=0.0) * 100
    close_from_low = _safe_div(df["close"] - df["low"], cr, fill=0.0) * 100
    short_rej = upper_pct * 0.60 + close_from_high * 0.40
    long_rej = lower_pct * 0.60 + close_from_low * 0.40
    rejection = pd.Series(np.where(z >= 0, short_rej, long_rej), index=df.index).clip(0, 100)

    vel_rev = pd.Series(np.where(dvel < 0, np.minimum(np.abs(dvel) / cfg.velocity_scale * 20.0, 20.0), 0.0), index=df.index)
    acc_rev = pd.Series(np.where(dacc < 0, np.minimum(np.abs(dacc) / cfg.acceleration_scale * 15.0, 15.0), 0.0), index=df.index)
    mom_rev = (((z < 0) & recent_up) | ((z > 0) & recent_dn)).astype(float) * 15.0
    reversal = (vel_rev + acc_rev + rejection / 100 * 25 + mom_rev + (1 - er.fillna(1.0)) * 15 + (100 - acceptance) / 100 * 10).clip(0, 100)

    slope_aligned = ((z > 0) & (slope > 0)) | ((z < 0) & (slope < 0))
    continuation = ((er.fillna(0.0) - .25) / .55 * 25).clip(0, 25)
    continuation += pd.Series(np.where(slope_aligned, np.minimum(abs_slope.fillna(0.0) / .75 * 20, 20), 0), index=df.index)
    continuation += pd.Series(np.where(dvel > 0, np.minimum(dvel / cfg.velocity_scale * 20, 20), 0), index=df.index)
    continuation += pd.Series(np.where(dacc > 0, np.minimum(dacc / cfg.acceleration_scale * 10, 10), 0), index=df.index)
    continuation += (acceptance / 100 * 15).clip(0, 15)
    continuation += pd.Series(np.where(expansion > 0, np.minimum(expansion / cfg.expansion_scale * 10, 10), 0), index=df.index)
    continuation = continuation.clip(0, 100)

    # Original mode logic (preserve as context)
    mode = pd.Series(0, index=df.index, dtype=int)
    final_score = pd.Series(0.0, index=df.index)
    nodata = z.isna()
    risk_up = trend_regime & (z >= cfg.mr_start_z) & (slope > 0)
    risk_dn = trend_regime & (z <= -cfg.mr_start_z) & (slope < 0)
    mr_zone = (abs_z >= cfg.mr_start_z) & ~trend_regime
    tr_l = trend_regime & trend_long_aligned & abs_z.between(cfg.trend_entry_min_z, cfg.trend_entry_max_z)
    tr_s = trend_regime & trend_short_aligned & abs_z.between(cfg.trend_entry_min_z, cfg.trend_entry_max_z)
    mode[risk_up] = 5
    mode[risk_dn] = 6
    mode[mr_zone & (z < 0)] = 1
    mode[mr_zone & (z > 0)] = 2
    mode[tr_l & ~(risk_up | risk_dn | mr_zone)] = 3
    mode[tr_s & ~(risk_up | risk_dn | mr_zone)] = 4
    mode[nodata] = -99
    final_score.loc[risk_up | risk_dn] = pd.concat([trend_score, mr_score], axis=1).max(axis=1).loc[risk_up | risk_dn]
    final_score.loc[mr_zone] = mr_score.loc[mr_zone]
    final_score.loc[tr_l | tr_s] = trend_score.loc[tr_l | tr_s]
    final_score.loc[nodata] = np.nan

    out["arash_vwap"] = vwap
    out["arash_sigma"] = sigma
    out["arash_z"] = z
    out["arash_abs_z"] = abs_z
    out["arash_adr"] = adr
    out["arash_adr_pct"] = adr_pct
    out["arash_rsi"] = rsi
    out["arash_stoch_k"] = k
    out["arash_stoch_d"] = d
    out["arash_efficiency_ratio"] = er
    out["arash_vwap_slope_sigma"] = slope
    out["arash_trend_regime"] = trend_regime.astype(float)
    out["arash_mean_regime"] = mean_regime.astype(float)
    out["arash_mr_score"] = mr_score
    out["arash_trend_score"] = trend_score
    out["arash_trend_long_aligned"] = trend_long_aligned.astype(float)
    out["arash_trend_short_aligned"] = trend_short_aligned.astype(float)
    out["arash_z_velocity"] = zvel
    out["arash_z_acceleration"] = zacc
    out["arash_band_expansion"] = expansion
    out["arash_acceptance"] = acceptance
    out["arash_rejection"] = rejection
    out["arash_reversal_pressure"] = reversal
    out["arash_continuation_pressure"] = continuation
    out["arash_mode"] = mode
    out["arash_final_score"] = final_score
    return out


# =============================================================================
# QUANT ENGINE PORT
# =============================================================================

def quant_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    q = cfg.quant_length
    momentum_raw = (df["close"] - df["close"].shift(q)) / df["close"].shift(q) * 100.0
    momentum_ma = _pine_sma(momentum_raw, q)
    momentum_std = _pine_stdev(momentum_raw, q)
    momentum_z = _safe_div(momentum_raw - momentum_ma, momentum_std, fill=0.0)

    price_ma = _pine_sma(df["close"], q * 2)
    mr_raw = _safe_div(df["close"] - price_ma, price_ma, fill=0.0) * 100.0
    mr_std = _pine_stdev(mr_raw, q)
    mr_z = _safe_div(mr_raw, mr_std, fill=0.0)

    vol_realized = _pine_stdev(df["close"].diff(), q) / df["close"] * 100.0
    vol_ma = _pine_sma(vol_realized, q)
    vol_z = _safe_div(vol_realized - vol_ma, vol_ma, fill=0.0)

    ema_trend = _pine_ema(df["close"], 50)
    is_down = df["close"] < ema_trend
    is_up = df["close"] > ema_trend
    raw_mr_signal = -mr_z
    mr_contrib = raw_mr_signal.copy()
    mr_contrib.loc[is_down] = np.minimum(raw_mr_signal.loc[is_down], 0.0)
    mr_contrib.loc[is_up] = np.maximum(raw_mr_signal.loc[is_up], 0.0)
    alpha = momentum_z * 0.5 + mr_contrib * 0.4 + (-vol_z * 0.1)
    alpha_std = _pine_stdev(alpha, q)
    alpha_z = _safe_div(alpha, alpha_std, fill=0.0)

    smile_std = _pine_stdev(df["close"], cfg.smile_length)
    smile_sma = _pine_sma(df["close"], cfg.smile_length)
    price_z = _safe_div(df["close"] - smile_sma, smile_std, fill=0.0)
    price_move = (df["close"] - df["close"].shift(1)).abs() / df["close"].shift(1) * 100.0
    vol_short = _pine_stdev(df["close"].diff(), 5) / df["close"] * 100.0

    # Quant source uses ta.vwap(close). For daily intraday data use session VWAP of close.
    day = pd.Series(df.index.normalize(), index=df.index)
    if "volume" in df.columns and (df["volume"] > 0).any():
        w = df["volume"].clip(lower=0).astype(float)
    else:
        w = pd.Series(1.0, index=df.index)
    cw = w.groupby(day).cumsum()
    vwap_close = (df["close"] * w).groupby(day).cumsum() / cw.replace(0, np.nan)
    price_vs_vwap = _safe_div(df["close"] - vwap_close, vwap_close, fill=0.0) * 100.0

    ret = df["close"].diff()
    corr1 = ret.rolling(cfg.exec_length).corr(ret.shift(1))
    corr5 = ret.rolling(cfg.exec_length).corr(ret.shift(5))
    corr = (corr1 + corr5) / 2.0

    out["quant_momentum_z"] = momentum_z
    out["quant_mr_z"] = mr_z
    out["quant_vol_z"] = vol_z
    out["quant_mr_contribution"] = mr_contrib
    out["quant_alpha"] = alpha
    out["quant_alpha_z"] = alpha_z
    out["quant_buy_signal"] = _cross_over(alpha_z, 0.0).astype(float)
    out["quant_sell_signal"] = _cross_under(alpha_z, 0.0).astype(float)
    out["quant_price_z"] = price_z
    out["quant_smile_extreme"] = (price_z.abs() > 2.0).astype(float)
    out["quant_exotic_opportunity"] = ((price_move > 1.0) & (vol_short > vol_ma * 1.5)).astype(float)
    out["quant_exotic_risk"] = ((price_move > 2.0) & (vol_short > vol_ma * 2.0)).astype(float)
    out["quant_price_vs_vwap_pct"] = price_vs_vwap
    out["quant_corr_structure"] = corr
    out["quant_corr_broken"] = (corr.abs() < 0.2).astype(float)
    return out


# =============================================================================
# BUY/SELL PORT
# =============================================================================

def buy_sell_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    ef = _pine_ema(df["close"], cfg.ema_fast_len)
    es = _pine_ema(df["close"], cfg.ema_slow_len)
    atr = _pine_atr(df, cfg.buy_sell_atr_len)
    bull = ef > es
    bear = ef < es
    change = bull != bull.shift(1)
    if cfg.buy_sell_confirm_candle:
        buy = bull & change & (df["close"] > df["open"])
        sell = bear & change & (df["close"] < df["open"])
    else:
        buy = bull & change
        sell = bear & change
    out["bs_ema_fast"] = ef
    out["bs_ema_slow"] = es
    out["bs_ema_spread_atr"] = _safe_div(ef - es, atr, fill=0.0)
    out["bs_buy_signal"] = buy.astype(float)
    out["bs_sell_signal"] = sell.astype(float)
    return out


# =============================================================================
# QTE SCALPER PORT
# =============================================================================

def qte_scalper_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    ema = _pine_ema(df["close"], cfg.qte_ema_period)
    atr10 = _pine_atr(df, 10)
    up_basic = (
        (df["high"] < df["high"].shift(1)) &
        (df["low"] < df["low"].shift(1)) &
        (df["open"] < df["close"]) &
        (df["close"] > ema) &
        (df["close"].shift(1) > ema.shift(1)) &
        (atr10 > cfg.qte_min_atr)
    )
    dn_basic = (
        (df["high"] > df["high"].shift(1)) &
        (df["low"] > df["low"].shift(1)) &
        (df["open"] > df["close"]) &
        (df["close"] < ema) &
        (df["close"].shift(1) < ema.shift(1)) &
        (atr10 > cfg.qte_min_atr)
    )
    body_touch = ((df["open"] < ema) & (df["close"] > ema)) | ((df["open"] > ema) & (df["close"] < ema))
    wick_up = (df["low"] <= ema) & (df["close"] > ema) & (df["open"] > ema)
    wick_dn = (df["high"] >= ema) & (df["close"] < ema) & (df["open"] < ema)
    no_up = df["low"] > ema
    no_dn = df["high"] < ema

    if cfg.qte_touch_type == "Body Touch":
        up = up_basic & (body_touch | wick_up | no_up)
        dn = dn_basic & (body_touch | wick_dn | no_dn)
    elif cfg.qte_touch_type == "Wick Touch":
        up = up_basic & (wick_up | no_up)
        dn = dn_basic & (wick_dn | no_dn)
    elif cfg.qte_touch_type == "No Touch":
        up = up_basic & no_up
        dn = dn_basic & no_dn
    else:
        raise ValueError("qte_touch_type must be Body Touch, Wick Touch or No Touch")

    out["qte_ema"] = ema
    out["qte_atr10"] = atr10
    out["qte_ema_distance_atr"] = _safe_div(df["close"] - ema, atr10, fill=0.0)
    out["qte_up_signal"] = up.astype(float)
    out["qte_down_signal"] = dn.astype(float)
    return out


# =============================================================================
# BOS / CHOCH PORT WITH ACTIVE ZONES
# =============================================================================

def _is_pivot_high(hi: np.ndarray, p: int, lb: int) -> bool:
    w = hi[p-lb:p+lb+1]
    return len(w) == 2*lb+1 and np.isfinite(w).all() and hi[p] == np.max(w)


def _is_pivot_low(lo: np.ndarray, p: int, lb: int) -> bool:
    w = lo[p-lb:p+lb+1]
    return len(w) == 2*lb+1 and np.isfinite(w).all() and lo[p] == np.min(w)


def structure_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    n = len(df)
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    op = df["open"].to_numpy(float)
    cl = df["close"].to_numpy(float)
    atr = _pine_atr(df, cfg.atr_length).to_numpy(float)

    cols = {
        "bos_bull_count": np.zeros(n), "bos_bear_count": np.zeros(n),
        "choch_bull_count": np.zeros(n), "choch_bear_count": np.zeros(n),
        "idm_bull_count": np.zeros(n), "idm_bear_count": np.zeros(n),
        "demand_mitigation_count": np.zeros(n), "supply_mitigation_count": np.zeros(n),
        "demand_break_count": np.zeros(n), "supply_break_count": np.zeros(n),
        "active_demand_count": np.zeros(n), "active_supply_count": np.zeros(n),
        "nearest_demand_dist_atr": np.full(n, np.nan), "nearest_supply_dist_atr": np.full(n, np.nan),
    }

    # State per lookback, mirroring the Pine arrays.
    states = []
    for lb in cfg.bos_lookbacks:
        states.append({
            "lb": int(lb), "last_hi": np.nan, "prev_hi": np.nan, "last_lo": np.nan, "prev_lo": np.nan,
            "last_hi_bar": None, "last_lo_bar": None,
            "last_hi_ohlc": None, "last_lo_ohlc": None,
        })
    demand: List[Dict[str, Any]] = []
    supply: List[Dict[str, Any]] = []

    for t in range(n):
        # A pivot at p=t-lb becomes knowable at t, exactly because right bars now exist.
        for st in states:
            lb = st["lb"]
            p = t - lb
            if p >= lb and p + lb == t:
                if _is_pivot_high(hi, p, lb):
                    st["prev_hi"] = st["last_hi"]
                    st["last_hi"] = hi[p]
                    st["last_hi_bar"] = p
                    st["last_hi_ohlc"] = (op[p], hi[p], lo[p], cl[p])
                if _is_pivot_low(lo, p, lb):
                    st["prev_lo"] = st["last_lo"]
                    st["last_lo"] = lo[p]
                    st["last_lo_bar"] = p
                    st["last_lo_ohlc"] = (op[p], hi[p], lo[p], cl[p])

            lh, ll = st["last_hi"], st["last_lo"]
            ph, pl = st["prev_hi"], st["prev_lo"]
            if not (np.isfinite(lh) and np.isfinite(ll)) or t == 0:
                continue
            bull_break = cl[t] > lh and hi[t-1] < lh
            bear_break = cl[t] < ll and lo[t-1] > ll
            if bull_break:
                is_choch = np.isfinite(ph) and lh < ph
                is_bos = (not np.isfinite(ph)) or lh > ph
                idm = np.isfinite(pl) and ll < pl
                if is_choch:
                    cols["choch_bull_count"][t] += 1
                elif is_bos:
                    cols["bos_bull_count"][t] += 1
                if idm:
                    cols["idm_bull_count"][t] += 1
                if st["last_lo_ohlc"] is not None:
                    o, h, l, c = st["last_lo_ohlc"]
                    demand.append({"top": max(o, c), "bottom": l, "mitigated": False, "lb": st["lb"]})
            if bear_break:
                is_choch = np.isfinite(pl) and ll > pl
                is_bos = (not np.isfinite(pl)) or ll < pl
                idm = np.isfinite(ph) and lh > ph
                if is_choch:
                    cols["choch_bear_count"][t] += 1
                elif is_bos:
                    cols["bos_bear_count"][t] += 1
                if idm:
                    cols["idm_bear_count"][t] += 1
                if st["last_hi_ohlc"] is not None:
                    o, h, l, c = st["last_hi_ohlc"]
                    supply.append({"top": h, "bottom": min(o, c), "mitigated": False, "lb": st["lb"]})

        # Manage zones after creation, matching break-before-mitigation order.
        new_demand = []
        for z in demand:
            if lo[t] < z["bottom"]:
                cols["demand_break_count"][t] += 1
                continue
            if not z["mitigated"] and lo[t] <= z["top"]:
                z["mitigated"] = True
                cols["demand_mitigation_count"][t] += 1
            new_demand.append(z)
        demand = new_demand[-500:]

        new_supply = []
        for z in supply:
            if hi[t] > z["top"]:
                cols["supply_break_count"][t] += 1
                continue
            if not z["mitigated"] and hi[t] >= z["bottom"]:
                z["mitigated"] = True
                cols["supply_mitigation_count"][t] += 1
            new_supply.append(z)
        supply = new_supply[-500:]

        cols["active_demand_count"][t] = len(demand)
        cols["active_supply_count"][t] = len(supply)
        a = atr[t]
        if np.isfinite(a) and a > 0:
            if demand:
                dists = [min(abs(cl[t]-z["top"]), abs(cl[t]-z["bottom"])) / a for z in demand]
                cols["nearest_demand_dist_atr"][t] = min(dists)
            if supply:
                dists = [min(abs(cl[t]-z["top"]), abs(cl[t]-z["bottom"])) / a for z in supply]
                cols["nearest_supply_dist_atr"][t] = min(dists)

    out = pd.DataFrame(cols, index=df.index)
    out["structure_bias"] = out["bos_bull_count"] + out["choch_bull_count"] - out["bos_bear_count"] - out["choch_bear_count"]
    return out


# =============================================================================
# UNIFIED FEATURE FRAME + REGIME
# =============================================================================

def build_features(raw: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    df = _validate_ohlcv(raw)
    out = pd.DataFrame(index=df.index)
    atr = _pine_atr(df, cfg.atr_length)
    out["atr"] = atr
    out["atr_pct"] = _safe_div(atr, df["close"], fill=0.0) * 100
    out["bar_range_atr"] = _safe_div(df["high"] - df["low"], atr, fill=0.0)
    out["body_atr"] = _safe_div((df["close"] - df["open"]).abs(), atr, fill=0.0)
    out["ret_1"] = df["close"].pct_change()
    out["ret_3"] = df["close"].pct_change(3)
    out["ret_10"] = df["close"].pct_change(10)

    out = out.join(arasharrow_features(df, cfg))
    out = out.join(quant_features(df, cfg))
    out = out.join(buy_sell_features(df, cfg))
    out = out.join(qte_scalper_features(df, cfg))
    out = out.join(structure_features(df, cfg))

    if "volume" in df.columns:
        vm = df["volume"].rolling(50, min_periods=10).median()
        out["relative_volume"] = _safe_div(df["volume"], vm, fill=0.0)
        vmean = df["volume"].rolling(50, min_periods=50).mean()
        vsd = df["volume"].rolling(50, min_periods=50).std(ddof=0)
        out["volume_z"] = _safe_div(df["volume"] - vmean, vsd, fill=0.0)
    else:
        out["relative_volume"] = 0.0
        out["volume_z"] = 0.0

    minutes = df.index.hour * 60 + df.index.minute
    out["tod_sin"] = np.sin(2*np.pi*minutes/1440.0)
    out["tod_cos"] = np.cos(2*np.pi*minutes/1440.0)
    dow = df.index.dayofweek
    out["dow_sin"] = np.sin(2*np.pi*dow/7.0)
    out["dow_cos"] = np.cos(2*np.pi*dow/7.0)

    for c in df.columns:
        if c.startswith(("timing_", "signal_", "ext_")):
            out[c] = pd.to_numeric(df[c], errors="coerce")

    # Deterministic regime is context, not a trade rule.
    reg = pd.Series("CHOP", index=df.index, dtype=object)
    tr = out["arash_trend_regime"] > 0
    reg.loc[tr & (out["arash_vwap_slope_sigma"] > 0)] = "TREND_UP"
    reg.loc[tr & (out["arash_vwap_slope_sigma"] < 0)] = "TREND_DOWN"
    mr = out["arash_mean_regime"] > 0
    reg.loc[mr & (out["arash_z"] >= cfg.mr_start_z)] = "MR_EXTREME_UP"
    reg.loc[mr & (out["arash_z"] <= -cfg.mr_start_z)] = "MR_EXTREME_DOWN"
    reg.loc[out["quant_exotic_risk"] > 0] = "VOL_EXPANSION"
    out["regime"] = reg
    return out.replace([np.inf, -np.inf], np.nan)


# =============================================================================
# SETUP FAMILIES: NO OR-COMBINATION
# =============================================================================

def _rising(cond: pd.Series) -> pd.Series:
    c = cond.fillna(False).astype(bool)
    return c & ~c.shift(1, fill_value=False)


def build_setup_events(raw: pd.DataFrame,
                       features: pd.DataFrame,
                       symbol: str = "UNKNOWN",
                       asset_class: str = "UNKNOWN",
                       timeframe: str = "UNKNOWN",
                       cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    df = _validate_ohlcv(raw)
    f = features.reindex(df.index)
    specs: List[Tuple[str, int, pd.Series]] = []

    # Arash families from original score/regime logic; use threshold crossing / rising event.
    mr_long = (f["arash_z"] <= -cfg.mr_start_z) & (f["arash_mr_score"] >= cfg.mr_go_score) & ~(f["arash_mode"] == 6)
    mr_short = (f["arash_z"] >= cfg.mr_start_z) & (f["arash_mr_score"] >= cfg.mr_go_score) & ~(f["arash_mode"] == 5)
    tr_long = (f["arash_trend_regime"] > 0) & (f["arash_trend_long_aligned"] > 0) & f["arash_abs_z"].between(cfg.trend_entry_min_z, cfg.trend_entry_max_z) & (f["arash_trend_score"] >= cfg.trend_go_score)
    tr_short = (f["arash_trend_regime"] > 0) & (f["arash_trend_short_aligned"] > 0) & f["arash_abs_z"].between(cfg.trend_entry_min_z, cfg.trend_entry_max_z) & (f["arash_trend_score"] >= cfg.trend_go_score)
    specs += [
        ("ARASH_MR_LONG", 1, _rising(mr_long)),
        ("ARASH_MR_SHORT", -1, _rising(mr_short)),
        ("ARASH_TREND_LONG", 1, _rising(tr_long)),
        ("ARASH_TREND_SHORT", -1, _rising(tr_short)),
        ("QUANT_CROSS_LONG", 1, f["quant_buy_signal"] > 0),
        ("QUANT_CROSS_SHORT", -1, f["quant_sell_signal"] > 0),
        ("EMA_BUY", 1, f["bs_buy_signal"] > 0),
        ("EMA_SELL", -1, f["bs_sell_signal"] > 0),
        ("QTE_SCALPER_LONG", 1, f["qte_up_signal"] > 0),
        ("QTE_SCALPER_SHORT", -1, f["qte_down_signal"] > 0),
    ]

    # BOS/CHOCH are preserved primarily as context. Still expose explicit events
    # as separate families so research can prove or disprove entry value.
    specs += [
        ("BOS_BULL", 1, f["bos_bull_count"] > 0),
        ("BOS_BEAR", -1, f["bos_bear_count"] > 0),
        ("CHOCH_BULL", 1, f["choch_bull_count"] > 0),
        ("CHOCH_BEAR", -1, f["choch_bear_count"] > 0),
    ]

    # External timings are never merged into one generic flag. Every column is
    # its own family so the model can learn that one timing works and another does not.
    for c in f.columns:
        if c.startswith("timing_long_"):
            specs.append((c.upper(), 1, f[c].fillna(0) > 0))
        elif c.startswith("timing_short_"):
            specs.append((c.upper(), -1, f[c].fillna(0) > 0))
        elif c.startswith("signal_long_"):
            specs.append((c.upper(), 1, f[c].fillna(0) > 0))
        elif c.startswith("signal_short_"):
            specs.append((c.upper(), -1, f[c].fillna(0) > 0))

    rows: List[Dict[str, Any]] = []
    for family, direction, cond in specs:
        idxs = np.flatnonzero(cond.fillna(False).to_numpy(bool))
        for i in idxs:
            rows.append({
                "event_time": df.index[i],
                "bar_index": int(i),
                "symbol": symbol,
                "asset_class": asset_class,
                "timeframe": timeframe,
                "setup_family": family,
                "direction": int(direction),
                "regime": str(f["regime"].iloc[i]),
            })
    if not rows:
        return pd.DataFrame(columns=["event_time","bar_index","symbol","asset_class","timeframe","setup_family","direction","regime"])
    return pd.DataFrame(rows).sort_values(["event_time","setup_family"]).reset_index(drop=True)


# =============================================================================
# EVENT LABELS: NEXT BAR, TRIPLE BARRIER, MFE/MAE
# =============================================================================

def label_events(raw: pd.DataFrame,
                 features: pd.DataFrame,
                 events: pd.DataFrame,
                 cfg: OutcomeConfig = OutcomeConfig()) -> pd.DataFrame:
    df = _validate_ohlcv(raw)
    f = features.reindex(df.index)
    out = events.copy()
    records = []
    for _, ev in out.iterrows():
        i = int(ev["bar_index"])
        direction = int(ev["direction"])
        if i + 1 >= len(df):
            continue
        atr = float(f["atr"].iloc[i]) if np.isfinite(f["atr"].iloc[i]) else np.nan
        if not np.isfinite(atr) or atr <= 0:
            continue
        entry_i = i + 1
        entry = float(df["open"].iloc[entry_i])
        stop_dist = cfg.stop_atr * atr
        target_dist = cfg.target_atr * atr
        stop = entry - direction * stop_dist
        target = entry + direction * target_dist
        end_i = min(len(df)-1, entry_i + cfg.horizon_bars - 1)

        first = "TIMEOUT"
        exit_i = end_i
        exit_px = float(df["close"].iloc[end_i])
        mfe = 0.0
        mae = 0.0
        ambiguous = False
        for j in range(entry_i, end_i+1):
            hi = float(df["high"].iloc[j]); lo = float(df["low"].iloc[j])
            if direction == 1:
                favorable = hi - entry
                adverse = entry - lo
                hit_t = hi >= target; hit_s = lo <= stop
            else:
                favorable = entry - lo
                adverse = hi - entry
                hit_t = lo <= target; hit_s = hi >= stop
            mfe = max(mfe, favorable / stop_dist)
            mae = max(mae, adverse / stop_dist)
            if hit_t and hit_s:
                ambiguous = True
                if cfg.ambiguous_policy == "stop_first":
                    first = "AMBIG_STOP"; exit_i = j; exit_px = stop
                else:
                    first = "AMBIGUOUS"
                break
            if hit_s:
                first = "SL"; exit_i = j; exit_px = stop; break
            if hit_t:
                first = "TP"; exit_i = j; exit_px = target; break

        if ambiguous and cfg.ambiguous_policy == "exclude":
            realized_r = np.nan
            positive = np.nan
        else:
            realized_r = direction * (exit_px - entry) / stop_dist
            realized_r = float(np.clip(realized_r, -cfg.stop_atr / cfg.stop_atr, cfg.target_atr / cfg.stop_atr))
            positive = float(realized_r > 0)

        rec = dict(ev)
        rec.update({
            "entry_time": df.index[entry_i],
            "entry_price": entry,
            "label_end_time": df.index[exit_i],
            "exit_price": exit_px,
            "outcome": first,
            "realized_r": realized_r,
            "positive_r": positive,
            "mfe_r": float(mfe),
            "mae_r": float(mae),
        })
        records.append(rec)
    return pd.DataFrame(records)


# =============================================================================
# EVENT FEATURE MATRIX
# =============================================================================

def event_feature_table(features: pd.DataFrame, labelled_events: pd.DataFrame) -> pd.DataFrame:
    if labelled_events.empty:
        return labelled_events.copy()
    skip_cols = {"atr", "arash_adr", "arash_vwap", "arash_sigma", "bs_ema_fast", "bs_ema_slow", "qte_ema", "qte_atr10"}
    numeric_cols = [c for c in features.columns if c not in skip_cols and c != "regime" and pd.api.types.is_numeric_dtype(features[c])]
    rows = []
    for _, ev in labelled_events.iterrows():
        i = int(ev["bar_index"])
        r = ev.to_dict()
        for c in numeric_cols:
            r[c] = features[c].iloc[i]
        rows.append(r)
    return pd.DataFrame(rows)


# =============================================================================
# AI MODEL
# =============================================================================

META_COLS = ["setup_family", "symbol", "asset_class", "timeframe", "regime", "direction"]
TARGET_COLS = {"positive_r", "realized_r", "outcome", "mfe_r", "mae_r", "entry_price", "exit_price", "entry_time", "label_end_time", "event_time", "bar_index"}


def _feature_columns(events: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cat = [c for c in META_COLS if c in events.columns]
    num = [c for c in events.columns if c not in set(cat) | TARGET_COLS and pd.api.types.is_numeric_dtype(events[c])]
    return num, cat


def _preprocessor(num: List[str], cat: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)], remainder="drop")


@dataclass
class AIModelBundle:
    classifier: Any
    regressor: Any
    num_cols: List[str]
    cat_cols: List[str]
    feature_config: FeatureConfig
    outcome_config: OutcomeConfig
    ai_config: AIConfig

    def predict(self, event_rows: pd.DataFrame) -> pd.DataFrame:
        X = event_rows[self.num_cols + self.cat_cols]
        p = self.classifier.predict_proba(X)[:, 1]
        er = self.regressor.predict(X)
        return pd.DataFrame({"p_positive": p, "expected_r": er}, index=event_rows.index)


def fit_ai(events: pd.DataFrame,
           feature_cfg: FeatureConfig,
           outcome_cfg: OutcomeConfig,
           ai_cfg: AIConfig) -> AIModelBundle:
    train = events.dropna(subset=["positive_r", "realized_r"]).copy()
    if len(train) < ai_cfg.min_train_events:
        raise ValueError(f"Need at least {ai_cfg.min_train_events} labelled events, got {len(train)}")
    if train["positive_r"].nunique() < 2:
        raise ValueError("Training labels contain only one class")
    num, cat = _feature_columns(train)
    prep_c = _preprocessor(num, cat)
    prep_r = _preprocessor(num, cat)
    clf = Pipeline([
        ("prep", prep_c),
        ("model", HistGradientBoostingClassifier(
            learning_rate=ai_cfg.learning_rate, max_iter=ai_cfg.max_iter,
            max_leaf_nodes=ai_cfg.max_leaf_nodes, min_samples_leaf=ai_cfg.min_samples_leaf,
            l2_regularization=ai_cfg.l2_regularization, random_state=ai_cfg.random_state,
        )),
    ])
    reg = Pipeline([
        ("prep", prep_r),
        ("model", HistGradientBoostingRegressor(
            learning_rate=ai_cfg.learning_rate, max_iter=ai_cfg.max_iter,
            max_leaf_nodes=ai_cfg.max_leaf_nodes, min_samples_leaf=ai_cfg.min_samples_leaf,
            l2_regularization=ai_cfg.l2_regularization, random_state=ai_cfg.random_state,
            loss="squared_error",
        )),
    ])
    X = train[num + cat]
    clf.fit(X, train["positive_r"].astype(int))
    reg.fit(X, train["realized_r"].astype(float))
    return AIModelBundle(clf, reg, num, cat, feature_cfg, outcome_cfg, ai_cfg)


# =============================================================================
# PURGED CALENDAR WALK-FORWARD, MULTI-MARKET
# =============================================================================

def prepare_market(raw: pd.DataFrame,
                   symbol: str,
                   asset_class: str,
                   timeframe: str,
                   feature_cfg: FeatureConfig = FeatureConfig(),
                   outcome_cfg: OutcomeConfig = OutcomeConfig()) -> Dict[str, Any]:
    df = _validate_ohlcv(raw)
    feat = build_features(df, feature_cfg)
    events = build_setup_events(df, feat, symbol, asset_class, timeframe, feature_cfg)
    labelled = label_events(df, feat, events, outcome_cfg)
    table = event_feature_table(feat, labelled)
    return {"raw": df, "features": feat, "events": table}


def combine_markets(markets: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    frames = []
    for name, pack in markets.items():
        e = pack["events"].copy()
        if not e.empty:
            frames.append(e)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["event_time"] = pd.to_datetime(out["event_time"], utc=True)
    out["label_end_time"] = pd.to_datetime(out["label_end_time"], utc=True)
    return out.sort_values(["event_time", "symbol", "setup_family"]).reset_index(drop=True)


def walk_forward_ai(event_table: pd.DataFrame,
                    feature_cfg: FeatureConfig = FeatureConfig(),
                    outcome_cfg: OutcomeConfig = OutcomeConfig(),
                    ai_cfg: AIConfig = AIConfig()) -> Dict[str, Any]:
    e = event_table.dropna(subset=["positive_r", "realized_r"]).copy()
    if e.empty:
        raise ValueError("No labelled events")
    e["event_time"] = pd.to_datetime(e["event_time"], utc=True)
    e["label_end_time"] = pd.to_datetime(e["label_end_time"], utc=True)
    e = e.sort_values("event_time").reset_index(drop=True)

    first = e["event_time"].min().normalize()
    last = e["event_time"].max().normalize()
    test_start = first + pd.Timedelta(days=ai_cfg.train_days)
    predictions = []
    folds = []
    fold = 0

    while test_start <= last:
        test_end = test_start + pd.Timedelta(days=ai_cfg.test_days)
        train_start = test_start - pd.Timedelta(days=ai_cfg.train_days)
        train = e[(e["event_time"] >= train_start) & (e["label_end_time"] < test_start)].copy()
        test = e[(e["event_time"] >= test_start) & (e["event_time"] < test_end)].copy()
        if len(test) == 0:
            test_start = test_end; fold += 1; continue
        if len(train) < ai_cfg.min_train_events or train["positive_r"].nunique() < 2:
            test_start = test_end; fold += 1; continue
        bundle = fit_ai(train, feature_cfg, outcome_cfg, ai_cfg)
        pred = bundle.predict(test)
        scored = test.copy()
        scored["p_positive"] = pred["p_positive"].to_numpy()
        scored["expected_r_pred"] = pred["expected_r"].to_numpy()
        scored["fold"] = fold
        scored["selected"] = (scored["p_positive"] >= ai_cfg.min_probability) & (scored["expected_r_pred"] >= ai_cfg.min_expected_r)
        predictions.append(scored)

        y = test["positive_r"].astype(int).to_numpy()
        p = scored["p_positive"].to_numpy()
        rtrue = test["realized_r"].to_numpy(float)
        rpred = scored["expected_r_pred"].to_numpy(float)
        row = {
            "fold": fold, "train_start": train_start, "train_end": test_start,
            "test_start": test_start, "test_end": test_end,
            "train_events": len(train), "test_events": len(test),
            "brier": float(brier_score_loss(y, p)),
            "mae_r": float(mean_absolute_error(rtrue, rpred)),
            "rmse_r": float(math.sqrt(mean_squared_error(rtrue, rpred))),
        }
        if len(np.unique(y)) == 2:
            row["auc"] = float(roc_auc_score(y, p))
            row["logloss"] = float(log_loss(y, np.column_stack([1-p, p]), labels=[0,1]))
        else:
            row["auc"] = np.nan; row["logloss"] = np.nan
        folds.append(row)
        test_start = test_end; fold += 1

    scored_all = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return {"scored_events": scored_all, "folds": pd.DataFrame(folds)}


# =============================================================================
# DECISION / PERFORMANCE AUDIT
# =============================================================================

def select_nonoverlapping_trades(scored_events: pd.DataFrame,
                                 ai_cfg: AIConfig = AIConfig()) -> pd.DataFrame:
    if scored_events.empty:
        return scored_events.copy()
    s = scored_events.copy()
    s = s[(s["p_positive"] >= ai_cfg.min_probability) & (s["expected_r_pred"] >= ai_cfg.min_expected_r)]
    if s.empty:
        return s
    # At each timestamp choose the candidate with highest predicted expected R.
    s = s.sort_values(["event_time", "expected_r_pred", "p_positive"], ascending=[True, False, False])
    s = s.groupby(["symbol", "event_time"], as_index=False).first()
    # One position per symbol: suppress overlapping label windows.
    rows = []
    next_free: Dict[str, pd.Timestamp] = {}
    for _, r in s.sort_values("event_time").iterrows():
        sym = str(r["symbol"])
        t = pd.Timestamp(r["event_time"])
        if sym in next_free and t <= next_free[sym]:
            continue
        rows.append(r.to_dict())
        next_free[sym] = pd.Timestamp(r["label_end_time"])
    return pd.DataFrame(rows)


def _r_metrics(trades: pd.DataFrame) -> Dict[str, Any]:
    if trades.empty:
        return {"trades": 0, "net_r": 0.0, "expectancy_r": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "max_drawdown_r": 0.0}
    r = trades["realized_r"].astype(float)
    gp = float(r[r > 0].sum()); gl = float(-r[r < 0].sum())
    eq = r.cumsum(); dd = eq - eq.cummax()
    return {
        "trades": int(len(r)), "net_r": float(r.sum()), "expectancy_r": float(r.mean()),
        "win_rate": float((r > 0).mean()),
        "profit_factor": float(gp/gl) if gl > 0 else (math.inf if gp > 0 else 0.0),
        "max_drawdown_r": float(dd.min()),
    }


def performance_report(scored_events: pd.DataFrame,
                       ai_cfg: AIConfig = AIConfig()) -> Dict[str, Any]:
    if scored_events.empty:
        return {"AI": _r_metrics(pd.DataFrame()), "BASELINES": {}, "FAIL_AUDIT": {}}
    ai_trades = select_nonoverlapping_trades(scored_events, ai_cfg)
    base = {}
    for fam, g in scored_events.groupby("setup_family"):
        base[str(fam)] = _r_metrics(g)

    # Quant fail-trade audit: did AI reject negative QUANT events and retain positive ones?
    q = scored_events[scored_events["setup_family"].isin(["QUANT_CROSS_LONG", "QUANT_CROSS_SHORT"])].copy()
    if len(q):
        selected = (q["p_positive"] >= ai_cfg.min_probability) & (q["expected_r_pred"] >= ai_cfg.min_expected_r)
        fail = q["realized_r"] <= 0
        good = q["realized_r"] > 0
        audit = {
            "quant_events": int(len(q)),
            "quant_fail_events": int(fail.sum()),
            "quant_fail_blocked_by_ai": int((fail & ~selected).sum()),
            "quant_fail_allowed_by_ai": int((fail & selected).sum()),
            "quant_good_events": int(good.sum()),
            "quant_good_kept_by_ai": int((good & selected).sum()),
            "quant_good_blocked_by_ai": int((good & ~selected).sum()),
        }
    else:
        audit = {}
    return {"AI": _r_metrics(ai_trades), "BASELINES": base, "FAIL_AUDIT": audit, "AI_TRADES": ai_trades}


# =============================================================================
# FINAL FIT / LIVE CANDIDATE SCORING / PERSISTENCE
# =============================================================================

def fit_final_model(event_table: pd.DataFrame,
                    feature_cfg: FeatureConfig = FeatureConfig(),
                    outcome_cfg: OutcomeConfig = OutcomeConfig(),
                    ai_cfg: AIConfig = AIConfig()) -> AIModelBundle:
    return fit_ai(event_table, feature_cfg, outcome_cfg, ai_cfg)


def save_bundle(bundle: AIModelBundle, path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(bundle, path)


def load_bundle(path: str | Path) -> AIModelBundle:
    return joblib.load(Path(path))


def live_candidates(raw: pd.DataFrame,
                    bundle: AIModelBundle,
                    symbol: str,
                    asset_class: str,
                    timeframe: str) -> pd.DataFrame:
    df = _validate_ohlcv(raw)
    f = build_features(df, bundle.feature_config)
    ev = build_setup_events(df, f, symbol, asset_class, timeframe, bundle.feature_config)
    if ev.empty:
        return pd.DataFrame()
    last_t = df.index[-1]
    ev = ev[ev["event_time"] == last_t].copy()
    if ev.empty:
        return ev
    # attach current features without labels
    fake = ev.copy()
    fake["entry_time"] = pd.NaT; fake["label_end_time"] = pd.NaT
    fake["positive_r"] = np.nan; fake["realized_r"] = np.nan; fake["outcome"] = "LIVE"
    fake["mfe_r"] = np.nan; fake["mae_r"] = np.nan; fake["entry_price"] = np.nan; fake["exit_price"] = np.nan
    table = event_feature_table(f, fake)
    pred = bundle.predict(table)
    table["p_positive"] = pred["p_positive"].to_numpy()
    table["expected_r_pred"] = pred["expected_r"].to_numpy()
    table["eligible"] = (table["p_positive"] >= bundle.ai_config.min_probability) & (table["expected_r_pred"] >= bundle.ai_config.min_expected_r)
    return table.sort_values(["eligible","expected_r_pred","p_positive"], ascending=[False,False,False])


# =============================================================================
# PARITY / CAUSALITY AUDIT
# =============================================================================

def prefix_invariance_audit(raw: pd.DataFrame,
                            cfg: FeatureConfig = FeatureConfig(),
                            checkpoints: int = 8,
                            compare_tail: int = 20,
                            atol: float = 1e-10) -> Dict[str, Any]:
    df = _validate_ohlcv(raw)
    full = build_features(df, cfg)
    ends = np.unique(np.linspace(max(200, compare_tail*2), len(df), checkpoints, dtype=int))
    failures = []
    numeric_cols = [c for c in full.columns if pd.api.types.is_numeric_dtype(full[c])]
    for end in ends:
        pref = build_features(df.iloc[:end], cfg)
        a = full[numeric_cols].iloc[max(0,end-compare_tail):end]
        b = pref[numeric_cols].iloc[max(0,end-compare_tail):end]
        diff = (a-b).abs()
        bad_cols = []
        if len(diff):
            for c in numeric_cols:
                arr = diff[c].to_numpy(dtype=float)
                finite = arr[np.isfinite(arr)]
                if len(finite) and float(finite.max()) > atol:
                    bad_cols.append(c)
        if bad_cols:
            failures.append({"end": int(end), "columns": bad_cols})
    return {"pass": len(failures)==0, "failures": failures, "checkpoints": [int(x) for x in ends]}


def source_coverage_manifest() -> Dict[str, Any]:
    return {
        "ArashArrow": "implemented as separate MR/TREND families + original/V3 context features",
        "QuantEngine": "implemented exact pasted factor/cross logic",
        "BOS_CHOCH": "implemented causal pivots, BOS/CHOCH, IDM, zones, mitigation/break, distances",
        "BuySell": "implemented EMA 5/13 signal logic as separate families",
        "QTEScalper": "implemented pasted EMA/ATR/touch logic including min ATR",
        "ExternalTimings": "each timing_long_*/timing_short_*/signal_long_*/signal_short_* column stays its own family",
        "AI": "purged calendar walk-forward classifier+regressor on setup-family events",
        "RealityChecks": "next-bar entry labels, ambiguous OHLC exclusion, MFE/MAE, prefix invariance, OOS-only reporting",
    }


def read_ohlcv_csv(path: str | Path, timestamp_col: str = "timestamp") -> pd.DataFrame:
    d = pd.read_csv(path)
    cols = {str(c).strip().lower(): c for c in d.columns}
    if timestamp_col not in d.columns:
        key = timestamp_col.lower()
        if key in cols:
            timestamp_col = cols[key]
        else:
            raise ValueError(f"CSV needs timestamp column; found {list(d.columns)}")
    idx = pd.to_datetime(d.pop(timestamp_col), utc=True, errors="raise")
    d.columns = [str(c).strip().lower() for c in d.columns]
    d.index = idx
    return d


def compact_report(wf: Mapping[str, Any], ai_cfg: AIConfig = AIConfig()) -> str:
    perf = performance_report(wf["scored_events"], ai_cfg)
    payload = {"folds": wf["folds"].to_dict(orient="records"), "performance": {k:v for k,v in perf.items() if k != "AI_TRADES"}}
    return json.dumps(payload, indent=2, default=str)


if __name__ == "__main__":
    print(json.dumps(source_coverage_manifest(), indent=2))
