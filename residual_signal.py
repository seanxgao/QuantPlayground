from typing import Any
import pandas as pd
from math import exp
import numpy as np

from Records import Records

def _default_mkt(r: Records, mkt: str | None) -> str:
    if mkt is not None:
        return mkt
    # Records keeps markets in r.mkt_tickers
    return r.mkt_tickers[0]


def static_residual_signal(r:Records, mkt: str | None = None):
    m = _default_mkt(r, mkt)
    resid = r.static_resid[m]
    z = resid.sub(resid.mean(axis=1), axis=0).div(resid.std(axis=1), axis=0)
    signal = -z
    return signal, z

def modified_residual_signal(r:Records, mkt: str | None = None, lamb_hi=0.4, lamb_lo=0.05, gamma=1.0):
    m = _default_mkt(r, mkt)
    companies = r.companies

    # --- window ---
    dates_all = r.get_dates()
    start = max(int(r.t) - 10, 0)
    dates = dates_all[start:]
    R = r.returns.reindex(dates).astype(float)
    M = r.mkt_ret[m].reindex(dates).astype(float)

    # --- data ---
    short_beta  = r.short_beta[m].reindex(index=dates, columns=companies)
    long_beta   = r.long_beta[m].reindex(index=dates, columns=companies)
    short_alpha = r.short_alpha[m].reindex(index=dates, columns=companies)
    long_alpha  = r.long_alpha[m].reindex(index=dates, columns=companies)

    # --- blend ---
    delta = (short_beta - long_beta).abs()
    lamb  = lamb_lo + (lamb_hi - lamb_lo) * np.exp(-gamma * delta)

    blend_beta  = short_beta.mul(lamb) + long_beta.mul(1 - lamb)
    blend_alpha = short_alpha.mul(lamb) + long_alpha.mul(1 - lamb)
    r.blend_beta[m] = blend_beta
    r.blend_alpha[m] = blend_alpha

    # --- residual ---
    resid = R.sub(blend_beta.mul(M, axis=0)).sub(blend_alpha)

    z = resid.sub(resid.mean(axis=1), axis=0).div(resid.std(axis=1), axis=0)

    signal = -z
    return signal, z

# --- regularization ---

tanh_signal = lambda z: -np.tanh(z / 2.0).fillna(0.0)
beta_neutral = lambda s, beta: s.sub(beta.mul((s.mul(beta).sum(axis=1) / beta.pow(2).sum(axis=1).clip(lower=0.01)), axis=0))
dollar_neutral = lambda s: s.sub(s.mean(axis=1), axis=0)
L1_regularization = lambda s: s.div(s.abs().sum(axis=1), axis=0).fillna(0.0)

def topk_z_gate(signal: pd.DataFrame, z: pd.DataFrame, k:int) -> pd.DataFrame:
    rank = z.abs().rank(axis=1, ascending=False, method="first")
    return signal.where(rank <= k, 0.0)

def topk_signal_to_weights(r:Records, signal: pd.DataFrame, k:int, mkt: str | None = None):
    m = _default_mkt(r, mkt)
    trading_days = r.get_trading_days()
    companies = r.companies
    weights = pd.DataFrame(0.0, index=trading_days, columns=companies)
    for day in trading_days:
        s = signal.loc[day].dropna()
        longs = s.nlargest(k).index
        shorts = s.nsmallest(k).index
        weights.loc[day, longs] =  1.0 / k
        weights.loc[day, shorts] = -1.0 / k
    return weights


def modified_topk_signal_to_weights(r:Records, signal: pd.DataFrame, z: pd.DataFrame, k:int=None, mkt: str | None = None):
    m = _default_mkt(r, mkt)
    trading_days = r.get_trading_days()
    companies = r.companies
    if not k:
        k = np.ceil(len(r.companies)*0.1)
    signal = tanh_signal(signal)

    k = np.ceil(len(companies)*0.1)
    signal = topk_z_gate(signal, z, k)
    # Use blend_beta when available; otherwise fall back to long_beta.
    beta_ref = r.blend_beta[m] if m in r.blend_beta else r.long_beta[m]
    signal = beta_neutral(signal, beta_ref)
    signal = topk_z_gate(signal, z, k)
    # --- cross-sectional normalization ---
    signal = L1_regularization(signal)

    weights = signal.reindex(trading_days).fillna(0.0)
    return weights

def extreme_signal_to_weights(r:Records, signal: pd.DataFrame, z: pd.DataFrame | None = None, mkt: str | None = None):
    m = _default_mkt(r, mkt)
    signal = tanh_signal(signal)
    # r.blend_beta is a dict; it may exist but not contain this market unless
    # modified_residual_signal() has been called for m.
    beta_ref = r.blend_beta[m] if m in r.blend_beta else r.long_beta[m]
    signal = beta_neutral(signal, beta_ref)
    signal = signal.where(signal.abs()>1, 0.0)
    signal = L1_regularization(signal)
    weights = signal.reindex(r.get_trading_days()).fillna(0.0)
    active_ratio = (weights.abs().sum(axis=1) > 0).mean()
    print("active_ratio = ", active_ratio)
    return weights

def both_extreme_signal_to_weights(r_1:Records, r_2:Records, signal_1: pd.DataFrame, signal_2: pd.DataFrame):
    signal_1 = tanh_signal(signal_1)
    signal_2 = tanh_signal(signal_2)
    mkt_1 = r_1.mkt_tickers[0]
    mkt_2 = r_2.mkt_tickers[0]
    signal_1 = beta_neutral(signal_1, r_1.blend_beta[mkt_1])
    signal_2 = beta_neutral(signal_2, r_2.blend_beta[mkt_2])
    signal = (signal_1+signal_2).div(2.0).where((signal_1.abs()>1) & (signal_2.abs()>1), 0.0)
    weights = L1_regularization(signal).reindex(r_1.get_trading_days(mkt_1)).fillna(0.0)
    active_ratio = (weights.abs().sum(axis=1) > 0).mean()
    print("active_ratio = ", active_ratio)
    return weights


def both_extreme_signal_to_weights_single(
    r: Records,
    signal_1: pd.DataFrame,
    signal_2: pd.DataFrame,
    mkt_1: str,
    mkt_2: str,
):
    """
    Combine two extreme signals from a SINGLE Records object but two different markets.
    """
    signal_1 = tanh_signal(signal_1)
    signal_2 = tanh_signal(signal_2)
    signal_1 = beta_neutral(signal_1, r.blend_beta[mkt_1])
    signal_2 = beta_neutral(signal_2, r.blend_beta[mkt_2])
    signal = (signal_1 + signal_2).div(2.0).where((signal_1.abs() > 1) & (signal_2.abs() > 1), 0.0)
    weights = L1_regularization(signal).reindex(r.get_trading_days()).fillna(0.0)
    active_ratio = (weights.abs().sum(axis=1) > 0).mean()
    print("active_ratio = ", active_ratio)
    return weights
