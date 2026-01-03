from typing import Any
import pandas as pd
from math import exp
import numpy as np

from Records import Records

def static_residual_signal(r:Records):
    resid = r.static_resid
    z = resid.sub(resid.mean(axis=1), axis=0).div(resid.std(axis=1), axis=0)
    signal = -z
    return signal, z

def modified_residual_signal(r:Records, lamb_hi=0.4, lamb_lo=0.05, gamma=1.0):
    companies = r.companies

    # --- window ---
    start = max(int(r.t) - 10, 0)
    R = r.returns.iloc[start:].astype(float)
    mkt = r.mkt_ret.iloc[start:].astype(float)
    dates = R.index

    # --- data ---
    short_beta  = r.short_beta.reindex(index=dates, columns=companies)
    long_beta   = r.long_beta.reindex(index=dates, columns=companies)
    short_alpha = r.short_alpha.reindex(index=dates, columns=companies)
    long_alpha  = r.long_alpha.reindex(index=dates, columns=companies)

    # --- blend ---
    delta = (short_beta - long_beta).abs()
    lamb  = lamb_lo + (lamb_hi - lamb_lo) * np.exp(-gamma * delta)

    blend_beta  = short_beta.mul(lamb) + long_beta.mul(1 - lamb)
    blend_alpha = short_alpha.mul(lamb) + long_alpha.mul(1 - lamb)
    r.blend_beta = blend_beta
    r.blend_alpha = blend_alpha

    # --- residual ---
    resid = R.sub(blend_beta.mul(mkt, axis=0)).sub(blend_alpha)

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

def topk_signal_to_weights(r:Records, signal: pd.DataFrame, k:int):
    trading_days = r.trading_days
    companies = r.companies
    weights = pd.DataFrame(0.0, index=trading_days, columns=companies)
    for day in trading_days:
        s = signal.loc[day].dropna()
        longs = s.nlargest(k).index
        shorts = s.nsmallest(k).index
        weights.loc[day, longs] =  1.0 / k
        weights.loc[day, shorts] = -1.0 / k
    return weights

def modified_topk_signal_to_weights(r:Records, signal: pd.DataFrame, z: pd.DataFrame, k:int=None):
    trading_days = r.trading_days
    companies = r.companies
    if not k:
        k = np.ceil(len(r.companies)*0.1)
    signal = tanh_signal(signal)

    k = np.ceil(len(companies)*0.1)
    signal = topk_z_gate(signal, z, k)
    signal = beta_neutral(signal, r.blend_beta)
    signal = topk_z_gate(signal, z, k)
    # --- cross-sectional normalization ---
    signal = L1_regularization(signal)

    weights = signal.reindex(trading_days).fillna(0.0)
    return weights

def extreme_signal_to_weights(r:Records, signal: pd.DataFrame, z: pd.DataFrame):
    signal = tanh_signal(signal)
    signal = beta_neutral(signal, r.blend_beta)
    signal = signal.where(signal.abs()>1, 0.0)
    signal = L1_regularization(signal)
    weights = signal.reindex(r.trading_days).fillna(0.0)
    active_ratio = (weights.abs().sum(axis=1) > 0).mean()
    print("active_ratio = ", active_ratio)
    return weights
