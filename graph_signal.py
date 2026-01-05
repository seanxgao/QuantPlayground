from Records import Records
import pandas as pd
import numpy as np
import scipy.sparse as sp
import ib_connection as IBC
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

def get_W_sym_by_date(r: Records,mkt: str = "SPY") -> dict:
    dates = r.dates
    trading_days = r.get_trading_days()
    companies = r.companies

    E = r.static_resid[mkt]

    W = int(r.window)
    k = 20
    epsilon = 1e-8

    payload = {
        "name": "W_sym_by_date",
        "format": "scipy_csr",
        "universe": "SP500_filtered",
        "mkt": mkt,
        "t_train": int(r.t),
        "W": int(W),
        "k": int(k),
        "epsilon": float(epsilon),
        "T": int(len(dates)),
        "N": int(len(companies)),
        "start": str(dates.min()),
        "end": str(dates.max()),
        "trade_start": str(trading_days.min()),
        "trade_end": str(trading_days.max()),
        "tickers": list(map(str, companies)),
    }

    mu_roll = E.rolling(W).mean()
    sd_roll = E.rolling(W).std(ddof=1)

    def _build_W_sym_by_date():
        W_sym_by_date = {}
        for t in range(W, len(dates)):
            X = E.iloc[t - W : t]

            mu = mu_roll.iloc[t - 1]
            sd = sd_roll.iloc[t - 1].replace(0.0, np.nan)

            Z = X.sub(mu, axis=1).div(sd, axis=1)
            Z = Z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            C = Z.corr()
            np.fill_diagonal(C.values, 0.0)

            W_tilde = pd.DataFrame(0.0, index=companies, columns=companies)
            absC = C.abs()
            k_eff = min(int(k), len(companies) - 1)
            for i in companies:
                top_j = absC.loc[i].nlargest(k_eff).index
                W_tilde.loc[i, top_j] = C.loc[i, top_j]

            W_sym = (W_tilde + W_tilde.T) / 2.0

            # Store as scipy sparse (CSR) in the fixed `companies` ordering.
            W_sym_by_date[dates[t]] = sp.csr_matrix(W_sym.to_numpy(dtype=float))

        return W_sym_by_date
    W_sym_by_date = IBC.fetch_with_cache(
        payload=payload,
        subdir="graphs",
        compute_fn=_build_W_sym_by_date,
        serializer="pickle",
        meta=payload,
    )
    return W_sym_by_date


def W_to_signal(
    r: Records,
    E: pd.DataFrame,
    W_by_date: dict,
    clip_eps: float | None = None,
) -> pd.DataFrame:

    dates = r.get_trading_days()
    tickers = r.companies

    S = pd.DataFrame(index=dates, columns=tickers, dtype=float)

    for t in dates:
        W = W_by_date.get(t)
        if W is None:
            continue

        eps = E.loc[t].to_numpy(dtype=float)
        if clip_eps is not None:
            eps = np.clip(eps, -clip_eps, clip_eps)

        if sp.issparse(W):
            S.loc[t] = W.dot(eps)
        elif isinstance(W, pd.DataFrame):
            Wm = W.reindex(index=tickers, columns=tickers).to_numpy(dtype=float)
            S.loc[t] = Wm @ eps
        else:
            S.loc[t] = np.asarray(W, dtype=float) @ eps

    return S

def topk_eig_sym(W_sym: csr_matrix, k: int = 10, which: str = "LM"):
    """
    W_sym: symmetric CSR matrix (N×N)
    which:
      - "LM": largest magnitude
      - "LA": largest algebraic (most positive)
    """
    vals, vecs = eigsh(W_sym, k=k, which=which)
    order = np.argsort(-np.abs(vals))
    return vals[order], vecs[:, order]   # vals:(k,), vecs:(N,k)


def apply_lowrank_operator(x: np.ndarray, vals: np.ndarray, vecs: np.ndarray):
    """
    Compute (U diag(vals) U^T) x without forming dense matrix.
    """
    return vecs @ (vals * (vecs.T @ x))

def spectral_truncated_signal(r: Records, W_sym_by_date: dict, E: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    trading_days = r.get_trading_days()
    companies = r.companies

    S = pd.DataFrame(index=trading_days, columns=companies, dtype=float)

    missing = 0
    for d in trading_days:
        Wsym = W_sym_by_date.get(d)
        if Wsym is None:
            missing += 1
            continue

        vals, U = topk_eig_sym(Wsym, k=int(k), which="LM")

        eps = E.loc[d].to_numpy(dtype=float)
        eps = np.nan_to_num(eps, nan=0.0, posinf=0.0, neginf=0.0)

        S.loc[d] = apply_lowrank_operator(eps, vals, U)

    print("missing spectral W on trading_days:", int(missing))
    return S