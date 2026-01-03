"""
Plotting utilities for strategy research.

Design notes:
- Notebook-friendly: no runtime checks; assumes the caller set the right properties already.
- English-only code/comments.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Records import Records


def plot_equity(equity_gross: pd.Series, equity_net: pd.Series, title: str = "Strategy Equity Curve"):
    plt.figure(figsize=(12, 5))
    plt.plot(equity_gross.index, equity_gross.values, label="Equity (gross)")
    plt.plot(equity_net.index, equity_net.values, label="Equity (net)")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity (start=1.0)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def alpha_diff_diagnostics(equity_net: pd.Series, W: int = 20) -> pd.DataFrame:
    # log returns
    r = np.log(equity_net).diff()

    # rolling slope (mean log-return)
    slope = r.rolling(W).mean()

    # rolling sharpe
    sharpe = r.rolling(W).mean() / r.rolling(W).std()

    return pd.DataFrame(
        {
            "equity": equity_net,
            "log_ret": r,
            "alpha_slope": slope,
            "alpha_sharpe": sharpe,
        }
    )


def plot_alpha_diff_diagnostics(equity_net: pd.Series, W: int = 20, title_prefix: str = ""):
    diag = alpha_diff_diagnostics(equity_net, W=W)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # (1) Equity curve
    axes[0].plot(diag.index, diag["equity"], label="Equity (net)")
    axes[0].set_title(f"{title_prefix}Equity Net".strip())
    axes[0].grid(True)

    # (2) Alpha slope
    axes[1].plot(diag.index, diag["alpha_slope"], label="Rolling Alpha Slope", color="orange")
    axes[1].axhline(0, linestyle="--", color="black", alpha=0.6)
    axes[1].set_title(f"{title_prefix}Rolling Alpha Slope".strip())
    axes[1].grid(True)

    # (3) Rolling Sharpe
    axes[2].plot(diag.index, diag["alpha_sharpe"], label="Rolling Sharpe", color="green")
    axes[2].axhline(0, linestyle="--", color="black", alpha=0.6)
    axes[2].axhline(0.5, linestyle=":", color="gray", alpha=0.7)
    axes[2].set_title(f"{title_prefix}Rolling Sharpe".strip())
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


def equity_from_records(
    r: Records,
    weights: Optional[pd.DataFrame] = None,
    tc_bps: float = 1.0,
) -> Tuple[pd.Series, pd.Series]:
    """
    Convenience wrapper:
    - Uses r.weights if weights is None.
    - Calls r.pnl_trading(...) and returns (equity_gross, equity_net).
    """
    if weights is None:
        weights = r.weights
    equity_gross, equity_net = r.pnl_trading(weights, tc_bps=tc_bps)
    return equity_gross, equity_net


def plot_equity_from_records(
    r: Records,
    weights: Optional[pd.DataFrame] = None,
    tc_bps: float = 1.0,
    title: str = "Strategy Equity Curve",
):
    equity_gross, equity_net = equity_from_records(r, weights=weights, tc_bps=tc_bps)
    plot_equity(equity_gross, equity_net, title=title)


def plot_alpha_diagnostics_from_records(
    r: Records,
    weights: Optional[pd.DataFrame] = None,
    tc_bps: float = 1.0,
    W: int = 20,
    title_prefix: str = "",
):
    _, equity_net = equity_from_records(r, weights=weights, tc_bps=tc_bps)
    plot_alpha_diff_diagnostics(equity_net, W=W, title_prefix=title_prefix)