import pandas as pd
import numpy as np
import copy
from typing import Optional

class Records:
    def __init__(self):
        # ---- config / inputs ----
        self.tickers = None
        self.dur = None
        self.size = None
        self.t = None  # training length (number of rows)
        self.long_window = None
        self.short_window = None
        self.mkt_ticker = "SPY"
        self.mkt_params = None

        # ---- data ----
        # NOTE:
        # - `returns` and `mkt_ret` are the only required data objects.
        # - We intentionally do NOT store a combined df (companies + mkt) to save memory.
        self.returns = None  # companies returns DataFrame
        self.mkt_ret = None  # market returns Series aligned with `returns.index`

        # ---- derived calendar ----
        self.companies = None
        self.dates = None
        self.training_days = None
        self.trading_days = None

        # ---- regression outputs ----
        self.static_resid = None
        self.static_beta = None
        self.static_alpha = None
        self.short_beta = None
        self.short_alpha = None
        self.long_beta = None
        self.long_alpha = None

        # ---- blend outputs (dangerous) ----
        self.blend_beta = None
        self.blend_alpha = None

        # ---- strategy outputs (optional) ----
        self.weights = None
        self.equity_gross = None
        self.equity_net = None
    

    @staticmethod
    def _copy_value(v, deep: bool):
        """
        Copy helper for common containers and pandas/numpy objects.

        Notes:
        - For pandas objects, we prefer `.copy()` to avoid pandas-specific pitfalls with deepcopy.
        - For plain scalars/immutables, returning the same object is fine.
        """
        if v is None:
            return None

        # pandas
        if isinstance(v, (pd.DataFrame, pd.Series, pd.Index)):
            if not deep:
                # Keep the same object for true sharing in shallow copies.
                return v
            try:
                return v.copy(deep=True)
            except TypeError:
                return v.copy()

        # numpy
        if isinstance(v, np.ndarray):
            return v.copy() if deep else v

        # containers
        if isinstance(v, (dict, list, tuple, set)):
            return copy.deepcopy(v) if deep else copy.copy(v)

        # default: try to deepcopy, otherwise keep reference
        if deep:
            try:
                return copy.deepcopy(v)
            except Exception:
                return v
        return v

    def copy(self, deep: Optional[bool] = None, **overrides) -> "Records":
        """
        Create a copy of this Records object.

        Default behavior:
        - If no overrides are provided: deep copy (deep=True).
        - If any overrides are provided (e.g. mkt_ticker="SPY") and deep is not explicitly set:
          automatically use a shallow copy (deep=False) to encourage memory sharing.

        Parameters:
        - deep: True for deep copy, False for shallow-ish copy, None to use the default rule above.
        - overrides: any attribute overrides to apply to the copied object.
        """
        if deep is None:
            deep = False if overrides else True

        new = Records()
        for k, v in self.__dict__.items():
            setattr(new, k, self._copy_value(v, deep=deep))

        for k, v in overrides.items():
            setattr(new, k, v)
        return new

    def __copy__(self):
        return self.copy(deep=False)

    def __deepcopy__(self, memo):
        # memo is handled internally by copy.deepcopy; we just return our deep copy.
        return self.copy(deep=True)

    def set_calendar(self):
        if self.returns is None:
            raise ValueError("Missing data: `returns` must be populated before calling set_calendar().")

        self.companies = self.returns.columns
        self.dates = self.returns.index
        t = int(self.t)
        self.training_days = self.dates[:t]
        self.trading_days = self.dates[t:]
        print("len(returns) =", len(self.returns))
        if self.mkt_ret is not None:
            print("len(mkt_ret) =", len(self.mkt_ret))
        print("t            =", self.t)
        print("date range   =", self.dates.min(), "->", self.dates.max())

    # =========================
    # Data fetch + cache
    # =========================
    def fetch_df(self, ib, progress=True):
        """
        Fetch companies returns + market returns, with caching.

        Notes:
        - Kept for backwards compatibility; it does NOT store a combined df.
        - Populates: returns, mkt_ret
        """
        self.fetch_comp(ib, progress=progress)
        self.fetch_mkt(ib, progress=progress)
        return self.returns, self.mkt_ret

    def fetch_comp(self, ib, progress=True):
        """
        Fetch companies returns only (no market), with caching.

        Populates: returns (+ calendar fields).
        """
        import ib_connection as IBC

        returns = IBC.fetch_companies_returns_with_cache(
            ib,
            tickers=self.tickers,
            duration=self.dur,
            bar_size=self.size,
            progress=progress,
        )
        self.returns = returns
        self.set_calendar()
        return self.returns

    def fetch_mkt(
        self,
        ib,
        progress=True,
        mkt_ticker=None,
        mkt_params=None,
    ):
        """
        Fetch market returns only, with caching.

        Populates: mkt_ret.
        """
        import ib_connection as IBC

        if mkt_ticker is not None:
            self.mkt_ticker = mkt_ticker
        if mkt_params is not None:
            self.mkt_params = mkt_params

        mkt_df = IBC.fetch_mkt_returns_with_cache(
            ib,
            mkt_ticker=self.mkt_ticker,
            duration=self.dur,
            bar_size=self.size,
            mkt_params=self.mkt_params,
            progress=progress,
        )
        self.mkt_ret = mkt_df["mkt"].copy()

        # Refresh calendar if returns exists (dates may need aligning later).
        if self.returns is not None:
            self.set_calendar()

        return self.mkt_ret
    
    def rolling_beta_alpha(self, wind:int):
        wind = int(wind)
        if self.returns is None or self.mkt_ret is None:
            raise ValueError("Missing data: `returns` and `mkt_ret` must be populated before calling rolling_beta_alpha().")

        rolling_alpha = pd.DataFrame(index=self.dates, columns=self.companies, dtype=float)
        rolling_beta = pd.DataFrame(index=self.dates, columns=self.companies, dtype=float)
        mkt_var = self.mkt_ret.rolling(wind).var()
        mkt_mean = self.mkt_ret.rolling(wind).mean()
        for c in self.companies:
            cov = self.returns[c].rolling(wind).cov(self.mkt_ret)
            rolling_beta[c] = cov / mkt_var
            rolling_alpha[c] = self.returns[c].rolling(wind).mean() - rolling_beta[c] * mkt_mean
        return rolling_beta, rolling_alpha
    
    def set_long_beta_alpha(self):
        self.long_beta, self.long_alpha = self.rolling_beta_alpha(self.long_window)
        return self.long_beta, self.long_alpha

    def set_short_beta_alpha(self):
        self.short_beta, self.short_alpha = self.rolling_beta_alpha(self.short_window)
        return self.short_beta, self.short_alpha

    def set_static_beta_alpha_residual(self):
        # Local import to avoid hard dependency when using non-regression parts of Records.
        import statsmodels.api as sm

        if self.returns is None or self.mkt_ret is None:
            raise ValueError("Missing data: `returns` and `mkt_ret` must be populated before calling set_static_beta_alpha_residual().")

        static_alpha = {}
        static_betas = {}
        residuals = {}
        for c in self.companies:
            y = self.returns[c].values[:self.t]
            X = sm.add_constant(self.mkt_ret.values[:self.t])
            model = sm.OLS(y, X).fit()
            static_betas[c] = model.params[1]
            static_alpha[c] = model.params[0]

        static_beta = pd.Series(static_betas)
        for c in self.companies:
            resid_c = (
                self.returns.loc[self.trading_days, c]
                - static_betas[c] * self.mkt_ret.loc[self.trading_days]
                - static_alpha[c]
            )
            residuals[c] = resid_c

        resid = pd.DataFrame(residuals, index=self.trading_days)
        self.static_beta = static_beta
        self.static_alpha = static_alpha
        self.static_resid = resid
        return self.static_beta, self.static_alpha, self.static_resid
    
    def lag_weights(self, weights: pd.DataFrame):
        weights_lag = weights.shift(1).reindex(self.trading_days).fillna(0.0)
        weights_lag = weights_lag.reindex(columns=self.companies).fillna(0.0)
        return weights_lag

    def pnl_trading(self, weights: pd.DataFrame, tc_bps: float = 1.0):
        weights_lag = self.lag_weights(weights)
        pnl = (weights_lag * self.returns.loc[self.trading_days]).sum(axis=1)

        turnover = weights_lag.diff().abs().sum(axis=1)
        cost = (float(tc_bps) / 1e4) * turnover
        pnl_net = pnl - cost

        ann_factor = 252
        sharpe_gross = (pnl.mean() / pnl.std()) * (ann_factor ** 0.5)
        sharpe_net = (pnl_net.mean() / pnl_net.std()) * (ann_factor ** 0.5)

        equity_gross = (1 + pnl).cumprod()
        equity_net = (1 + pnl_net).cumprod()

        print("sharpe_gross = ", sharpe_gross)
        print("sharpe_net = ", sharpe_net)
        print("mean turnover:", turnover.mean())
        print("cost.sum() = ", cost.sum())
        print("equity_net.iloc[-1] = ", equity_net.iloc[-1])

        self.equity_gross = equity_gross
        self.equity_net = equity_net
        return self.equity_gross, self.equity_net