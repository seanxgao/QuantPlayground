import pandas as pd
import numpy as np
import copy
from typing import Dict, List, Optional

class Records:
    def __init__(self):
        # ---- config / inputs ----
        self.tickers = None
        self.dur = None
        self.size = None
        self.t = None  # training length (number of rows)
        self.long_window = None
        self.short_window = None
        self.mkt_tickers = ["SPY"]
        self.mkt_params_by_ticker: Dict[str, dict] = {}

        # ---- data ----
        # Raw prices (fetched/cached). No computations should happen during fetch.
        self.comp_px = None  # companies adjusted close prices DataFrame
        self.mkt_px = None  # market adjusted close prices DataFrame, columns = mkt_tickers

        # Derived data (computed in prepare_data / set_all).
        self.returns = None  # companies returns DataFrame
        self.mkt_ret = None  # market returns DataFrame, columns = mkt_tickers

        # ---- derived calendar ----
        # NOTE:
        # This codebase uses a SINGLE unified calendar:
        # - `dates`: dates where ALL companies and ALL markets are simultaneously available
        # - `training_days`: first `t` rows of `dates`
        # - `trading_days`: remaining rows of `dates`
        self.companies = None
        self.dates: pd.DatetimeIndex | None = None
        self.training_days: pd.DatetimeIndex | None = None
        self.trading_days: pd.DatetimeIndex | None = None

        # ---- regression outputs ----
        # Static outputs per market
        self.static_resid: Dict[str, pd.DataFrame] = {}
        # Index = companies, columns = markets
        self.static_beta = None
        self.static_alpha = None

        # Rolling outputs per market: market -> DataFrame(index=dates, columns=companies)
        self.short_beta: Dict[str, pd.DataFrame] = {}
        self.short_alpha: Dict[str, pd.DataFrame] = {}
        self.long_beta: Dict[str, pd.DataFrame] = {}
        self.long_alpha: Dict[str, pd.DataFrame] = {}

        # ---- blend outputs (dangerous) ----
        self.blend_beta: Dict[str, pd.DataFrame] = {}
        self.blend_alpha: Dict[str, pd.DataFrame] = {}

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
        raise RuntimeError("set_calendar() is deprecated; use prepare_data()/set_all() which populate per-market calendars.")

    def _default_mkt(self) -> str:
        if not self.mkt_tickers:
            raise ValueError("mkt_tickers is empty.")
        return self.mkt_tickers[0]

    def get_dates(self) -> pd.DatetimeIndex:
        """
        Unified calendar where ALL companies and ALL markets are simultaneously available.
        Populated by set_all()/prepare_data().
        """
        if self.dates is None:
            raise KeyError("Dates not prepared. Call set_all() first.")
        return self.dates

    def get_training_days(self) -> pd.DatetimeIndex:
        """
        First `t` rows of `dates`.
        """
        if self.training_days is None:
            raise KeyError("Training days not prepared. Call set_all() first.")
        return self.training_days

    def get_trading_days(self) -> pd.DatetimeIndex:
        """
        Trading days = dates[t:].
        """
        if self.trading_days is None:
            raise KeyError("Trading days not prepared. Call set_all() first.")
        return self.trading_days

    @staticmethod
    def _ensure_datetime_index(obj):
        """
        Ensure obj.index is a tz-naive pandas.DatetimeIndex.
        Works for pandas Series/DataFrame. No-op for None.
        """
        if obj is None:
            return None
        idx = pd.to_datetime(obj.index)
        if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
            idx = idx.tz_convert(None)
        obj.index = idx
        return obj

    # =========================
    # Data fetch + cache
    # =========================
    def fetch_df(self, ib, progress=True):
        """
        Legacy convenience wrapper.

        Fetch companies prices + market prices only (no computations).

        Notes:
        - Does NOT compute returns, does NOT set calendar.
        - Populates: comp_px, mkt_px
        """
        self.fetch_comp(ib, progress=progress)
        self.fetch_mkt(ib, progress=progress)
        return self.comp_px, self.mkt_px

    def fetch_comp(self, ib, progress=True):
        """
        Fetch companies prices only (no market), with caching.

        Populates: comp_px only. No computations are performed here.
        """
        import ib_connection as IBC

        tickers_list = list(self.tickers)
        tickers_key = sorted(tickers_list)

        payload = {
            "name": "companies_prices_v2",
            "tickers": tickers_key,
            "duration": self.dur,
            "bar_size": self.size,
        }

        prices = IBC.fetch_with_cache(
            payload=payload,
            subdir="companies",
            compute_fn=lambda: IBC.fetch_companies_prices(
                ib,
                tickers=tickers_key,
                duration=self.dur,
                bar_size=self.size,
                progress=progress,
            ),
            serializer="pickle",
            meta=payload,
        )
        if not isinstance(prices, pd.DataFrame):
            raise RuntimeError("Cached companies prices is not a DataFrame.")
        prices = prices.reindex(columns=tickers_list)

        # Strict rule: fetch should only store raw data.
        self.comp_px = prices
        return self.comp_px

    def fetch_mkt(
        self,
        ib,
        progress=True,
        mkt_tickers: Optional[List[str]] = None,
        mkt_params_by_ticker: Optional[Dict[str, dict]] = None,
    ):
        """
        Fetch market prices only, with caching.

        Populates: mkt_px only. No computations are performed here.
        """
        import ib_connection as IBC

        if mkt_tickers is not None:
            self.mkt_tickers = list(mkt_tickers)
        if mkt_params_by_ticker is not None:
            self.mkt_params_by_ticker = dict(mkt_params_by_ticker)

        frames = {}
        for tkr in self.mkt_tickers:
            params = self.mkt_params_by_ticker.get(tkr, {})
            payload = {
                "name": "mkt_prices_v2",
                "mkt_ticker": tkr,
                "duration": self.dur,
                "bar_size": self.size,
                "mkt_params": params or {},
            }
            px = IBC.fetch_with_cache(
                payload=payload,
                subdir="mkt",
                compute_fn=lambda tkr=tkr, params=params: IBC.fetch_mkt_prices(
                    ib,
                    mkt_ticker=tkr,
                    duration=self.dur,
                    bar_size=self.size,
                    mkt_params=params or {},
                    progress=progress,
                ),
                serializer="pickle",
                meta=payload,
            )
            if isinstance(px, pd.DataFrame) and px.shape[1] == 1:
                px = px.iloc[:, 0]
            if not isinstance(px, pd.Series):
                raise RuntimeError(f"Cached market prices is not a Series: {tkr}")
            frames[tkr] = px

        # Strict rule: fetch should only store raw data.
        # Outer join to keep raw availability; alignment happens in prepare_data().
        self.mkt_px = pd.concat(frames, axis=1).sort_index()
        self.mkt_px.columns = list(frames.keys())
        return self.mkt_px

    # =========================
    # Load from prebuilt caches
    # =========================
    def load_cached(
        self,
        universe: str,
        markets: Optional[List[str]] = None,
        tickers_path: str = "tickers.json",
    ):
        """
        Load prices from prebuilt cache files (no IB fetch).

        Conventions:
        - Companies panel: cache/companies/{universe}_prices_panel.pkl
        - Market prices:   cache/mkt/{mkt}_prices.pkl

        After calling, `comp_px` and `mkt_px` are populated and you can call `set_all()`.
        """
        import ib_connection as IBC

        u = IBC.load_universe(universe, path=tickers_path)
        tickers = list(u.get("tickers", []))
        if not tickers:
            raise ValueError(f"Universe has no tickers: {universe}")
        self.tickers = tickers

        if markets is not None:
            self.mkt_tickers = list(markets)
        if not self.mkt_tickers:
            self.mkt_tickers = ["SPY"]

        # Companies panel (single merged cache file)
        comp_px = IBC.load_companies_prices_panel(universe)
        # Subset to the universe tickers in order; missing columns become all-NaN columns
        self.comp_px = comp_px.reindex(columns=self.tickers)

        # Markets (single series cache per market)
        mkt_frames = {}
        for m in self.mkt_tickers:
            mkt_frames[m] = IBC.load_market_prices_cache(m)
        self.mkt_px = pd.concat(mkt_frames, axis=1).sort_index()
        self.mkt_px.columns = list(mkt_frames.keys())
        return self.comp_px, self.mkt_px

    def prepare_data(self):
        """
        Prepare aligned returns and calendar from raw prices.

        Steps:
        - Normalize indexes to DatetimeIndex
        - Build a single unified calendar where ALL companies and ALL markets are present
        - Compute returns for companies and market
        - Align returns again (pct_change drops the first row)
        - Populate calendar fields
        """
        if self.comp_px is None or self.mkt_px is None:
            raise ValueError("Missing data: call fetch_comp() and fetch_mkt() before prepare_data()/set_all().")

        comp_px = self._ensure_datetime_index(self.comp_px.copy()).sort_index()
        mkt_px = self._ensure_datetime_index(self.mkt_px.copy()).sort_index()

        if not self.mkt_tickers:
            raise ValueError("mkt_tickers is empty.")

        # 1) companies: dates where ALL companies have prices
        comp_good = comp_px.dropna(axis=0, how="any")
        common_dates = comp_good.index

        # 2) markets: intersect dates where EACH market has prices
        for tkr in self.mkt_tickers:
            if tkr not in mkt_px.columns:
                raise ValueError(f"Missing market price column: {tkr}.")
            m_good = mkt_px[tkr].dropna()
            common_dates = common_dates.intersection(m_good.index)

        common_dates = pd.DatetimeIndex(common_dates).sort_values()
        if len(common_dates) == 0:
            raise ValueError("No common dates after intersecting companies and markets.")

        # 3) subset prices to unified calendar, then compute returns
        comp_px_aligned = comp_px.reindex(common_dates)
        mkt_px_aligned = mkt_px.reindex(common_dates)

        returns = comp_px_aligned.pct_change().dropna(axis=0, how="any")
        mkt_ret = mkt_px_aligned.pct_change().dropna(axis=0, how="any")

        # pct_change drops the first row; enforce exact alignment
        common_ret_dates = returns.index.intersection(mkt_ret.index)
        returns = returns.reindex(common_ret_dates)
        mkt_ret = mkt_ret.reindex(common_ret_dates)

        t = int(self.t)
        if t <= 0:
            raise ValueError(f"Invalid training length t={t}. Must be positive.")
        if len(common_ret_dates) <= t:
            raise ValueError(f"Not enough aligned rows: have {len(common_ret_dates)}, need > t={t}.")

        # 4) populate unified calendar + back-compat dicts
        self.returns = returns
        self.mkt_ret = mkt_ret
        self.companies = self.returns.columns

        self.dates = self.returns.index
        self.training_days = self.dates[:t]
        self.trading_days = self.dates[t:]

        print("len(returns) =", len(self.returns))
        print("markets      =", list(self.mkt_tickers))
        print("t            =", self.t)
        print("date range   =", self.dates.min(), "->", self.dates.max())
    
    def rolling_beta_alpha(self, wind: int, mkt: Optional[str] = None):
        wind = int(wind)
        m = mkt or self._default_mkt()
        if self.returns is None or self.mkt_ret is None:
            raise ValueError("Missing data: call prepare_data()/set_all() before rolling_beta_alpha().")
        if self.dates is None:
            raise KeyError("Dates not prepared. Call set_all() first.")
        if m not in self.mkt_ret.columns:
            raise KeyError(f"Market not prepared: {m}. Call set_all() first.")

        dates = self.dates
        R = self.returns.reindex(dates).astype(float)
        M = self.mkt_ret[m].reindex(dates).astype(float)

        rolling_alpha = pd.DataFrame(index=dates, columns=self.companies, dtype=float)
        rolling_beta = pd.DataFrame(index=dates, columns=self.companies, dtype=float)

        mkt_var = M.rolling(wind).var()
        mkt_mean = M.rolling(wind).mean()
        for c in self.companies:
            cov = R[c].rolling(wind).cov(M)
            rolling_beta[c] = cov / mkt_var
            rolling_alpha[c] = R[c].rolling(wind).mean() - rolling_beta[c] * mkt_mean
        return rolling_beta, rolling_alpha
    
    def set_long_beta_alpha(self, mkt: Optional[str] = None):
        m = mkt or self._default_mkt()
        beta, alpha = self.rolling_beta_alpha(self.long_window, mkt=m)
        self.long_beta[m] = beta
        self.long_alpha[m] = alpha
        return self.long_beta[m], self.long_alpha[m]

    def set_short_beta_alpha(self, mkt: Optional[str] = None):
        m = mkt or self._default_mkt()
        beta, alpha = self.rolling_beta_alpha(self.short_window, mkt=m)
        self.short_beta[m] = beta
        self.short_alpha[m] = alpha
        return self.short_beta[m], self.short_alpha[m]

    def set_static_beta_alpha_residual(self, mkt: Optional[str] = None):
        # Prefer statsmodels when available; fall back to a lightweight OLS implementation otherwise.
        try:
            import statsmodels.api as sm  # type: ignore
        except Exception:
            sm = None

        m = mkt or self._default_mkt()
        if self.returns is None or self.mkt_ret is None:
            raise ValueError("Missing data: call prepare_data()/set_all() before set_static_beta_alpha_residual().")
        if self.dates is None or self.training_days is None or self.trading_days is None:
            raise KeyError("Calendar not prepared. Call set_all() first.")
        if m not in self.mkt_ret.columns:
            raise KeyError(f"Market not prepared: {m}. Call set_all() first.")

        dates = self.dates
        training_days = self.training_days
        trading_days = self.trading_days

        R = self.returns.reindex(dates).astype(float)
        M = self.mkt_ret[m].reindex(dates).astype(float)

        static_alpha: Dict[str, float] = {}
        static_betas: Dict[str, float] = {}
        residuals: Dict[str, pd.Series] = {}

        # Fit on training window
        x = M.loc[training_days].astype(float)
        x_mean = float(x.mean())
        x_var = float(x.var(ddof=0))
        if x_var <= 0:
            x_var = 1e-12

        for c in self.companies:
            y = R[c].loc[training_days].astype(float)
            if sm is not None:
                X = sm.add_constant(x.values)
                model = sm.OLS(y.values, X).fit()
                static_betas[c] = float(model.params[1])
                static_alpha[c] = float(model.params[0])
            else:
                # OLS with intercept: y = a + b x
                y_mean = float(y.mean())
                cov_xy = float(((x - x_mean) * (y - y_mean)).mean())
                b = cov_xy / x_var
                a = y_mean - b * x_mean
                static_betas[c] = float(b)
                static_alpha[c] = float(a)

        # Store static beta/alpha as DataFrame: index=companies, columns=mkts
        beta_col = pd.Series(static_betas, name=m)
        alpha_col = pd.Series(static_alpha, name=m)
        if self.static_beta is None:
            self.static_beta = beta_col.to_frame()
        else:
            self.static_beta[m] = beta_col
        if self.static_alpha is None:
            self.static_alpha = alpha_col.to_frame()
        else:
            self.static_alpha[m] = alpha_col

        for c in self.companies:
            resid_c = (
                R[c].loc[trading_days]
                - static_betas[c] * M.loc[trading_days]
                - static_alpha[c]
            )
            residuals[c] = resid_c

        resid = pd.DataFrame(residuals, index=trading_days)
        self.static_resid[m] = resid
        return self.static_beta, self.static_alpha, self.static_resid[m]
    
    def set_all(self, mkts: Optional[List[str]] = None):
        """
        Prepare data once, then compute all betas/alphas/residuals.

        By default runs for all markets in `mkt_tickers`.
        """
        self.prepare_data()
        targets = mkts or list(self.mkt_tickers)
        for m in targets:
            if self.mkt_ret is None or m not in self.mkt_ret.columns:
                continue
            self.set_long_beta_alpha(mkt=m)
            self.set_short_beta_alpha(mkt=m)
            self.set_static_beta_alpha_residual(mkt=m)
    
    def lag_weights(self, weights: pd.DataFrame):
        trading_days = self.get_trading_days()
        weights_lag = weights.shift(1).reindex(trading_days).fillna(0.0)
        weights_lag = weights_lag.reindex(columns=self.companies).fillna(0.0)
        return weights_lag

    def pnl_trading(self, weights: pd.DataFrame, tc_bps: float = 1.0):
        trading_days = self.get_trading_days()
        weights_lag = self.lag_weights(weights)
        pnl = (weights_lag * self.returns.loc[trading_days]).sum(axis=1)

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