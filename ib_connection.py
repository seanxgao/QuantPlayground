from ib_insync import IB, Stock, util
import time
from datetime import datetime
from pathlib import Path
import hashlib
import json
import pickle
from typing import Any, Dict, Optional, Tuple

import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

class IBConnection:

    def __init__(self, host='127.0.0.1', port=4002, client_id=3):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        self._connected = False

    def connect(self):
        if self._connected:
            return self.ib

        try:
            self.ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id
            )
            self._connected = True
            return self.ib
        except Exception as e:
            raise ConnectionError(f"Could not connect to IB at {self.host}:{self.port}") from e

    def disconnect(self):
        """Disconnect from TWS/Gateway."""
        if self._connected:
            self.ib.disconnect()
            self._connected = False

    def is_connected(self):
        """Check if connection is active."""
        return self._connected and self.ib.isConnected()


def quick_connect(host='127.0.0.1', port=4002):
    conn = IBConnection(host=host, port=port, client_id=1)
    ib = conn.connect()
    return conn, ib

import json
def show_universe(name, path="tickers.json"):
    with open(path, "r") as f:
        config = json.load(f)
    u = config[name]
    tickers = u["tickers"]
    print(f"===== Running universe: {name} ({len(tickers)} names) =====")
    print(f"Universe:{name}")
    print(f"Description : \n  {u['description']}")
    print(f"Tickers ({len(tickers)}):")
    print("  " + ", ".join(tickers))
    return tickers


def get_adj_close(ib, contract, duration="3 Y", bar_size="1 day"):
    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=duration,
        barSizeSetting=bar_size,
        whatToShow="ADJUSTED_LAST",
        useRTH=True,
        formatDate=1
    )
    df = util.df(bars)
    if df.empty:
        raise RuntimeError(f"No data for {contract.localSymbol or contract.symbol}")
    df = df.set_index("date")
    return df["close"]

def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _stable_json_dumps(obj: Any) -> str:
    """
    Deterministic JSON serialization for cache keys.

    Notes:
    - Uses sort_keys=True to make dict ordering stable.
    - Uses compact separators to avoid platform-specific whitespace differences.
    """
    return json.dumps(obj, separators=(",", ":"), sort_keys=True, default=str)


def _md5_key(payload: Any) -> str:
    return hashlib.md5(_stable_json_dumps(payload).encode("utf-8")).hexdigest()


def _cache_read_pickle(path: Path) -> Any:
    t0 = time.perf_counter()
    with path.open("rb") as f:
        obj = pickle.load(f)
    load_sec = time.perf_counter() - t0
    print(f"[cache] HIT  path={path} load_sec={load_sec:.3f} at={_now_str()}")
    return obj


def _cache_write_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[cache] writing... path={path} at={_now_str()}")
    t0 = time.perf_counter()
    with path.open("wb") as f:
        pickle.dump(obj, f)
    write_sec = time.perf_counter() - t0
    print(f"[cache] WRITE path={path} write_sec={write_sec:.3f} at={_now_str()}")


def _collect_invalid_contract_symbols(ib, contracts):
    """
    Try to qualify contracts and collect all invalid ones without stopping early.

    A contract is considered invalid if it cannot be qualified or ends up without a conId.
    Returns a sorted list of unique symbols that failed.
    """
    invalid = []

    # Fast path: try batch qualify first. Some IB setups may raise if any contract is bad.
    try:
        ib.qualifyContracts(*contracts)
    except Exception:
        # Fallback: qualify one-by-one to identify *all* problematic tickers.
        for c in contracts:
            try:
                ib.qualifyContracts(c)
            except Exception:
                invalid.append(getattr(c, "symbol", str(c)))
                continue

            if not getattr(c, "conId", 0):
                invalid.append(getattr(c, "symbol", str(c)))
    else:
        # Batch qualify succeeded; still verify each contract has a conId.
        for c in contracts:
            if not getattr(c, "conId", 0):
                invalid.append(getattr(c, "symbol", str(c)))

    # Deduplicate while keeping deterministic ordering for UX
    return sorted(set(invalid))


def fancy_fetch_with_cache(
    ib,
    tickers,
    duration="3 Y",
    bar_size="1 day",
    mkt_ticker="SPY",
    exchange="SMART",
    currency="USD",
    mkt_params: Optional[Dict[str, Any]] = None,
    progress=True,
):
    """
    Fetch adjusted close for tickers and market ticker, compute returns, and cache result.

    Returns:
        df: DataFrame with columns [<tickers...>, "mkt"] of returns aligned by date
        returns: DataFrame of ticker returns (df without "mkt")
        spy_ret: Series of market returns (df["mkt"])
    """
    returns = fetch_companies_returns_with_cache(
        ib,
        tickers=tickers,
        duration=duration,
        bar_size=bar_size,
        exchange=exchange,
        currency=currency,
        progress=progress,
    )

    mkt_df = fetch_mkt_returns_with_cache(
        ib,
        mkt_ticker=mkt_ticker,
        duration=duration,
        bar_size=bar_size,
        exchange=exchange,
        currency=currency,
        mkt_params=mkt_params,
        progress=progress,
    )

    tickers_list = list(tickers)
    returns_out = returns.reindex(columns=tickers_list).copy() if set(tickers_list).issubset(set(returns.columns)) else returns.copy()
    df = returns_out.join(mkt_df, how="inner")
    spy_ret = df["mkt"].copy()
    return df, df.drop(columns=["mkt"]).copy(), spy_ret


def fetch_companies_returns_with_cache(
    ib,
    tickers,
    duration="3 Y",
    bar_size="1 day",
    exchange="SMART",
    currency="USD",
    progress=True,
) -> pd.DataFrame:
    """
    Fetch companies returns only (no market), with caching.

    Returns:
        returns: DataFrame of companies returns aligned by date.
    """
    # Cache location is fixed to ./cache for simplicity.
    cache_dir_path = Path("cache")
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    tickers_list = list(tickers)
    tickers_key = sorted(tickers_list)

    companies_payload = {
        "name": "companies_returns_v2",
        "tickers": tickers_key,
        "duration": duration,
        "bar_size": bar_size,
        "exchange": exchange,
        "currency": currency,
    }
    companies_cache_path = cache_dir_path / "companies" / f"{_md5_key(companies_payload)}.pkl"

    if companies_cache_path.exists():
        returns = _cache_read_pickle(companies_cache_path)
        if not isinstance(returns, pd.DataFrame):
            raise RuntimeError(f"Companies cache is not a DataFrame: {companies_cache_path}")
        # Ensure deterministic column order for downstream users
        returns = returns.reindex(columns=tickers_list).copy() if set(tickers_list).issubset(set(returns.columns)) else returns.copy()
        return returns

    print(f"[cache] MISS companies=True mkt=False at={_now_str()}")

    # ---- all api: tickers ----
    contracts = [Stock(t, exchange, currency) for t in tickers_key]
    invalid_tickers = _collect_invalid_contract_symbols(ib, contracts)
    if invalid_tickers:
        raise ValueError(
            "Invalid tickers detected (contract qualification failed): "
            + ", ".join(invalid_tickers)
        )

    if (tqdm is None) or (not progress):
        print(f"[api] fetching adjusted close for {len(contracts)} tickers... at={_now_str()}")

    iterable = contracts
    if (tqdm is not None) and progress:
        iterable = tqdm(contracts, desc="Fetching adjusted close", unit="ticker")

    price_frames = {}
    for c in iterable:
        price_frames[c.symbol] = get_adj_close(ib, c, duration, bar_size)

    prices = pd.concat(price_frames, axis=1).dropna()
    returns = prices.pct_change().dropna()

    _cache_write_pickle(companies_cache_path, returns)

    returns = returns.reindex(columns=tickers_list).copy() if set(tickers_list).issubset(set(returns.columns)) else returns.copy()
    return returns


def fetch_mkt_returns_with_cache(
    ib,
    mkt_ticker="SPY",
    duration="3 Y",
    bar_size="1 day",
    exchange="SMART",
    currency="USD",
    mkt_params: Optional[Dict[str, Any]] = None,
    progress=True,
) -> pd.DataFrame:
    """
    Fetch market returns only, with caching.

    Returns:
        mkt_df: DataFrame with a single column "mkt" of market returns.
    """
    # Cache location is fixed to ./cache for simplicity.
    cache_dir_path = Path("cache")
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    mkt_payload = {
        "name": "mkt_returns_v2",
        "mkt_ticker": mkt_ticker,
        "duration": duration,
        "bar_size": bar_size,
        "exchange": exchange,
        "currency": currency,
        "mkt_params": mkt_params or {},
    }
    mkt_cache_path = cache_dir_path / "mkt" / f"{_md5_key(mkt_payload)}.pkl"

    if mkt_cache_path.exists():
        mkt_df = _cache_read_pickle(mkt_cache_path)
        if isinstance(mkt_df, pd.Series):
            mkt_df = mkt_df.to_frame("mkt")
        if not isinstance(mkt_df, pd.DataFrame):
            raise RuntimeError(f"Market cache is not a DataFrame/Series: {mkt_cache_path}")
        if "mkt" not in mkt_df.columns:
            if mkt_df.shape[1] == 1:
                mkt_df = mkt_df.rename(columns={mkt_df.columns[0]: "mkt"})
            else:
                raise RuntimeError(f"Market cache missing 'mkt' column: {mkt_cache_path}")
        return mkt_df.copy()

    print(f"[cache] MISS companies=False mkt=True at={_now_str()}")

    if (tqdm is None) or (not progress):
        print(f"[api] fetching {mkt_ticker}... at={_now_str()}")

    mkt = Stock(mkt_ticker, exchange, currency)
    invalid_mkt = _collect_invalid_contract_symbols(ib, [mkt])
    if invalid_mkt:
        raise ValueError(
            "Invalid market ticker detected (contract qualification failed): "
            + ", ".join(invalid_mkt)
        )

    mkt_px = get_adj_close(ib, mkt, duration, bar_size)
    mkt_df = mkt_px.pct_change().dropna().rename("mkt").to_frame()

    _cache_write_pickle(mkt_cache_path, mkt_df)
    return mkt_df.copy()