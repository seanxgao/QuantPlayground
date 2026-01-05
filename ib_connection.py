from ib_insync import IB, Stock, util
import time
from datetime import datetime
from pathlib import Path
import hashlib
import json
import pickle
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import numpy as np

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


def quick_connect(client_id=1):
    conn = IBConnection(host='127.0.0.1', port=4002, client_id= client_id)
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


def load_universe(name: str, path: str = "tickers.json") -> Dict[str, Any]:
    """
    Load universe config without printing.

    Returns:
        dict with keys including: description, tickers
    """
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    if name not in config:
        raise KeyError(f"Universe not found: {name}")
    return config[name]


def load_companies_prices_panel(universe: str, cache_dir: str = "cache/companies") -> pd.DataFrame:
    """
    Load a pre-assembled companies prices panel from a single cache file:
        {cache_dir}/{universe}_prices_panel.pkl
    """
    path = Path(cache_dir) / f"{universe}_prices_panel.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Companies panel cache not found: {path}")
    obj = pd.read_pickle(path)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Companies panel cache is not a DataFrame: {path}")
    return obj


def load_market_prices_cache(mkt_ticker: str, cache_dir: str = "cache/mkt") -> pd.Series:
    """
    Load a cached market prices Series from:
        {cache_dir}/{mkt_ticker}_prices.pkl
    """
    path = Path(cache_dir) / f"{mkt_ticker}_prices.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Market price cache not found: {path}")
    obj = pd.read_pickle(path)
    if isinstance(obj, pd.DataFrame) and obj.shape[1] == 1:
        obj = obj.iloc[:, 0]
    if not isinstance(obj, pd.Series):
        raise TypeError(f"Market price cache is not a Series: {path}")
    return obj


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
    if bars is None or (hasattr(bars, "__len__") and len(bars) == 0):
        raise RuntimeError(f"No bars returned for {contract.localSymbol or contract.symbol}")

    df = util.df(bars)
    if df is None or getattr(df, "empty", True):
        raise RuntimeError(f"No data for {contract.localSymbol or contract.symbol}")
    # Normalize to pandas datetime to avoid mixed `datetime.date` vs `Timestamp` index issues downstream.
    df["date"] = pd.to_datetime(df["date"])
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


def fetch_with_cache(
    *,
    payload: Dict[str, Any],
    subdir: str,
    compute_fn,
    serializer: str = "pickle",
    ext: Optional[str] = None,
    meta: Optional[Any] = None,
    verbose: bool = True,
) -> Any:
    """
    Generic cache helper.

    This is intentionally lightweight so it can be reused for non-fetch artifacts
    (e.g. precomputed matrices, intermediate research objects).

    Parameters
    ----------
    payload : dict
        Dict used to build a stable cache key (md5 of stable JSON).
    subdir : str
        Sub-directory under ./cache/ (e.g. "companies", "mkt", "graphs").
    compute_fn : callable
        Function with no args that computes the object to cache.
    serializer : {"pickle","json","npz"}
        Storage format.
    ext : str or None
        File extension override. Defaults to "pkl"/"json"/"npz" based on serializer.
    meta : any or None
        Optional metadata to write to a sidecar file "<cache>.meta.txt".
    verbose : bool
        Print HIT/MISS logs.
    """
    cache_root = Path("cache") / subdir
    cache_root.mkdir(parents=True, exist_ok=True)

    serializer = str(serializer).lower().strip()
    if ext is None:
        ext = {"pickle": "pkl", "json": "json", "npz": "npz"}.get(serializer, "pkl")

    key = _md5_key(payload)
    cache_path = cache_root / f"{key}.{ext}"
    meta_path = cache_root / f"{key}.meta.txt"

    def _load() -> Any:
        if serializer == "pickle":
            return _cache_read_pickle(cache_path)
        if serializer == "json":
            t0 = time.perf_counter()
            with cache_path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            load_sec = time.perf_counter() - t0
            if verbose:
                print(f"[cache] HIT  path={cache_path} load_sec={load_sec:.3f} at={_now_str()}")
            return obj
        if serializer == "npz":
            t0 = time.perf_counter()
            obj = dict(np.load(cache_path, allow_pickle=True))
            load_sec = time.perf_counter() - t0
            if verbose:
                print(f"[cache] HIT  path={cache_path} load_sec={load_sec:.3f} at={_now_str()}")
            return obj
        raise ValueError(f"Unknown serializer: {serializer}")

    def _save(obj: Any) -> None:
        if meta is not None:
            meta_path.write_text(_stable_json_dumps(meta), encoding="utf-8")
        if serializer == "pickle":
            _cache_write_pickle(cache_path, obj)
            return
        if serializer == "json":
            if verbose:
                print(f"[cache] writing... path={cache_path} at={_now_str()}")
            t0 = time.perf_counter()
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False)
            write_sec = time.perf_counter() - t0
            if verbose:
                print(f"[cache] WRITE path={cache_path} write_sec={write_sec:.3f} at={_now_str()}")
            return
        if serializer == "npz":
            if verbose:
                print(f"[cache] writing... path={cache_path} at={_now_str()}")
            t0 = time.perf_counter()
            if isinstance(obj, dict):
                np.savez_compressed(cache_path, **obj)
            else:
                np.savez_compressed(cache_path, arr=obj)
            write_sec = time.perf_counter() - t0
            if verbose:
                print(f"[cache] WRITE path={cache_path} write_sec={write_sec:.3f} at={_now_str()}")
            return
        raise ValueError(f"Unknown serializer: {serializer}")

    if cache_path.exists():
        return _load()

    if verbose:
        print(f"[cache] MISS path={cache_path} at={_now_str()}")

    obj = compute_fn()
    _save(obj)
    return obj


def fetch_companies_prices(
    ib,
    tickers,
    duration: str = "3 Y",
    bar_size: str = "1 day",
    exchange: str = "SMART",
    currency: str = "USD",
    progress: bool = True,
) -> pd.DataFrame:
    """
    Fetch companies adjusted close prices only (no cache).

    Returns a DataFrame of adjusted close prices (may contain NaNs).
    """
    tickers_key = sorted(list(tickers))
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

    price_frames: Dict[str, pd.Series] = {}
    failed = []
    for c in iterable:
        try:
            price_frames[c.symbol] = get_adj_close(ib, c, duration, bar_size)
        except Exception:
            failed.append(c.symbol)
            continue

    if failed:
        failed = sorted(set(failed))
        print(f"[api] WARNING: no data for {len(failed)} tickers (showing up to 30): {failed[:30]}")

    if not price_frames:
        raise RuntimeError("No company price series were fetched (all tickers failed).")

    prices = pd.concat(price_frames, axis=1).sort_index()
    return prices


def fetch_mkt_prices(
    ib,
    mkt_ticker: str = "SPY",
    duration: str = "3 Y",
    bar_size: str = "1 day",
    exchange: str = "SMART",
    currency: str = "USD",
    mkt_params: Optional[Dict[str, Any]] = None,
    progress: bool = True,
) -> pd.Series:
    """
    Fetch market adjusted close prices only (no cache).
    """
    if (tqdm is None) or (not progress):
        print(f"[api] fetching {mkt_ticker}... at={_now_str()}")

    mkt = Stock(mkt_ticker, exchange, currency)
    invalid_mkt = _collect_invalid_contract_symbols(ib, [mkt])
    if invalid_mkt:
        raise ValueError(
            "Invalid market ticker detected (contract qualification failed): "
            + ", ".join(invalid_mkt)
        )

    # mkt_params is currently unused for Stock/reqHistoricalData, but we keep it in the signature
    # so callers can include it in cache keys for future extensions.
    _ = mkt_params or {}
    mkt_px = get_adj_close(ib, mkt, duration, bar_size).sort_index()
    return mkt_px.copy()


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


def filter_valid_tickers(ib, tickers, exchange="SMART", currency="USD"):
    """
    Validate a list of ticker strings via IB contract qualification.

    Returns:
        valid: list[str]
        invalid: list[str]
    """
    tickers_list = list(tickers)
    contracts = [Stock(t, exchange, currency) for t in tickers_list]
    invalid = set(_collect_invalid_contract_symbols(ib, contracts))
    valid = [t for t in tickers_list if t not in invalid]
    return valid, sorted(invalid)
