from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd


def load_universe_tickers(tickers_json_path: str | Path, universe: str) -> List[str]:
    import json

    cfg = json.loads(Path(tickers_json_path).read_text(encoding="utf-8"))
    if universe not in cfg:
        raise KeyError(f"Universe not found in tickers.json: {universe}")
    tickers = cfg[universe].get("tickers", [])
    if not isinstance(tickers, list):
        raise TypeError(f"Expected list tickers for universe={universe}, got {type(tickers)}")
    return [str(x).strip() for x in tickers if str(x).strip()]


def normalize_dt_index(df_or_s):
    idx = pd.to_datetime(df_or_s.index)
    if isinstance(idx, pd.DatetimeIndex) and idx.tz is not None:
        idx = idx.tz_convert(None)
    out = df_or_s.copy()
    out.index = idx
    return out


def load_pickle_if_exists(path: str | Path):
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_pickle(p)


def save_pickle(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(obj, p)


def load_prices_panel(panel_cache_path: str | Path) -> Optional[pd.DataFrame]:
    obj = load_pickle_if_exists(panel_cache_path)
    if obj is None:
        return None
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"Unexpected panel type at {panel_cache_path}: {type(obj)}")
    return obj


def load_market_prices(market_cache_path: str | Path) -> Optional[pd.Series]:
    obj = load_pickle_if_exists(market_cache_path)
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame) and obj.shape[1] == 1:
        obj = obj.iloc[:, 0]
    if not isinstance(obj, pd.Series):
        raise TypeError(f"Unexpected market cache type at {market_cache_path}: {type(obj)}")
    return obj


@dataclass
class FetchResult:
    prices: pd.DataFrame
    invalid_tickers: List[str]


def build_companies_prices_panel_from_ib(
    ib,
    tickers: Sequence[str],
    duration: str,
    bar_size: str = "1 day",
    batch_size: int = 50,
    exchange: str = "SMART",
    currency: str = "USD",
    progress: bool = True,
) -> FetchResult:
    """
    Build a prices panel for many tickers using existing ib_connection helper functions.
    Invalid tickers are skipped (so a single bad ticker won't fail the whole batch).
    """
    import ib_connection as IBC

    frames: List[pd.DataFrame] = []
    invalid_all: List[str] = []

    for i in range(0, len(tickers), int(batch_size)):
        chunk = list(tickers[i : i + int(batch_size)])
        chunk_valid, invalid = IBC.filter_valid_tickers(ib, chunk, exchange=exchange, currency=currency)
        invalid_all.extend(invalid)
        if not chunk_valid:
            continue

        tickers_list = list(chunk_valid)
        tickers_key = sorted(tickers_list)
        payload = {
            "name": "companies_prices_v2",
            "tickers": tickers_key,
            "duration": duration,
            "bar_size": bar_size,
            "exchange": exchange,
            "currency": currency,
        }
        px_chunk = IBC.fetch_with_cache(
            payload=payload,
            subdir="companies",
            compute_fn=lambda: IBC.fetch_companies_prices(
                ib,
                tickers=tickers_key,
                duration=duration,
                bar_size=bar_size,
                exchange=exchange,
                currency=currency,
                progress=progress,
            ),
            serializer="pickle",
            meta=payload,
        )
        if not isinstance(px_chunk, pd.DataFrame):
            raise TypeError(f"Expected DataFrame for prices chunk, got {type(px_chunk)}")
        px_chunk = px_chunk.reindex(columns=tickers_list)
        frames.append(px_chunk)

    if not frames:
        raise RuntimeError("No valid tickers were fetched. Check IB connectivity/universe.")

    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()].sort_index()
    return FetchResult(prices=prices, invalid_tickers=sorted(set(invalid_all)))


def fetch_market_prices_from_ib(
    ib,
    market: str,
    duration: str,
    bar_size: str = "1 day",
    exchange: str = "SMART",
    currency: str = "USD",
    mkt_params: Optional[dict] = None,
    progress: bool = True,
) -> pd.Series:
    import ib_connection as IBC

    payload = {
        "name": "mkt_prices_v2",
        "mkt_ticker": market,
        "duration": duration,
        "bar_size": bar_size,
        "exchange": exchange,
        "currency": currency,
        "mkt_params": mkt_params or {},
    }
    px = IBC.fetch_with_cache(
        payload=payload,
        subdir="mkt",
        compute_fn=lambda: IBC.fetch_mkt_prices(
            ib,
            mkt_ticker=market,
            duration=duration,
            bar_size=bar_size,
            exchange=exchange,
            currency=currency,
            mkt_params=mkt_params or {},
            progress=progress,
        ),
        serializer="pickle",
        meta=payload,
    )
    if isinstance(px, pd.DataFrame) and px.shape[1] == 1:
        px = px.iloc[:, 0]
    if not isinstance(px, pd.Series):
        raise TypeError(f"Expected Series for market prices, got {type(px)}")
    return px


def coverage_table(prices: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    present = [t for t in tickers if t in prices.columns]
    if not present:
        return pd.DataFrame(columns=["non_nan", "start", "end"])
    px = prices[present]
    return (
        pd.DataFrame(
            {
                "non_nan": px.notna().sum(axis=0),
                "start": px.apply(lambda s: s.first_valid_index()),
                "end": px.apply(lambda s: s.last_valid_index()),
            }
        )
        .sort_values(["non_nan"], ascending=True)
    )


def greedy_exclude_for_max_dates(
    prices: pd.DataFrame,
    mkt_px: pd.Series,
    tickers: Sequence[str],
) -> dict:
    """
    Strict intersection objective:
      maximize number of dates where *all remaining tickers* AND the market have prices.

    Greedy step:
      remove the ticker that unlocks the most additional valid dates.
    """
    prices = normalize_dt_index(prices)
    mkt_px = normalize_dt_index(mkt_px)

    # Keep only tickers that exist in the prices frame
    tickers = [t for t in tickers if t in prices.columns]
    px = prices[tickers]

    # Align on a common index
    common_idx = px.index.intersection(mkt_px.index)
    px = px.loc[common_idx]
    mkt_px = mkt_px.loc[common_idx]

    # Missing mask includes market
    miss = px.isna().astype(np.int16)
    miss_count = miss.sum(axis=1) + mkt_px.isna().astype(np.int16)

    valid = miss_count == 0
    history: List[Tuple[int, int, Optional[str], int]] = []

    # Pre-compute per-ticker missing rows (as booleans) for speed
    miss_bool = miss.astype(bool)

    removed: List[str] = []
    while True:
        valid_len = int(valid.sum())

        # A date becomes newly valid if it currently has exactly one missing value,
        # and that missing value belongs to the removed ticker (market missing is not removable).
        one_missing = miss_count == 1
        market_missing = mkt_px.isna()
        candidate_rows = one_missing & (~market_missing)

        if not bool(candidate_rows.any()):
            history.append((len(tickers), valid_len, None, 0))
            break

        improvements: Dict[str, int] = {t: int((candidate_rows & miss_bool[t]).sum()) for t in tickers}
        best_ticker = max(improvements, key=improvements.get)
        best_gain = improvements[best_ticker]

        history.append((len(tickers), valid_len, best_ticker, best_gain))
        if best_gain <= 0:
            break

        removed.append(best_ticker)
        tickers.remove(best_ticker)

        miss_count = miss_count - miss_bool[best_ticker].astype(np.int16)
        valid = miss_count == 0

    final_valid_dates = valid[valid].index
    return {
        "kept": tickers,
        "removed": removed,
        "valid_dates": final_valid_dates,
        "history": pd.DataFrame(history, columns=["n_tickers", "n_valid_dates", "removed_ticker", "gain_dates"]),
    }


def filter_tickers_by_min_valid_dates(
    prices: pd.DataFrame,
    tickers: Sequence[str],
    min_valid_dates: int = 2000,
) -> dict:
    """
    Filter tickers by per-ticker coverage in a prices panel.

    Definition:
        valid_dates(ticker) := count of non-NaN rows in prices[ticker]

    Returns:
        dict with:
          - kept: list[str]
          - removed: list[str]
          - stats: DataFrame(index=ticker) with columns [non_nan, start, end]
    """
    if prices is None or not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")

    px = normalize_dt_index(prices)
    present = [t for t in tickers if t in px.columns]

    stats = coverage_table(px, present)
    # coverage_table returns sorted ascending; we want original order preserved for kept/removed
    non_nan = px[present].notna().sum(axis=0).astype(int)

    kept = [t for t in present if int(non_nan.get(t, 0)) >= int(min_valid_dates)]
    removed = [t for t in present if t not in set(kept)]

    # Provide stats in a stable index order for downstream usage
    stats_out = pd.DataFrame(
        {
            "non_nan": non_nan.reindex(present),
            "start": px[present].apply(lambda s: s.first_valid_index()),
            "end": px[present].apply(lambda s: s.last_valid_index()),
        },
        index=present,
    )

    stats_kept = stats_out.loc[kept].sort_values("non_nan", ascending=True) if kept else stats_out.head(0)
    stats_removed = stats_out.loc[removed].sort_values("non_nan", ascending=True) if removed else stats_out.head(0)

    min_non_nan_kept = int(stats_kept["non_nan"].min()) if len(stats_kept) else 0

    return {
        "kept": kept,
        "removed": removed,
        "stats": stats_out,
        "stats_kept": stats_kept,
        "stats_removed": stats_removed,
        "min_non_nan_kept": min_non_nan_kept,
    }


def write_universe_to_tickers_json(
    tickers_json_path: str | Path,
    key: str,
    tickers: Sequence[str],
    description: str,
) -> None:
    import json

    p = Path(tickers_json_path)
    cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg[key] = {"description": description, "tickers": list(tickers)}
    p.write_text(json.dumps(cfg, indent=4), encoding="utf-8")


def save_filtered_prices_panel(
    prices: pd.DataFrame,
    tickers: Sequence[str],
    out_path: str | Path,
) -> pd.DataFrame:
    """
    Save a filtered companies prices panel (single merged cache file).

    Returns the filtered DataFrame (also written to out_path).
    """
    px = normalize_dt_index(prices)
    tickers = [t for t in tickers if t in px.columns]
    out = px[tickers].copy()
    save_pickle(out_path, out)
    return out


def filter_and_cache_sp500_by_min_valid_dates(
    comp_px: pd.DataFrame,
    sp500_tickers: Sequence[str],
    min_valid_dates: int = 2000,
    tickers_json_path: str | Path = "tickers.json",
    out_universe_key: str = "SP500_filtered",
    out_panel_path: str | Path | None = None,
) -> dict:
    """
    One-shot helper:
    - filter SP500 tickers by per-ticker non-NaN count >= min_valid_dates
    - write a new universe key into tickers.json
    - save the merged filtered companies panel into a single cache file

    Returns the filter result dict plus:
      - out_panel_path
      - filtered_panel_shape
    """
    if out_panel_path is None:
        out_panel_path = Path("cache") / "companies" / f"{out_universe_key}_prices_panel.pkl"

    result = filter_tickers_by_min_valid_dates(comp_px, sp500_tickers, min_valid_dates=min_valid_dates)
    kept = result["kept"]

    # Sanity check: by definition, all kept must satisfy the threshold.
    if kept and int(result.get("min_non_nan_kept", 0)) < int(min_valid_dates):
        raise RuntimeError(
            f"Filter invariant violated: min non_nan among kept is {result.get('min_non_nan_kept')} < {min_valid_dates}"
        )

    desc = f"SP500 filtered: tickers with >= {int(min_valid_dates)} valid dates (non-NaN prices)"
    write_universe_to_tickers_json(tickers_json_path, out_universe_key, kept, desc)

    filtered_panel = save_filtered_prices_panel(comp_px, kept, out_panel_path)
    result["out_panel_path"] = str(out_panel_path)
    result["filtered_panel_shape"] = tuple(filtered_panel.shape)
    return result

