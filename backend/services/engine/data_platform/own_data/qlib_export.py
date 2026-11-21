"""Build a release-pinned whole-universe Qlib provider from integrated local data.

Stock adjusted close is the frozen vendor HFQ close; factor=HFQ/raw. Amount is
CNY and Qlib volume is raw shares/factor. This preserves reversible units while
leaving missing sessions as NaN, without certifying corporate-action truth.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from backend.shared.stock_utils import StockCodeUtil
from .sync import atomic_json


def build_provider(con, target, *, symbols=None, start=None, end=None):
    target = Path(target)
    from .warehouse import literal

    def where(alias=""):
        prefix = alias + "." if alias else ""
        clauses = []
        if symbols is not None:
            if not symbols:
                raise ValueError("Qlib task has an empty universe")
            clauses.append(prefix + "symbol IN (" + ",".join(literal(StockCodeUtil.normalize(s)) for s in symbols) + ")")
        if start:
            clauses.append(prefix + "time::DATE>=" + literal(str(start)) + "::DATE")
        if end:
            clauses.append(prefix + "time::DATE<=" + literal(str(end)) + "::DATE")
        return " WHERE " + " AND ".join(clauses) if clauses else ""
    for name in ["calendars", "instruments", "features"]:
        (target / name).mkdir(parents=True, exist_ok=True)
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    dates = set()
    for table in [
        "quantdb_daily_unadjusted",
        "quantdb_etf_kline",
        "quantdb_index_daily",
    ]:
        if table in tables:
            dates.update(
                r[0]
                for r in con.execute(
                    f"SELECT DISTINCT time::DATE FROM {table}" + where()
                ).fetchall()
            )
    # Qlib's executor asks for calendar[index+1] even on the last evaluation day.
    # A known next session is calendar metadata, not a fabricated future bar.
    if end and "quantdb_trading_calendar" in tables:
        calendar_date = "coalesce(try_cast(TradingDate AS DATE),try_strptime(CAST(TradingDate AS VARCHAR),'%Y%m%d')::DATE)"
        following = con.execute(
            f"SELECT min({calendar_date}) FROM quantdb_trading_calendar WHERE IsTradingDay=1 AND {calendar_date}>?::DATE",
            [str(end)],
        ).fetchone()[0]
        if following:
            dates.add(following)
    dates = sorted(dates)
    if not dates:
        raise ValueError("No observed daily sessions for Qlib export")
    index = {pd.Timestamp(day): i for i, day in enumerate(dates)}
    (target / "calendars/day.txt").write_text("\n".join(str(d) for d in dates) + "\n")
    instruments = []
    summary = {"stock": 0, "etf": 0, "index": 0, "skipped_rows": 0}

    def dump(frame, kind):
        if frame.empty:
            return
        if frame.duplicated(["symbol", "trade_date"]).any():
            raise ValueError("Duplicate Qlib bars")
        for symbol, group in frame.groupby("symbol", sort=True):
            group = group.sort_values("trade_date").reset_index(drop=True)
            positions = group.trade_date.map(index).to_numpy(dtype=np.int64)
            factor = group.factor.to_numpy(dtype=float)
            valid = np.isfinite(factor) & (factor > 0)
            for field in ["open", "high", "low", "close"]:
                value = group[field].to_numpy(dtype=float)
                valid &= np.isfinite(value) & (value > 0)
            valid &= (
                group.high >= group[["open", "close", "low"]].max(axis=1)
            ).to_numpy()
            valid &= (group.low <= group[["open", "close"]].min(axis=1)).to_numpy()
            for field in ["volume", "amount"]:
                value = group[field].to_numpy(dtype=float)
                valid &= np.isfinite(value) & (value >= 0)
            summary["skipped_rows"] += int((~valid).sum())
            if not valid.any():
                continue
            start, end = int(positions.min()), int(positions.max())
            span = end - start + 1
            code = StockCodeUtil.to_qlib(symbol)
            if not re.fullmatch(r"(sh|sz|bj)\d{6}", code):
                raise ValueError("Unexpected instrument identifier")
            folder = target / "features" / code
            folder.mkdir(exist_ok=True)
            values = {
                f: group[f].to_numpy(dtype=float)
                for f in ["open", "high", "low", "close", "amount"]
            }
            values.update(
                volume=group.volume.to_numpy(dtype=float) / factor, factor=factor
            )
            aligned = {}
            for field, array in values.items():
                vector = np.full(span, np.nan, dtype="<f4")
                vector[positions[valid] - start] = array[valid]
                aligned[field] = vector
            change = np.full(span, np.nan, dtype="<f4")
            with np.errstate(invalid="ignore", divide="ignore"):
                change[1:] = aligned["close"][1:] / aligned["close"][:-1] - 1
            aligned["change"] = change
            for field, array in aligned.items():
                np.r_[np.float32(start), array].astype("<f4").tofile(
                    folder / (field + ".day.bin")
                )
            instruments.append(f"{code}\t{dates[start]}\t{dates[end]}")
            summary[kind] += 1
            if summary[kind] % 1000 == 0:
                print("qlib", kind, summary[kind], flush=True)

    stock = con.execute("""SELECT h.symbol,h.time::DATE AS trade_date,h.open,h.high,h.low,h.close,
        r.volume,r.amount*10000 AS amount,h.close/nullif(r.close,0) AS factor
        FROM quantdb_daily_backward h JOIN quantdb_daily_unadjusted r
        ON h.symbol=r.symbol AND h.time=r.time""" + where("h") + " ORDER BY h.symbol,h.time").fetchdf()
    dump(stock, "stock")
    del stock
    for table, kind in [("quantdb_etf_kline", "etf"), ("quantdb_index_daily", "index")]:
        if table not in tables:
            continue
        frame = con.execute(
            f"SELECT symbol,time::DATE AS trade_date,open,high,low,close,volume,amount*10000 AS amount,1.0 AS factor FROM {table}" + where() + " ORDER BY symbol,time"
        ).fetchdf()
        dump(frame, kind)
        del frame
    (target / "instruments/all.txt").write_text("\n".join(instruments) + "\n")
    summary.update(
        first=str(dates[0]),
        last=str(dates[-1]),
        sessions=len(dates),
        stock_price_basis="vendor frozen HFQ; factor=HFQ/raw; volume=shares/factor; amount=CNY",
        etf_price_basis="raw only; ETF distributions not reconstructed",
        historical_state_independently_verified=False,
        research_only=True,
    )
    atomic_json(target / "manifest.json", summary)
    return summary
