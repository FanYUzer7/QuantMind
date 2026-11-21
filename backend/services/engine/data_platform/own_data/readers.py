"""Explicit source readers; no network, source selection, mutation or auto-fill."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

from .contracts import DailyBatch, DailyQuery, SourceSpec, DataContractError, cn_symbol

MAX_JSON_BYTES = 25 * 1024 * 1024


def number(value):
    if value in (None, "", "-", "<null>"):
        return None
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def day(value):
    text = str(value)
    if re.fullmatch(r"\d{8}", text):
        return datetime.strptime(text, "%Y%m%d").date().isoformat()
    if len(text) > 10 and text[10] in " T" and text[11:19] not in {"00:00:00", "00:00"}:
        raise DataContractError(
            "Intraday timestamp cannot be silently truncated to a daily date"
        )
    return date.fromisoformat(text[:10]).isoformat()


def load_json(path):
    p = Path(path)
    if p.stat().st_size > MAX_JSON_BYTES:
        raise DataContractError("Saved JSON exceeds 25MiB pilot limit")
    return json.loads(p.read_text(encoding="utf-8-sig"))


def normalize(raw, spec, origin, *, collected_at=None, kind="historical_daily"):
    symbol = cn_symbol(raw["symbol"])
    values = {k: number(raw.get(k)) for k in ("open", "high", "low", "close")}
    volume, amount = number(raw.get("volume")), number(raw.get("amount"))
    flags = []
    if any(v is None for v in values.values()):
        flags.append("missing_or_invalid_ohlc")
    elif (
        min(values.values()) <= 0
        or values["high"] < max(values["open"], values["close"], values["low"])
        or values["low"] > min(values["open"], values["close"])
    ):
        flags.append("invalid_ohlc")
    if volume is None or amount is None:
        flags.append("missing_or_invalid_activity")
    if any(v is not None and v < 0 for v in (volume, amount)):
        flags.append("negative_activity")
    if spec.adjustment == "unknown":
        flags.append("unknown_adjustment")
    if spec.volume_unit == "unknown" or spec.amount_unit == "unknown":
        flags.append("unknown_units")
    if spec.evidence_only:
        flags.append("sample_evidence_only")
    if kind == "quote_snapshot":
        flags.append("snapshot_not_confirmed_daily_bar")
    amount_scale = {"yuan": 1, "thousand_yuan": 1000, "ten_thousand_yuan": 10000}.get(
        spec.amount_unit
    )
    volume_scale = {"shares": 1, "lots": 100}.get(spec.volume_unit)
    return {
        "contract_version": "0.1",
        "symbol": symbol,
        "market": "CN",
        "asset_type": spec.asset_type,
        "trade_date": day(raw["trade_date"]),
        **values,
        "volume_raw": volume,
        "amount_raw": amount,
        "volume_shares": volume * volume_scale
        if volume is not None and volume_scale
        else None,
        "amount_cny": amount * amount_scale
        if amount is not None and amount_scale
        else None,
        "volume_unit": spec.volume_unit,
        "amount_unit": spec.amount_unit,
        "currency": spec.currency,
        "price_adjustment": spec.adjustment,
        "source_id": spec.source_id,
        "dataset_version": spec.dataset_version,
        "source_record": origin,
        "collected_at": collected_at,
        "available_at": None,
        "point_in_time_verified": False,
        "data_kind": kind,
        "quality_flags": flags,
        "raw_values": raw,
    }


class _Reader:
    def __init__(self, spec):
        self.spec = spec

    def read_daily(self, query):
        if self.spec.snapshot and not query.allow_snapshots:
            raise DataContractError(
                "Quote snapshots require explicit allow_snapshots=True"
            )
        out = DailyBatch(
            self.spec.source_id,
            self.spec.dataset_version,
            evidence_only=self.spec.evidence_only,
        )
        if query.adjustment is not None and query.adjustment != self.spec.adjustment:
            return out
        seen = set()
        for raw, origin, collected_at, kind in self.records(query):
            try:
                row = normalize(
                    raw, self.spec, origin, collected_at=collected_at, kind=kind
                )
            except (KeyError, ValueError, TypeError) as e:
                out.issues.append(
                    {"code": "unparseable_record", "record": origin, "message": str(e)}
                )
                continue
            if (
                row["symbol"] not in query.symbols
                or not query.start <= row["trade_date"] <= query.end
            ):
                continue
            if len(out.rows) >= query.max_rows:
                out.truncated = True
                break
            key = (row["symbol"], row["trade_date"])
            if key in seen:
                row["quality_flags"].append("duplicate_key")
                out.issues.append(
                    {"code": "duplicate_key", "key": key, "record": origin}
                )
            seen.add(key)
            out.rows.append(row)
        out.rows.sort(key=lambda r: (r["symbol"], r["trade_date"], r["source_record"]))
        return out


class TabularDailyReader(_Reader):
    """Explicit CSV/Parquet files; mapped source columns and bounded queries.

    No recursive glob, archive extraction, or inferred schema/units at runtime.
    CSV scan cap is an error, not a false empty/full result.
    """

    def __init__(
        self,
        spec,
        files,
        columns=None,
        encoding="utf-8-sig",
        scan_limit=100000,
        symbol_map=None,
    ):
        super().__init__(spec)
        self.files = tuple(Path(p) for p in files)
        self.columns = columns or {
            k: k
            for k in (
                "symbol",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "amount",
            )
        }
        self.symbol_map = {str(k): cn_symbol(v) for k, v in (symbol_map or {}).items()}
        self.encoding = encoding
        if not self.files or len(self.files) > 100 or not 1 <= scan_limit <= 100000:
            raise DataContractError("Use 1..100 explicit files and scan_limit <=100000")
        self.scan_limit = scan_limit

    def records(self, query):
        scanned = 0
        for path in self.files:
            if path.suffix.lower() == ".csv":
                with path.open(encoding=self.encoding, newline="") as f:
                    source = csv.DictReader(f)
                    if not set(self.columns.values()).issubset(source.fieldnames or []):
                        raise DataContractError("CSV lacks configured columns")
                    for i, row in enumerate(source, 2):
                        scanned += 1
                        if scanned > self.scan_limit:
                            raise DataContractError(
                                "CSV scan cap reached; narrow files before retrying"
                            )
                        mapped = {k: row[v] for k, v in self.columns.items()}
                        mapped["symbol"] = self.symbol_map.get(
                            str(mapped["symbol"]), mapped["symbol"]
                        )
                        yield mapped, f"{path}#row={i}", None, "historical_daily"
            elif path.suffix.lower() == ".parquet":
                import duckdb

                def ident(s):
                    return '"' + s.replace('"', '""') + '"'

                projection = ",".join(
                    f"{ident(v)} AS {ident(k)}" for k, v in self.columns.items()
                )
                # Only canonical/prefix symbols; bare codes are intentionally rejected.
                aliases = set(query.symbols)
                from backend.shared.stock_utils import StockCodeUtil

                for symbol in query.symbols:
                    aliases.update(
                        [StockCodeUtil.to_prefix(symbol), StockCodeUtil.to_qlib(symbol)]
                    )
                aliases.update(
                    k for k, v in self.symbol_map.items() if v in query.symbols
                )
                slots = ",".join("?" for _ in aliases)
                dt = ident(self.columns["trade_date"])
                date_expr = f"COALESCE(TRY_CAST({dt} AS DATE),CAST(try_strptime(CAST({dt} AS VARCHAR),'%Y%m%d') AS DATE))"
                sql = (
                    f"SELECT {projection} FROM read_parquet(?) WHERE {date_expr} BETWEEN ? AND ? "
                    f"AND {ident(self.columns['symbol'])} IN ({slots}) LIMIT ?"
                )
                with duckdb.connect(":memory:") as conn:
                    cursor = conn.execute(
                        sql,
                        [
                            str(path),
                            query.start,
                            query.end,
                            *sorted(aliases),
                            query.max_rows + 1,
                        ],
                    )
                    names = [c[0] for c in cursor.description]
                    for i, values in enumerate(cursor.fetchall()):
                        mapped = dict(zip(names, values, strict=True))
                        mapped["symbol"] = self.symbol_map.get(
                            str(mapped["symbol"]), mapped["symbol"]
                        )
                        yield mapped, f"{path}#query_row={i}", None, "historical_daily"
            else:
                raise DataContractError(
                    "Only explicit CSV/Parquet inputs are supported"
                )


class QuantDBPreviewReader(_Reader):
    """Previously saved remote previews, never labelled a local full dataset."""

    def __init__(self, spec, path, dataset="daily_unadjusted"):
        if not spec.evidence_only:
            raise DataContractError("Preview reader must be evidence_only")
        super().__init__(spec)
        self.path, self.dataset = Path(path), dataset

    def records(self, query):
        for item in load_json(self.path):
            if item.get("dataset") != self.dataset or item.get("status") != 200:
                continue
            for i, row in enumerate(item.get("body", {}).get("data", [])):
                raw = {**row, "trade_date": row.get("time")}
                yield (
                    raw,
                    f"{self.path}#{self.dataset}/{i}",
                    item.get("checked_at"),
                    "historical_daily",
                )


class EastmoneySavedReader(_Reader):
    def __init__(self, spec, files, group_ids):
        super().__init__(spec)
        self.files, self.group_ids = tuple(Path(p) for p in files), frozenset(group_ids)
        if not self.files or len(self.files) > 100 or not self.group_ids:
            raise DataContractError("Explicit saved files and group IDs are required")

    def records(self, query):
        for path in self.files:
            saved = load_json(path)
            if (
                saved.get("format") != "quant-eastmoney-demo"
                or saved.get("schema_version") != 1
            ):
                raise DataContractError("Unsupported saved demo schema")
            for g in saved.get("groups", []):
                if g.get("job_id") not in self.group_ids:
                    continue
                if g.get("scope", "CN") != "CN":
                    raise DataContractError(
                        "Non-CN group is outside the first daily contract"
                    )
                if g.get("status") in {"error", "skipped", "empty"}:
                    continue
                is_snapshot = (
                    g.get("data_kind")
                    == "quote_snapshot_with_day_ohlc_and_optional_flow"
                )
                if is_snapshot != self.spec.snapshot:
                    raise DataContractError("Snapshot/history kind mismatch")
                actual_adjustment = {
                    "unadjusted": "raw",
                    "qfq": "qfq",
                    "hfq": "hfq",
                }.get(g.get("price_adjustment"), "unknown")
                if actual_adjustment != self.spec.adjustment:
                    raise DataContractError(
                        "Saved price adjustment disagrees with source declaration"
                    )
                if not is_snapshot and g.get("data_kind") != "historical_kline":
                    raise DataContractError("Group is not historical daily kline")
                # Explicit allow-list blocks minute/trend/flow groups that also have OHLC.
                if not is_snapshot and (
                    g.get("dataset") not in {None, "daily"}
                    or g.get("request_params", {}).get("klt", 101) != 101
                    or not (
                        g.get("dataset") == "daily"
                        or g.get("request_params", {}).get("klt") == 101
                    )
                ):
                    raise DataContractError("Intraday data cannot be read as daily")
                for i, record in enumerate(g.get("rows", [])):
                    r = dict(record)
                    if "raw_columns" in g and "record" in r:
                        r = dict(
                            zip(g["raw_columns"], r["record"].split(","), strict=True)
                        )
                    secid = r.get("secid") or g.get("secid", "")
                    match = re.fullmatch(r"([01])\.(\d{6})", secid)
                    if not match:
                        raise DataContractError(
                            "Saved CN group requires explicit Eastmoney market/code"
                        )
                    market, code = match.groups()
                    exchange = "SH" if market == "1" else "SZ"
                    if market == "0" and code.startswith(
                        ("43", "83", "87", "88", "92")
                    ):
                        exchange = "BJ"
                    raw = {
                        "symbol": cn_symbol(code, exchange=exchange),
                        "trade_date": r.get("source_date") or r.get("time"),
                        "open": r.get("open"),
                        "high": r.get("high"),
                        "low": r.get("low"),
                        "close": r.get("last") if is_snapshot else r.get("close"),
                        "volume": r.get("volume_raw"),
                        "amount": r.get("amount_raw"),
                    }
                    yield (
                        raw,
                        f"{path}#{g['job_id']}/{i}",
                        g.get("collected_at") or saved.get("saved_at"),
                        "quote_snapshot" if is_snapshot else "historical_daily",
                    )


class OwnDataRepository:
    def __init__(self, readers):
        self.readers = {}
        for reader in readers:
            if reader.spec.source_id in self.readers:
                raise DataContractError("Repeated source ID")
            self.readers[reader.spec.source_id] = reader

    def list_sources(self):
        return [asdict(r.spec) for r in self.readers.values()]

    def read_daily(self, source_id, query):
        return self.readers[source_id].read_daily(query)


def compare_daily(left, right):
    """Report overlap only; never choose a winner or collapse duplicate records."""

    def group(rows):
        out = {}
        for row in rows:
            out.setdefault((row["symbol"], row["trade_date"]), []).append(row)
        return out

    a, b = group(left.rows), group(right.rows)
    pairs = []
    for key in sorted(a.keys() & b.keys()):
        x, y = a[key], b[key]
        reasons = []
        if len(x) != 1 or len(y) != 1:
            reasons.append("duplicate_key")
        if (
            x[0]["price_adjustment"] != y[0]["price_adjustment"]
            or x[0]["price_adjustment"] == "unknown"
        ):
            reasons.append("adjustment_mismatch_or_unknown")
        if x[0]["data_kind"] != y[0]["data_kind"]:
            reasons.append("snapshot_vs_daily")
        if x[0]["asset_type"] != y[0]["asset_type"]:
            reasons.append("asset_type_mismatch")
        critical = {"invalid_ohlc", "missing_or_invalid_ohlc", "negative_activity"}
        if critical.intersection(x[0]["quality_flags"] + y[0]["quality_flags"]):
            reasons.append("invalid_values")
        deltas = (
            {}
            if reasons
            else {
                k: y[0][k] - x[0][k]
                if x[0][k] is not None and y[0][k] is not None
                else None
                for k in ("open", "high", "low", "close", "volume_shares", "amount_cny")
            }
        )
        pairs.append(
            {
                "symbol": key[0],
                "trade_date": key[1],
                "comparable_prices": not reasons,
                "reasons": reasons,
                "right_minus_left": deltas,
            }
        )
    return {
        "left": left.source_id,
        "right": right.source_id,
        "overlap_keys": len(pairs),
        "left_only": len(a.keys() - b.keys()),
        "right_only": len(b.keys() - a.keys()),
        "pairs": pairs,
        "selection_performed": False,
        "conclusion": "inspect_differences" if pairs else "insufficient_overlap",
        "coverage_scope": "bounded query results only",
    }
