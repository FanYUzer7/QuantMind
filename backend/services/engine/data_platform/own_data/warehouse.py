"""Local Parquet warehouse. Explicit sources, immutable objects, no network reads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb

from .contracts import cn_symbol


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def identifier(value):
    return '"' + str(value).replace('"', '""') + '"'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Warehouse:
    def __init__(self, root, *, read_only=True):
        self.root = Path(root).resolve()
        self._access = None
        if (self.root / "STORE.json").is_file():
            from .current import access

            self._access = access(self.root, write=not read_only)
            self._access.__enter__()
        from .publication import current_release

        self.release = current_release(self.root)
        try:
            self.connection = duckdb.connect(
                str(self.release["database"]), read_only=read_only
            )
        except BaseException:
            if self._access:
                self._access.__exit__(None, None, None)
            raise

    def close(self):
        try:
            self.connection.close()
        finally:
            if self._access:
                self._access.__exit__(None, None, None)
                self._access = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def daily(self, symbols, start, end, *, adjustment="raw", source=None):
        if not symbols or start > end or adjustment not in {"raw", "qfq", "hfq"}:
            raise ValueError("Explicit symbols, date range and adjustment required")
        symbols = [cn_symbol(s) for s in symbols]
        integrated = source == "integrated" or (
            source is None and self.release["release_id"] != "initial"
        )
        table = (
            "research_daily_bars"
            if integrated
            else "quantdb_daily_bars"
            if source == "quantdb"
            else "daily_bars"
        )
        sql = (
            f"SELECT * FROM {table} WHERE symbol IN ("
            + ",".join("?" for _ in symbols)
            + ") AND trade_date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE) "
            "AND price_adjustment=?"
        )
        params = [*symbols, start, end, adjustment]
        if source and source not in {"offline", "quantdb", "integrated"}:
            sql += " AND source_id=?"
            params.append(source)
        frame = self.connection.execute(sql, params).fetchdf()
        if frame.duplicated(["symbol", "trade_date"]).any():
            raise ValueError("Conflicting sources/versions: select an explicit source")
        return frame.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    def factors(self, symbols, start, end, fields, *, source="l1_l2_factors"):
        if source not in {"l1_factors", "l1_l2_factors", "features_daily"}:
            raise ValueError("Unknown factor source")
        table = "quantdb_" + source
        columns = {
            r[0]
            for r in self.connection.execute("DESCRIBE " + identifier(table)).fetchall()
        }
        if not fields or any(f not in columns for f in fields):
            raise ValueError("Explicit available factor fields are required")
        import re

        if any(
            re.search(r"(^|_)(label|future|fwd|next)(_|$)|^return_\d+d$", f, re.I)
            for f in fields
        ):
            raise ValueError(
                "Forward labels/next-day states cannot be admitted as research inputs"
            )
        key = next((k for k in ("date", "time", "trade_date") if k in columns), None)
        if key is None:
            raise ValueError("Factor date column unavailable")
        values = [cn_symbol(s) for s in symbols]
        if not values:
            raise ValueError("Specify symbols for a bounded factor query")
        sql = (
            f"SELECT symbol,try_cast({identifier(key)} AS DATE) AS trade_date,"
            + ",".join(identifier(f) for f in fields)
        )
        sql += (
            f" FROM {identifier(table)} WHERE symbol IN ("
            + ",".join("?" for _ in values)
            + f") AND try_cast({identifier(key)} AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE) ORDER BY symbol,trade_date"
        )
        frame = self.connection.execute(sql, [*values, start, end]).fetchdf()
        if frame.duplicated(["symbol", "trade_date"]).any():
            raise ValueError("Ambiguous factor revisions")
        return frame

    def tables(self):
        return self.connection.execute(
            "SELECT * FROM catalog_tables ORDER BY table_name"
        ).fetchdf()


def export_dictionary(connection, output):
    """Document the actual published schema, including unverified source fields."""
    rows = connection.execute(
        "SELECT table_name,column_name,data_type,description FROM catalog_fields "
        "ORDER BY table_name,ordinal"
    ).fetchall()
    lines = [
        "# 实际数据库字段字典",
        "",
        "由已发布数据库生成；未知源字段含义明确保留为待核验。",
        "",
    ]
    tables = connection.execute(
        "SELECT table_name,source_family,row_count,notes FROM catalog_tables ORDER BY table_name"
    ).fetchall()
    lines += [
        "## 表目录",
        "",
        "| 表 | 来源家族 | 行数 | 说明 |",
        "|---|---|---:|---|",
    ]
    for name, family, count, notes in tables:
        clean = str(notes).replace("|", " / ").replace("\n", " ")
        lines.append(
            f"| {name} | {family} | {count if count is not None else '待统计'} | {clean} |"
        )
    lines += ["", "## 字段明细", ""]
    current = None
    for table, column, dtype, description in rows:
        if current != table:
            lines += [f"## {table}", "", "| 字段 | 类型 | 含义 |", "|---|---|---|"]
            current = table
        clean = str(description).replace("|", " / ").replace("\n", " ")
        if clean.startswith('{"col":'):
            try:
                definition = json.loads(description)
                clean = str(definition.get("cn") or column)
                if definition.get("note"):
                    clean += "；" + str(definition["note"])
                clean += "（来源字典定义；实际存储类型见类型列）"
                clean = clean.replace("|", " / ").replace("\n", " ")
            except (ValueError, TypeError):
                pass
        lines.append(f"| {column} | {dtype} | {clean} |")
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")
