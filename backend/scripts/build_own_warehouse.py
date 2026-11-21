"""Ingest owned market data without executing any code shipped with datasets.

Run in the lab image using scripts/quant-data-exec. Re-running resumes verified
objects; source revisions create new content-addressed objects and are retained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from backend.shared.stock_utils import StockCodeUtil
from backend.services.engine.data_platform.own_data.warehouse import (
    export_dictionary,
    identifier as qi,
    literal as qs,
    sha256,
)
from backend.services.engine.data_platform.own_data.field_semantics import (
    description as source_description,
)


NOTES = {
    "symbol": "证券后缀码；股票命名空间经 StockCodeUtil 转换，不对指数使用股票推断",
    "trade_date": "源数据交易日期；不是采集日期",
    "open": "开盘价，价格口径见 price_adjustment",
    "high": "最高价，价格口径见 price_adjustment",
    "low": "最低价，价格口径见 price_adjustment",
    "close": "收盘价，价格口径见 price_adjustment",
    "volume": "成交量（股）；仅对具有股单位证据的日线转换",
    "amount": "成交额（人民币元）；仅对具有元单位证据的日线转换",
    "price_adjustment": "raw 不复权 / qfq 前复权 / hfq 后复权；不同版本不混合",
    "source_id": "明确来源与版本标识；多来源不自动覆盖",
    "quality_flags": "异常标记，空字符串表示已实现检查未发现问题，不代表真实性认证",
    "is_st": "源 是否ST 映射 是=1 否=0；未知为 NULL，未经独立历史核验",
    "float_shares": "来源流通股本（股）",
    "_qm_source_file": "来源文件名或 ZIP 成员名；完整原始包见 catalog_files",
    "_qm_collected_at": "源采集时间 UTC 字符串；不能代替数据所属日期",
    "_qm_saved_at": "JSON 保存时间 UTC 字符串",
    "_qm_available_at": "保守可用时间：本次采集时间，不能据此声称过去可见",
    "_qm_group_status": "采集器分组状态；complete 只表示采集状态",
    "_qm_is_eod_confirmed": "源采集器是否确认收盘，不据采集时间推断",
    "_qm_source_date": "源记录自带日期，缺失不以采集日期补齐",
    "_qm_metadata": "原采集分组元信息 JSON，包含单位、分页、覆盖与错误",
}


class Builder:
    def __init__(self, root, source, eastmoney):
        self.root, self.source, self.eastmoney = root.resolve(), source, eastmoney
        for part in ("objects", "parquet"):
            (self.root / part).mkdir(parents=True, exist_ok=True)
        management = self.root / "_ops" if (self.root / "STORE.json").is_file() else self.root
        for part in ("staging", "runs"):
            (management / part).mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.root / "warehouse.duckdb"))
        self.con.execute("SET threads=4")
        self.con.execute("SET memory_limit='8GB'")
        self.con.execute("SET preserve_insertion_order=false")
        self.con.execute("SET temp_directory=" + qs(management / "staging"))
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS catalog_files (file_id VARCHAR PRIMARY KEY, source_path VARCHAR, sha256 VARCHAR, bytes BIGINT, object_path VARCHAR, imported_at TIMESTAMPTZ)"
        )
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS catalog_tables (table_name VARCHAR PRIMARY KEY, source_family VARCHAR, files_json VARCHAR, row_count BIGINT, notes VARCHAR)"
        )
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS catalog_fields (table_name VARCHAR, ordinal INTEGER, column_name VARCHAR, data_type VARCHAR, description VARCHAR, PRIMARY KEY(table_name,column_name))"
        )
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS ingest_events (event_at TIMESTAMPTZ, source_path VARCHAR, status VARCHAR, detail VARCHAR)"
        )
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS catalog_members (archive_sha256 VARCHAR, member_name VARCHAR, crc32 BIGINT, bytes BIGINT, parquet_path VARCHAR, PRIMARY KEY(archive_sha256,member_name))"
        )
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS catalog_sources (source_id VARCHAR PRIMARY KEY, original_root VARCHAR, read_mount VARCHAR)"
        )
        self.con.execute(
            "INSERT OR REPLACE INTO catalog_sources VALUES ('offline','/mnt/DockerStor/stock','/sources/stock'),('eastmoney','/mnt/c/Users/Admin/Desktop/stock-daily','/sources/eastmoney')"
        )
        self.docs = {}
        dictionary = (
            self.source
            / "无名游资256维度Agent量化回测数据包/【必下载】0_字段字典与样例_0.3G/数据字典.csv"
        )
        if dictionary.exists():
            stored_dictionary = self.object(dictionary)
            with stored_dictionary.open(encoding="utf-8-sig") as f:
                for row in csv.DictReader(f):
                    self.docs[row["col"]] = json.dumps(row, ensure_ascii=False)

    def event(self, path, status, detail):
        self.con.execute(
            "INSERT INTO ingest_events VALUES (now(),?,?,?)",
            [str(path), status, str(detail)],
        )
        print(status, str(path), str(detail)[:150], flush=True)

    def object(self, path):
        before = path.stat()
        known = self.con.execute(
            "SELECT object_path,sha256 FROM catalog_files WHERE source_path=? AND bytes=?",
            [str(path), before.st_size],
        ).fetchone()
        if known and sha256(path) == known[1]:
            after = path.stat()
            if (
                before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
            ):
                raise ValueError("Source changed while verifying")
            target = Path(known[0])
            digest = known[1]
            if not target.is_file() or sha256(target) != digest:
                raise ValueError("Stored object checksum mismatch")
            return target
        # One source read for new files; verification reads only the local copy.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        part = self.root / "objects" / ("copy-" + stamp + ".part")
        h = hashlib.sha256()
        with path.open("rb") as src, part.open("xb") as dst:
            for block in iter(lambda: src.read(4 * 1024 * 1024), b""):
                h.update(block)
                dst.write(block)
        after = path.stat()
        if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
            raise ValueError("Source changed while copying")
        digest = h.hexdigest()
        if sha256(part) != digest:
            raise ValueError("Copied object checksum mismatch")
        target = self.root / "objects" / (digest + path.suffix.lower())
        if target.exists():
            if sha256(target) != digest:
                raise ValueError("Stored object checksum mismatch")
            part.unlink()
        else:
            part.rename(target)
        self.con.execute(
            "INSERT OR IGNORE INTO catalog_files VALUES (?,?,?,?,?,now())",
            [digest, str(path), digest, path.stat().st_size, str(target)],
        )
        return target

    def publish(self, name, paths, family, notes):
        if not paths:
            return
        paths = [str(p) for p in paths]
        relation = (
            "read_parquet([" + ",".join(qs(p) for p in paths) + "],union_by_name=true)"
        )
        self.con.execute(
            f"CREATE OR REPLACE VIEW {qi(name)} AS SELECT * FROM {relation}"
        )
        count = self.con.execute(f"SELECT count(*) FROM {qi(name)}").fetchone()[0]
        self.con.execute(
            "INSERT OR REPLACE INTO catalog_tables VALUES (?,?,?,?,?)",
            [name, family, json.dumps(paths), count, notes],
        )
        self.con.execute("DELETE FROM catalog_fields WHERE table_name=?", [name])
        schema = self.con.execute(f"DESCRIBE {qi(name)}").fetchall()
        for ordinal, col in enumerate(schema, 1):
            key, dtype = col[:2]
            if name.startswith("offline_daily"):
                description = NOTES.get(key)
            elif key.startswith("_qm_"):
                description = NOTES.get(key)
            elif name.startswith("wuming"):
                description = self.docs.get(key)
            else:
                description = None
            description = description or source_description(key)
            if name.startswith("quantdb_") and key.lower() == "amount":
                description = "QuantDB原始成交额；文档口径为万元，规范日线另转换为元；保留原始值供逐文件核验"
            if not description:
                description = (
                    "源字段："
                    + key
                    + "；保留源口径与单位；未单独核验，不自动用作训练特征"
                )
            self.con.execute(
                "INSERT INTO catalog_fields VALUES (?,?,?,?,?)",
                [name, ordinal, key, dtype, description],
            )
        self.event(name, "published", f"{count} rows / {len(schema)} fields")

    def parquet_copy(self, path, name, family, notes):
        obj = self.object(path)
        self.publish(name, [obj], family, notes)

    def daily(self):
        base = self.source / "A股分钟线+日频量价"
        datasets = []
        for adjustment, chinese in [
            ("raw", "不复权"),
            ("qfq", "前复权"),
            ("hfq", "后复权"),
        ]:
            archive = base / "股票更新至4.30/每只股票一个文件" / (chinese + ".zip")
            obj = self.object(archive)
            outputs = []
            for kind in ("archive", "updates"):
                paths = (
                    sorted((base / "每天一个文件" / chinese).rglob("*.csv"))
                    if kind == "updates"
                    else []
                )
                signature = (
                    sha256(obj)
                    if kind == "archive"
                    else hashlib.sha256(
                        "".join(str(p) + sha256(p) for p in paths).encode()
                    ).hexdigest()
                )
                out = (
                    self.root
                    / "parquet"
                    / f"daily_{adjustment}_{kind}_{signature[:16]}.parquet"
                )
                source_id = f"offline_20260430_{adjustment}_{kind}_{signature[:12]}"
                if not out.exists():
                    with tempfile.TemporaryDirectory(dir=self.root / "staging") as temp:
                        tmp = Path(temp)
                        if kind == "archive":
                            with zipfile.ZipFile(obj) as z:
                                members = [
                                    m
                                    for m in z.infolist()
                                    if m.filename.lower().endswith(".csv")
                                    and not m.filename.startswith("__MACOSX/")
                                ]
                                for i, member in enumerate(members):
                                    with (
                                        z.open(member) as src,
                                        (
                                            tmp
                                            / f"{i:06d}_{Path(member.filename).name}"
                                        ).open("wb") as dst,
                                    ):
                                        shutil.copyfileobj(src, dst)
                                (
                                    self.root / "runs" / f"{signature}-members.json"
                                ).write_text(
                                    json.dumps(
                                        [
                                            {
                                                "name": m.filename,
                                                "bytes": m.file_size,
                                                "crc32": m.CRC,
                                            }
                                            for m in members
                                        ],
                                        ensure_ascii=False,
                                    )
                                )
                        else:
                            for p in paths:
                                stored = self.object(p)
                                shutil.copyfile(stored, tmp / p.name)
                        if not list(tmp.glob("*.csv")):
                            continue
                        rel = f"read_csv({qs(tmp / '*.csv')},header=true,all_varchar=true,union_by_name=true,filename=true,nullstr=['','-','<null>','null','None'],strict_mode=true)"
                        codes = self.con.execute(
                            f'SELECT DISTINCT trim("代码") FROM {rel}'
                        ).fetchall()
                        self.con.execute(
                            "CREATE OR REPLACE TEMP TABLE symbol_mapping(code VARCHAR, symbol VARCHAR)"
                        )
                        mapping = [
                            (code, StockCodeUtil.normalize(code))
                            for (code,) in codes
                            if code
                        ]
                        self.con.executemany(
                            "INSERT INTO symbol_mapping VALUES (?,?)", mapping
                        )
                        columns = ",".join(
                            f"try_cast({qi(c)} AS DOUBLE) AS {qi(n)}"
                            for c, n in [
                                ("开盘价", "open"),
                                ("最高价", "high"),
                                ("最低价", "low"),
                                ("收盘价", "close"),
                                ("成交量（股）", "volume"),
                                ("成交额（元）", "amount"),
                                ("流通股本（股）", "float_shares"),
                            ]
                        )
                        sql = f"""SELECT s.* EXCLUDE(filename), regexp_replace(filename, '^.*/', '') AS _qm_source_file,
                            m.symbol, try_cast("日期" AS DATE) AS trade_date, {columns},
                            CASE "是否ST" WHEN '是' THEN 1 WHEN '否' THEN 0 END::TINYINT AS is_st,
                            {qs(adjustment)} AS price_adjustment, {qs(source_id)} AS source_id
                            FROM {rel} s LEFT JOIN symbol_mapping m ON trim(s."代码")=m.code"""
                        sql = f"""SELECT *, concat_ws(';',
                          CASE WHEN NOT regexp_full_match(coalesce(symbol,''),'[0-9]{{6}}\\.(SH|SZ|BJ)') THEN 'invalid_symbol' END,
                          CASE WHEN trade_date IS NULL THEN 'invalid_date' END,
                          CASE WHEN open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR NOT isfinite(open) OR NOT isfinite(high) OR NOT isfinite(low) OR NOT isfinite(close) OR least(open,high,low,close)<=0 OR high<greatest(open,low,close) OR low>least(open,close) THEN 'invalid_ohlc' END,
                          CASE WHEN volume IS NULL OR amount IS NULL OR volume<0 OR amount<0 OR NOT isfinite(volume) OR NOT isfinite(amount) THEN 'invalid_activity' END,
                          CASE WHEN is_st IS NULL THEN 'unknown_st' END) AS quality_flags FROM ({sql})"""
                        part = out.with_suffix(".part.parquet")
                        self.con.execute(
                            f"COPY ({sql}) TO {qs(part)} (FORMAT PARQUET, COMPRESSION ZSTD)"
                        )
                        part.rename(out)
                outputs.append(out)
            name = f"offline_daily_{adjustment}"
            self.publish(
                name,
                outputs,
                "offline_daily",
                "原始中文列完整保留；规范字段额外添加；重叠版本保留，在 daily_bars 中选择更新包；单位来自表头股/元。",
            )
            datasets.append(name)
        # Repeated dates within an input remain visible as conflicts, not silently discarded.
        union = " UNION ALL BY NAME ".join(f"SELECT * FROM {qi(t)}" for t in datasets)
        self.con.execute(f"CREATE OR REPLACE VIEW daily_all_versions AS {union}")
        self.con.execute("""CREATE OR REPLACE VIEW daily_bars AS
            SELECT * EXCLUDE(_priority) FROM (
              SELECT *, max(CASE WHEN contains(source_id,'_updates_') THEN 1 ELSE 0 END)
                OVER (PARTITION BY symbol,trade_date,price_adjustment) AS _priority
              FROM daily_all_versions
            ) WHERE CASE WHEN contains(source_id,'_updates_') THEN 1 ELSE 0 END=_priority""")
        self.con.execute("""CREATE OR REPLACE VIEW daily_conflicts AS SELECT symbol,trade_date,price_adjustment,count(*) AS copies
            FROM daily_bars GROUP BY ALL HAVING count(*)>1""")

    def eastmoney_import(self, incremental=False):
        groups = {}
        changed = set()
        known = set()
        if incremental:
            known = {r[0] for r in self.con.execute("SELECT sha256 FROM catalog_files").fetchall()}
            for name, paths in self.con.execute("SELECT table_name,files_json FROM catalog_tables WHERE source_family='eastmoney'").fetchall():
                groups[name] = [Path(p) for p in json.loads(paths)]
        for path in sorted(self.eastmoney.rglob("*.json")):
            if path.is_symlink():
                continue
            if incremental and sha256(path) in known:
                continue
            obj = self.object(path)
            x = json.loads(obj.read_text(encoding="utf-8-sig"))
            if x.get("format") != "quant-eastmoney-demo":
                self.event(
                    path, "unsupported", "Unrecognized JSON format; raw object retained"
                )
                continue
            for g in x.get("groups", []):
                name = "eastmoney_" + re.sub("[^a-zA-Z0-9_]", "_", g["job_id"])
                changed.add(name)
                out = self.root / "parquet" / f"{name}_{obj.stem}.parquet"
                rows = []
                for r in g.get("rows", []):
                    r = dict(r)
                    r.update(
                        _qm_collected_at=g.get("collected_at"),
                        _qm_saved_at=x.get("saved_at"),
                        _qm_available_at=g.get("collected_at") or x.get("saved_at"),
                        _qm_source_file=str(path.relative_to(self.eastmoney)),
                        _qm_group_status=g.get("status"),
                        _qm_is_eod_confirmed=g.get("is_eod_confirmed", False),
                        _qm_source_date=r.get("source_date"),
                        _qm_metadata=json.dumps(
                            {k: v for k, v in g.items() if k != "rows"},
                            ensure_ascii=False,
                        ),
                    )
                    # Nested values remain lossless JSON rather than unstable Arrow structs.
                    rows.append(
                        {
                            k: json.dumps(v, ensure_ascii=False)
                            if isinstance(v, dict | list)
                            else v
                            for k, v in r.items()
                        }
                    )
                if not rows:
                    self.event(
                        path,
                        "empty_group",
                        g["job_id"]
                        + ": "
                        + str(g.get("error") or g.get("reason") or g.get("status")),
                    )
                    continue
                if not out.exists():
                    names = list(dict.fromkeys(k for r in rows for k in r))
                    arrays = []
                    for key in names:
                        vals = [r.get(key) for r in rows]
                        nonnull = [v for v in vals if v is not None]
                        if any(isinstance(v, str) for v in nonnull):
                            vals = [str(v) if v is not None else None for v in vals]
                        arrays.append(pa.array(vals))
                    part = out.with_suffix(".part.parquet")
                    pq.write_table(
                        pa.Table.from_arrays(arrays, names=names),
                        part,
                        compression="zstd",
                    )
                    part.rename(out)
                if out not in groups.setdefault(name, []):
                    groups[name].append(out)
        for name, paths in groups.items():
            if incremental and name not in changed:
                continue
            self.publish(
                name,
                paths,
                "eastmoney",
                "源日期与采集时间分开；各个对象、资金流、宏观、事件分表；不作为已确认历史日线。原JSON完整归档。",
            )

    def rich(self):
        base = self.source / "无名游资256维度Agent量化回测数据包"
        wide = [
            self.object(p)
            for p in sorted(
                (base / "【必下载】2_宽表256维度Agent用_2.5G").glob("*.parquet")
            )
        ]
        self.publish(
            "wuming_wide",
            wide,
            "wuming",
            "含未来收益、估算成员和待核验财务时点；研究原表，不自动作为模型特征。",
        )
        for p in sorted(
            (base / "【必下载】5_明细数据_研报新闻席位报表_37G").glob("*.csv")
        ):
            name = "wuming_" + p.stem
            obj = self.object(p)
            out = self.root / "parquet" / f"{name}_{obj.stem}.parquet"
            if not out.exists():
                part = out.with_suffix(
                    ".part-"
                    + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
                    + ".parquet"
                )
                # Codes retain leading zeroes and text fields retain their original values.
                self.con.execute(
                    f"COPY (SELECT * FROM read_csv({qs(obj)},header=true,delim=',',quote='\"',escape='\"',all_varchar=true,strict_mode=true,max_line_size=67108864)) TO {qs(part)} (FORMAT PARQUET,COMPRESSION ZSTD)"
                )
                part.rename(out)
            self.publish(
                name,
                [out],
                "wuming_detail",
                "明细原字段无损保留为字符串，避免代码/日期自动推断失真；训练需经公告时点与字段口径检查。",
            )
        for pattern in ["产业链*.zip", "监管*.zip", "纳指*.zip", "无名游资扩展*.zip"]:
            for p in sorted(self.source.glob(pattern)):
                obj = self.object(p)
                with zipfile.ZipFile(obj) as z:
                    for m in z.infolist():
                        if m.filename.lower().endswith(".parquet"):
                            tag = hashlib.sha256(m.filename.encode()).hexdigest()[:8]
                            name = "special_" + Path(m.filename).stem + "_" + tag
                            out = (
                                self.root / "parquet" / f"{obj.stem[:16]}_{tag}.parquet"
                            )
                            if not out.exists():
                                part = out.with_suffix(".part.parquet")
                                with z.open(m) as src, part.open("wb") as dst:
                                    shutil.copyfileobj(src, dst)
                                pq.ParquetFile(part)  # validate before publication
                                part.rename(out)
                            self.publish(
                                name,
                                [out],
                                "special",
                                f"原始包 {p.name}；成员 {m.filename}；LASSO成员为估算，席位画像可能使用全样本未来收益；海外价格另有口径。",
                            )

    def quantdb(self, plan_path):
        base = self.root / "quantdb"
        if plan_path is None:
            raise ValueError("quantdb phase requires --quantdb-plan")
        plan = json.loads(plan_path.read_text())
        groups = {}
        compat = self.root / "quantdb-compatible"
        for dataset, _category, obj in plan["entries"]:
            tag = hashlib.sha256(obj["key"].encode()).hexdigest()
            p = base / dataset / (tag + ".parquet")
            if not p.is_file() or p.stat().st_size != obj["size"]:
                raise ValueError("Incomplete QuantDB download: " + dataset)
            rel = Path(obj.get("relative_path") or obj["key"])
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Unsafe manifest path")
            target = compat / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(p, target)
            elif sha256(target) != sha256(p):
                raise ValueError("Compatibility path conflict")
            groups.setdefault(dataset, []).append(p)
        shutil.copyfile(
            plan_path,
            self.root / "runs" / ("quantdb-plan-" + sha256(plan_path)[:16] + ".json"),
        )
        for dataset, paths in groups.items():
            self.publish(
                "quantdb_" + dataset,
                paths,
                "quantdb",
                "原始QuantDB字段与单位；财务需按实际公告时点连接；已核对全量清单文件数与大小。",
            )
        if {"daily_unadjusted", "daily_forward", "daily_backward"} <= groups.keys():
            # Audit the documented shares / 10k CNY convention per partition.
            self.con.execute("""CREATE OR REPLACE TABLE quantdb_unit_audit AS
                SELECT try_cast(time AS DATE) AS trade_date,
                  median(close*volume/nullif(amount,0)) FILTER (WHERE close>0 AND volume>0 AND amount>0) AS median_price_volume_amount_ratio
                FROM quantdb_daily_unadjusted GROUP BY 1""")
            parts = []
            for dataset, adjustment in [
                ("daily_unadjusted", "raw"),
                ("daily_forward", "qfq"),
                ("daily_backward", "hfq"),
            ]:
                parts.append(f"""SELECT symbol,try_cast(time AS DATE) AS trade_date,
                    open,high,low,close,volume,amount*10000 AS amount,
                    amount AS amount_source, 'ten_thousand_yuan' AS amount_source_unit,
                    NULL::TINYINT AS is_st,NULL::DOUBLE AS float_shares,
                    {qs(adjustment)} AS price_adjustment, {qs("quantdb_" + dataset)} AS source_id,
                    concat_ws(';','unknown_st',
                      CASE WHEN try_cast(time AS DATE) IS NULL THEN 'invalid_date' END,
                      CASE WHEN open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR NOT isfinite(open) OR NOT isfinite(high) OR NOT isfinite(low) OR NOT isfinite(close) OR least(open,high,low,close)<=0 OR high<greatest(open,low,close) OR low>least(open,close) THEN 'invalid_ohlc' END,
                      CASE WHEN volume IS NULL OR amount IS NULL OR volume<0 OR amount<0 OR NOT isfinite(volume) OR NOT isfinite(amount) THEN 'invalid_activity' END,
                      CASE WHEN u.median_price_volume_amount_ratio IS NULL OR u.median_price_volume_amount_ratio NOT BETWEEN 1000 AND 100000 THEN 'unit_mismatch' END) AS quality_flags
                    FROM quantdb_{dataset} d LEFT JOIN quantdb_unit_audit u ON try_cast(d.time AS DATE)=u.trade_date""")
            self.con.execute(
                "CREATE OR REPLACE VIEW quantdb_daily_bars AS "
                + " UNION ALL BY NAME ".join(parts)
            )

    def minute(self):
        """Keep actual intraday timestamps; do not infer units/adjustment from size."""
        base = self.source / "A股分钟线+日频量价/A股分钟数据"
        for path in sorted((base / "5m_price_zip").glob("*.zip")):
            obj = self.object(path)
            paths = []
            folder = (
                self.root / "parquet" / ("minute5_" + path.stem + "_" + obj.stem[:12])
            )
            folder.mkdir(exist_ok=True)
            with zipfile.ZipFile(obj) as z:
                for m in z.infolist():
                    if not m.filename.endswith(".parquet"):
                        continue
                    tag = hashlib.sha256(m.filename.encode()).hexdigest()
                    out = folder / (tag + ".parquet")
                    if not out.exists():
                        part = out.with_suffix(".part.parquet")
                        with z.open(m) as src, part.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
                        pq.ParquetFile(part)
                        part.rename(out)
                    self.con.execute(
                        "INSERT OR REPLACE INTO catalog_members VALUES (?,?,?,?,?)",
                        [obj.stem, m.filename, m.CRC, m.file_size, str(out)],
                    )
                    paths.append(out)
            self.publish(
                "minute5_" + path.stem,
                paths,
                "offline_minute5",
                "5分钟源Parquet，保留trade_time原始完整时间，Asia/Shanghai语境；vol/amount单位、价格复权待独立核验，未并入规范日线。",
            )

    def finish(self):
        # Also inventory helper/canonical views and metadata tables in the field dictionary.
        existing = {
            r[0]
            for r in self.con.execute(
                "SELECT table_name FROM catalog_tables"
            ).fetchall()
        }
        for (table,) in self.con.execute("SHOW TABLES").fetchall():
            if table in existing or table == "symbol_mapping":
                continue
            for i, row in enumerate(
                self.con.execute(f"DESCRIBE {qi(table)}").fetchall(), 1
            ):
                self.con.execute(
                    "INSERT OR REPLACE INTO catalog_fields VALUES (?,?,?,?,?)",
                    [
                        table,
                        i,
                        row[0],
                        row[1],
                        NOTES.get(row[0], "目录/检查字段：" + row[0]),
                    ],
                )
            if table not in {"catalog_fields", "catalog_tables"}:
                self.con.execute(
                    "INSERT OR IGNORE INTO catalog_tables VALUES (?,?,?,NULL,?)",
                    [
                        table,
                        "catalog_or_canonical",
                        "[]",
                        "辅助目录/规范视图；行数由最终全库验证填写",
                    ],
                )
        # Refresh source explanations even for tables imported by previous phases.
        updates = []
        for table, key in self.con.execute(
            "SELECT table_name,column_name FROM catalog_fields"
        ).fetchall():
            desc = None
            if table.startswith("offline_daily") or table in {
                "daily_bars",
                "daily_all_versions",
                "quantdb_daily_bars",
            }:
                desc = NOTES.get(key)
            elif key.startswith("_qm_"):
                desc = NOTES.get(key)
            elif table.startswith("wuming"):
                desc = self.docs.get(key)
            desc = desc or source_description(key)
            if (
                table.startswith("quantdb_")
                and table != "quantdb_daily_bars"
                and key.lower() == "amount"
            ):
                desc = "QuantDB原始成交额，文档单位为万元；规范日线另转换元，保留原值供逐文件审计"
            if desc:
                updates.append((desc, table, key))
        if updates:
            self.con.execute("BEGIN TRANSACTION")
            self.con.executemany(
                "UPDATE catalog_fields SET description=? WHERE table_name=? AND column_name=?",
                updates,
            )
            self.con.execute("COMMIT")
        export_dictionary(self.con, self.root / "FIELD-DICTIONARY.md")
        self.con.execute("CHECKPOINT")
        self.con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--source", type=Path, default=Path("/sources/stock"))
    ap.add_argument("--eastmoney", type=Path, default=Path("/sources/eastmoney"))
    ap.add_argument(
        "--phase",
        choices=["daily", "eastmoney", "rich", "minute", "quantdb", "dictionary"],
        required=True,
    )
    ap.add_argument("--quantdb-plan", type=Path)
    a = ap.parse_args()
    b = Builder(a.root, a.source, a.eastmoney)
    try:
        if a.phase == "daily":
            b.daily()
        elif a.phase == "eastmoney":
            b.eastmoney_import()
        elif a.phase == "rich":
            b.rich()
        elif a.phase == "minute":
            b.minute()
        elif a.phase == "quantdb":
            b.quantdb(a.quantdb_plan)
        b.finish()
    except Exception as exc:
        b.event(a.phase, "failed", type(exc).__name__ + ": " + str(exc))
        b.con.close()
        raise


if __name__ == "__main__":
    main()
