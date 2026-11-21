"""Build an isolated catalog, validate, then atomically publish one release pointer."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import duckdb

from backend.shared.stock_utils import StockCodeUtil
from .sync import atomic_json, safe_relative
from .warehouse import identifier as qi, literal as qs, sha256


def current_release(root):
    root = Path(root).resolve()
    if (root / "STORE.json").is_file():
        return {
            "database": root / "warehouse.duckdb",
            "compatible": root,
            "release_id": "current",
        }
    pointer = root / "CURRENT.json"
    if not pointer.exists():
        return {
            "database": root / "warehouse.duckdb",
            "compatible": root / "quantdb-compatible",
            "release_id": "initial",
        }
    value = json.loads(pointer.read_text())
    result = {"release_id": value["release_id"]}
    for key in ("database", "compatible"):
        path = (root / value[key]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Release pointer escaped warehouse")
        result[key] = path
    if value.get("qlib"):
        path = (root / value["qlib"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Qlib pointer escaped warehouse")
        result["qlib"] = path
    return result


def symlink_replace(link, target):
    link = Path(link)
    temporary = link.with_name(link.name + ".next")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target, target_is_directory=Path(target).is_dir())
    os.replace(temporary, link)


def clone_compatible(source, target):
    if source.is_dir():
        shutil.copytree(source, target, copy_function=os.link)
    else:
        target.mkdir()


def materialize_state(builder, root):
    con = builder.con
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    risk = "special_03_风险警示_逐日状态_71a19eb0"
    if risk not in tables:
        con.execute(
            "CREATE OR REPLACE VIEW security_state_candidates AS SELECT NULL::VARCHAR AS symbol,NULL::DATE AS trade_date,NULL::TINYINT AS is_st,NULL::VARCHAR AS confidence,NULL::VARCHAR AS source WHERE false"
        )
        return
    info = con.execute(
        "SELECT files_json FROM catalog_tables WHERE table_name=?", [risk]
    ).fetchone()
    source_paths = json.loads(info[0])
    signature = hashlib.sha256(
        json.dumps([(p, sha256(p)) for p in source_paths]).encode()
    ).hexdigest()
    folder = root / "derived/security_state"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (signature + ".parquet")
    if not target.exists():
        codes = [
            r[0]
            for r in con.execute(f"SELECT DISTINCT code FROM {qi(risk)}").fetchall()
        ]
        con.execute("CREATE TEMP TABLE state_symbol_map (code VARCHAR,symbol VARCHAR)")
        con.executemany(
            "INSERT INTO state_symbol_map VALUES (?,?)",
            [(c, StockCodeUtil.normalize(c)) for c in codes],
        )
        sql = f"SELECT m.symbol,try_cast(r.date AS DATE) AS trade_date,r.is_st::TINYINT AS is_st,r.confidence,r.source FROM {qi(risk)} r JOIN state_symbol_map m USING(code)"
        duplicate = con.execute(
            f"SELECT count(*) FROM (SELECT symbol,trade_date FROM ({sql}) GROUP BY 1,2 HAVING count(*)>1)"
        ).fetchone()[0]
        if duplicate:
            raise ValueError("Historical ST mapping has ambiguous symbol/date keys")
        part = target.with_suffix(".part.parquet")
        con.execute(f"COPY ({sql}) TO {qs(part)} (FORMAT PARQUET,COMPRESSION ZSTD)")
        os.replace(part, target)
        con.execute("DROP TABLE state_symbol_map")
    builder.publish(
        "security_state_candidates",
        [target],
        "security_state",
        "Supplier historical ST candidates, source/confidence retained; no independent PIT certification",
    )


def research_views(builder, root):
    con = builder.con
    materialize_state(builder, root)
    # Freeze each candidate source in the release. No 'current name => old ST' inference.
    con.execute(
        """CREATE OR REPLACE VIEW security_state AS SELECT * FROM security_state_candidates"""
    )
    con.execute("""CREATE OR REPLACE VIEW own_instrument_snapshot AS
        SELECT Symbol AS symbol,
          coalesce(try_cast(HqDate AS DATE),try_strptime(HqDate,'%Y%m%d')::DATE) AS trade_date,
          CASE WHEN try_cast(IsSTGP AS INTEGER) IN (0,1) THEN try_cast(IsSTGP AS TINYINT) END AS is_st
        FROM quantdb_instrument_detail""")
    con.execute("""CREATE OR REPLACE VIEW research_stock_bars AS WITH prices AS (
        SELECT symbol,try_cast(time AS DATE) AS trade_date,open,high,low,close,volume,amount*10000 AS amount,
          'raw' AS price_adjustment,'quantdb_daily_unadjusted' AS source_id FROM quantdb_daily_unadjusted
        UNION ALL
        SELECT symbol,try_cast(time AS DATE),open,high,low,close,volume,amount*10000,
          'qfq','quantdb_daily_forward' FROM quantdb_daily_forward
        UNION ALL
        SELECT symbol,try_cast(time AS DATE),open,high,low,close,volume,amount*10000,
          'hfq','quantdb_daily_backward' FROM quantdb_daily_backward
        UNION ALL
        SELECT symbol,trade_date,open,high,low,close,volume,amount,price_adjustment,source_id
          FROM daily_bars WHERE price_adjustment='raw' AND trade_date<DATE '2016-01-04'
            AND symbol NOT LIKE '90%.SH' AND symbol NOT LIKE '20%.SZ'
      ), annotated AS (
        SELECT p.*, 'stock' AS asset_type,
          CASE WHEN s.confidence='high' AND o.is_st IS NOT NULL AND s.is_st<>o.is_st THEN NULL
            ELSE coalesce(CASE WHEN s.confidence='high' THEN s.is_st END,o.is_st,i.is_st) END::TINYINT AS is_st,
          coalesce(o.float_shares,try_cast(v.circulating_capital AS DOUBLE)) AS float_shares,
          CASE WHEN s.confidence='high' AND o.is_st IS NOT NULL AND s.is_st<>o.is_st THEN 'conflicting_sources'
            WHEN s.confidence='high' THEN 'historical_supplier_high'
            WHEN o.is_st IS NOT NULL THEN 'offline_vendor'
            WHEN i.is_st IS NOT NULL THEN 'exact_day_instrument_snapshot' ELSE 'unknown' END AS state_source
        FROM prices p LEFT JOIN security_state s ON p.symbol=s.symbol AND p.trade_date=s.trade_date
        LEFT JOIN daily_bars o ON p.symbol=o.symbol AND p.trade_date=o.trade_date AND o.price_adjustment='raw'
        LEFT JOIN own_instrument_snapshot i ON p.symbol=i.symbol AND p.trade_date=i.trade_date
        LEFT JOIN quantdb_valuation v ON p.symbol=v.symbol AND p.trade_date=try_cast(v.time AS DATE)
      )
      SELECT *,concat_ws(';',
        CASE WHEN trade_date IS NULL THEN 'invalid_date' END,
        CASE WHEN open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR
          NOT isfinite(open) OR NOT isfinite(high) OR NOT isfinite(low) OR NOT isfinite(close) OR
          least(open,high,low,close)<=0 OR high<greatest(open,low,close) OR low>least(open,close)
          THEN 'invalid_ohlc' END,
        CASE WHEN volume IS NULL OR amount IS NULL OR volume<0 OR amount<0 OR NOT isfinite(volume) OR NOT isfinite(amount) THEN 'invalid_activity' END,
        CASE WHEN is_st IS NULL THEN 'unknown_st' END,
        CASE WHEN state_source='conflicting_sources' THEN 'state_conflict' END) AS quality_flags
      FROM annotated""")
    con.execute("""CREATE OR REPLACE VIEW research_etf_bars AS
        SELECT symbol,try_cast(time AS DATE) AS trade_date,open,high,low,close,volume,amount*10000 AS amount,
          'raw' AS price_adjustment,'quantdb_etf_kline' AS source_id,'etf' AS asset_type,
          NULL::TINYINT AS is_st,NULL::DOUBLE AS float_shares,'not_applicable' AS state_source,
          concat_ws(';',CASE WHEN close IS NULL OR close<=0 OR high<greatest(open,low,close) OR low>least(open,close) THEN 'invalid_ohlc' END,
            CASE WHEN volume IS NULL OR amount IS NULL OR volume<0 OR amount<0 THEN 'invalid_activity' END) AS quality_flags
        FROM quantdb_etf_kline""")
    con.execute(
        "CREATE OR REPLACE VIEW research_daily_bars AS SELECT * FROM research_stock_bars UNION ALL BY NAME SELECT * FROM research_etf_bars"
    )


def publish(root, base, run, plan, state, eastmoney, *, build_qlib=False):
    # Legacy catalog reconstruction is retained for migration/testing only.
    if build_qlib:
        raise ValueError("Persistent Qlib output is disabled; use temporary_provider")
    from backend.scripts.build_own_warehouse import Builder

    root, base, run = Path(root).resolve(), Path(base).resolve(), Path(run).resolve()
    if not eastmoney.is_dir():
        raise ValueError("Saved Eastmoney source directory is unavailable")
    active = (
        current_release(root)
        if (root / "CURRENT.json").exists()
        else current_release(base)
    )
    release_id = run.name
    release = root / "releases" / release_id
    if release.exists():
        # Failed stages are retained. A new stage name prevents overwriting evidence.
        release_id += "-retry-" + datetime.now(timezone.utc).strftime("%H%M%S%f")
        release = root / "releases" / release_id
    release.mkdir(parents=True)
    shutil.copyfile(active["database"], release / "warehouse.duckdb")
    compatible = release / "quantdb-compatible"
    clone_compatible(active["compatible"], compatible)
    selected = {}
    for dataset, _, obj in plan["entries"]:
        row = state.lookup(dataset, obj)
        if not row or row[2] != "complete":
            raise ValueError("Not all planned object versions are complete")
        path = Path(row[0])
        if not path.exists() or path.stat().st_size != obj["size"]:
            raise ValueError("Publish object missing or corrupt")
        selected.setdefault(dataset, []).append((obj, path))
    b = Builder(release, Path("/no-offline-source"), eastmoney)
    try:
        b.con.execute("BEGIN TRANSACTION")
        for dataset, objects in selected.items():
            # Replace only this dataset's link directory, preserving older releases.
            parents = {safe_relative(obj).parts[:2] for obj, _ in objects}
            if len(parents) != 1:
                raise ValueError("Unexpected dataset storage layout")
            directory = compatible.joinpath(*next(iter(parents)))
            if directory.exists():
                shutil.rmtree(directory)
            for obj, path in objects:
                target = compatible / safe_relative(obj)
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(path, target)
            b.publish(
                "quantdb_" + dataset,
                [path for _, path in objects],
                "quantdb",
                "Current manifest revisions; immutable originals retained in prior releases",
            )
        b.con.execute("""CREATE OR REPLACE TABLE quantdb_unit_audit AS
            SELECT try_cast(time AS DATE) AS trade_date,
              median(close*volume/nullif(amount,0)) FILTER(WHERE close>0 AND volume>0 AND amount>0) AS median_price_volume_amount_ratio
            FROM quantdb_daily_unadjusted GROUP BY 1""")
        bad_units = b.con.execute(
            "SELECT count(*) FROM quantdb_unit_audit WHERE median_price_volume_amount_ratio IS NULL OR median_price_volume_amount_ratio NOT BETWEEN 1000 AND 100000"
        ).fetchone()[0]
        if bad_units:
            raise ValueError(
                "QuantDB shares / 10k CNY convention failed the per-partition audit"
            )
        b.eastmoney_import(incremental=True)
        research_views(b, root)
        b.con.execute(
            "CREATE TABLE IF NOT EXISTS update_watermarks (release_id VARCHAR,dataset VARCHAR,last_source_date DATE,objects BIGINT)"
        )
        watermarks = {}
        for dataset, objects in selected.items():
            dates = [
                obj.get("trade_date") for obj, _ in objects if obj.get("trade_date")
            ]
            last = max(dates) if dates else None
            watermarks[dataset] = {"last_source_date": last, "objects": len(objects)}
            b.con.execute(
                "INSERT INTO update_watermarks VALUES (?,?,?,?)",
                [release_id, dataset, last, len(objects)],
            )
        checks = {}
        for dataset in [
            "daily_unadjusted",
            "daily_forward",
            "daily_backward",
            "etf_kline",
            "l1_factors",
            "l1_l2_factors",
            "features_daily",
        ]:
            if dataset not in selected:
                continue
            latest = max(
                selected[dataset],
                key=lambda pair: pair[0].get("trade_date", pair[0]["key"]),
            )[1]
            rel = f"read_parquet({qs(latest)})"
            columns = {
                r[0].lower(): r[0]
                for r in b.con.execute("DESCRIBE SELECT * FROM " + rel).fetchall()
            }
            symbol = columns.get("symbol")
            if not symbol:
                raise ValueError("Latest dataset is missing symbol: " + dataset)
            count, unique = b.con.execute(
                f"SELECT count(*),count(DISTINCT {qi(symbol)}) FROM {rel}"
            ).fetchone()
            if not count or count != unique:
                raise ValueError(
                    "Latest daily partition has empty/duplicate symbols: " + dataset
                )
            checks[dataset] = {"latest_rows": count, "fields": len(columns)}
        b.con.execute("COMMIT")
        qlib_result = None
        if build_qlib:
            from .qlib_export import build_provider

            qlib_result = build_provider(b.con, release / "qlib")
        b.finish()
    except Exception:
        try:
            b.con.execute("ROLLBACK")
        except Exception:
            pass
        b.con.close()
        raise
    source_list = {"manifests": plan["manifests"], "entries": plan["entries"]}
    atomic_json(release / "SOURCE-LIST.json", source_list)
    report = {
        "release_id": release_id,
        "status": "validated",
        "watermarks": watermarks,
        "latest_partition_checks": checks,
        "historical_state_independently_verified": False,
        "etf_adjustments": "raw_only",
        "one_minute_downloaded": False,
        "qlib": qlib_result,
    }
    atomic_json(release / "VALIDATION.json", report)
    pointer = {
        "release_id": release_id,
        "database": str((release / "warehouse.duckdb").relative_to(root)),
        "compatible": str(compatible.relative_to(root)),
        "source_list": str((release / "SOURCE-LIST.json").relative_to(root)),
        "validation_sha256": sha256(release / "VALIDATION.json"),
        "published_at": datetime.now(timezone.utc).isoformat(),
    }
    if build_qlib:
        pointer["qlib"] = str((release / "qlib").relative_to(root))
    # Researchers read CURRENT once to pin their whole session. Old releases remain usable.
    symlink_replace(root / "warehouse.duckdb", release / "warehouse.duckdb")
    symlink_replace(root / "quantdb-current", compatible)
    if build_qlib:
        symlink_replace(root / "qlib-current", release / "qlib")
    atomic_json(root / "CURRENT.json", pointer)
    atomic_json(root / "READY.json", pointer)
    shutil.copyfile(release / "FIELD-DICTIONARY.md", root / "FIELD-DICTIONARY.md")
    return report
