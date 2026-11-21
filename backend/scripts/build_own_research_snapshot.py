"""Validate continuous source data and atomically publish a research-only snapshot."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from backend.services.engine.data_platform.own_data import features
from backend.services.engine.data_platform.own_data.contracts import cn_symbol


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_panels(run, manifest):
    frames = {}
    for item in manifest["files"]:
        path = run / item["sample"]
        if digest(path) != item["sha256"]:
            raise ValueError("Continuous source sample hash mismatch")
        source = pd.read_csv(path, dtype={"__symbol": str, "__trade_date": str})
        base = pd.DataFrame(
            {
                "symbol": source["__symbol"].map(cn_symbol),
                "trade_date": source["__trade_date"],
            }
        )
        for canonical, original in (
            ("open", "开盘价"),
            ("high", "最高价"),
            ("low", "最低价"),
            ("close", "收盘价"),
        ):
            base[canonical] = pd.to_numeric(source[original], errors="raise")
        if item["dataset"] == "factor_crosscheck":
            base["adj_factor"] = pd.to_numeric(source["复权因子"], errors="raise")
        else:
            base["volume"] = pd.to_numeric(source["成交量（股）"], errors="raise")
            base["amount"] = pd.to_numeric(source["成交额（元）"], errors="raise")
            base["float_shares"] = pd.to_numeric(
                source["流通股本（股）"], errors="raise"
            )
            # No inferred ST=0 default. Unknown values remain missing and fail validation.
            base["is_st"] = source["是否ST"].map({"否": 0, "是": 1})
        frames.setdefault(item["dataset"], []).append(base)
    return {key: pd.concat(value, ignore_index=True) for key, value in frames.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, default=Path("/quant"))
    ap.add_argument("--run", default="experiments/data-integration/own-data-continuous-20261101-01")
    ap.add_argument(
        "--calendar", default="configs/calendar-cn-sh-sz-2025-research-window.json"
    )
    ap.add_argument("--dataset", default="datasets/own-daily-research-v001")
    ap.add_argument("--attempt", default="01")
    args = ap.parse_args()
    run, target = args.workspace / args.run, args.workspace / args.dataset
    report_path = run / f"continuous-validation-{args.attempt}.json"
    if target.exists() or report_path.exists():
        raise SystemExit("Dataset/report already exists; use a new version/attempt")
    raw_manifest = json.loads((run / "manifest.json").read_text())
    calendar_path = args.workspace / args.calendar
    calendar = json.loads(calendar_path.read_text())
    report = {
        "source_manifest_relative": args.run + "/manifest.json",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "validating",
        "research_only": True,
        "production_enabled": False,
        "backtest_execution_ready": False,
        "calendar": calendar,
        "source_manifest_sha256": digest(run / "manifest.json"),
        "calendar_sha256": digest(calendar_path),
        "feature_implementation_sha256": digest(Path(features.__file__)),
        "errors": [],
    }
    try:
        if raw_manifest["status"] != "complete":
            raise ValueError("Source extraction incomplete")
        symbols = list(raw_manifest["symbols"].values())
        sessions = features.calendar_sessions(
            calendar, raw_manifest["start"], raw_manifest["end"]
        )
        panels = load_panels(run, raw_manifest)
        report["source_validation"] = features.validate_continuous(
            panels["raw"],
            panels["qfq"],
            panels["hfq"],
            panels["factor_crosscheck"],
            symbols,
            sessions,
        )
        report["sessions"] = sessions
        frame = features.build_research_features(
            panels["raw"],
            panels["qfq"],
            symbols,
            sessions,
            "2025-07-01",
            "2025-09-30",
            research_preview=True,
        )
        code_snapshot = run / ("features-implementation-" + args.attempt + ".py")
        with code_snapshot.open("xb") as handle:
            handle.write(Path(features.__file__).read_bytes())
        report["feature_implementation_snapshot_relative"] = (
            args.run + "/" + code_snapshot.name
        )
        stage = run / ("snapshot-staging-" + args.attempt)
        stage.mkdir(exist_ok=False)
        hashes = {}
        for year, group in frame.groupby(frame["trade_date"].dt.year):
            path = stage / f"model_features_{year}.parquet"
            group.to_parquet(path, index=False)
            hashes[path.name] = digest(path)
        metadata = {
            **report,
            "status": "published_research_preview",
            "dataset_id": target.name,
            "features": features.FEATURES,
            "feature_definitions": features.FEATURE_DEFINITIONS,
            "rows": len(frame),
            "symbols": symbols,
            "first": str(frame["trade_date"].min().date()),
            "last": str(frame["trade_date"].max().date()),
            "files": hashes,
            "warmup_start": raw_manifest["start"],
            "ohlc_price_basis": "vendor_qfq_frozen_at_source_file_version",
            "raw_execution_columns": ["raw_open", "raw_high", "raw_low", "raw_close"],
            "st_status_source": "vendor historical CSV 是否ST; copied, not independently verified",
            "available_at_verified": False,
            "calendar_scope": "SH/SZ reviewed window only",
            "native_compatibility": "LOCAL parquet; factor_source unset; training/inference only; no Qlib/simulation activation",
            "runtime_versions": {
                name: importlib.metadata.version(name)
                for name in ["pandas", "pyarrow", "numpy"]
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        # Publish only a complete new directory. No old dataset is overwritten.
        if target.exists():
            raise FileExistsError("Refusing to replace an existing dataset")
        stage.rename(target)
        report.update(
            status="published_research_preview",
            dataset=str(target),
            rows=len(frame),
            files=hashes,
        )
    except Exception as error:
        report["status"] = "blocked"
        report["errors"].append({"type": type(error).__name__, "message": str(error)})
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "dataset": report["dataset"],
                "rows": report["rows"],
                "adjustment_events": len(
                    report["source_validation"]["adjustment_events"]
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
