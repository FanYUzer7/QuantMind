"""Manual incremental update of one current warehouse; never builds Qlib data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time

from backend.shared.runtime_secrets import get_secret
from backend.services.engine.data_platform.own_data.sync import (
    Remote,
    SyncState,
    atomic_json,
)


def enforce_budget(plan, usage, config):
    available = max(
        0,
        min(config["monthly_budget_bytes"], usage["traffic_limit"])
        - usage["used_traffic"],
    )
    if plan["download_bytes"] > min(available, config["max_run_bytes"]):
        raise ValueError(
            f"Planned {plan['download_bytes']} bytes exceeds this run/cycle's remaining budget {min(available, config['max_run_bytes'])}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/quant"))
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--phase", choices=["plan", "apply", "publish", "daily"], default="daily"
    )
    parser.add_argument("--run-id")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    config_path = args.config or workspace / "configs/own-data-update.json"
    config = json.loads(config_path.read_text())
    root = workspace / config["warehouse"]
    from backend.services.engine.data_platform.own_data.current import is_current

    if not is_current(root):
        raise SystemExit("Current warehouse required; run migrate_own_current first")
    root.mkdir(parents=True, exist_ok=True)
    lock_handle = (root / ".update.lock").open("a")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit("Another update is already running") from exc
    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "update-%Y%m%dT%H%M%S%fZ"
    )
    if any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for c in run_id
    ):
        raise ValueError("Invalid run id")
    run = root / "_ops/runs" / run_id
    run.mkdir(parents=True, exist_ok=True)
    report_path = run / (args.phase + "-report.json")
    if report_path.exists():
        if args.phase not in {"apply", "publish"}:
            raise ValueError("Use a new run id; reports are immutable")
        attempt = 2
        while (run / f"{args.phase}-report-{attempt:02d}.json").exists():
            attempt += 1
        report_path = run / f"{args.phase}-report-{attempt:02d}.json"
    state = SyncState(root)
    report = {
        "run_id": run_id,
        "phase": args.phase,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(run / "config.json", config)
    try:
        remote = None if args.offline else Remote(get_secret("QUANTDB_API_KEY"))
        plan_path = run / "plan.json"
        if args.phase in {"plan", "daily"}:
            if plan_path.exists():
                raise ValueError("Frozen plan already exists")
            manifests = {}
            if args.offline:
                from backend.services.engine.data_platform.own_data.current import sources
                from backend.services.engine.data_platform.own_data.warehouse import Warehouse

                with Warehouse(root) as warehouse:
                    previous = sources(warehouse.connection)
                manifests = {
                    d: m for d, m in previous["manifests"].items()
                    if d in config["datasets"]
                }
                usage = {
                    "used_traffic": 0,
                    "traffic_limit": config["monthly_budget_bytes"],
                }
            else:
                usage = remote.usage()
                atomic_json(run / "usage-before.json", usage)
                for dataset, category in config["datasets"].items():
                    manifests[dataset] = remote.manifest(dataset, category)
                    print(
                        "manifest",
                        dataset,
                        len(manifests[dataset]["files"]),
                        manifests[dataset].get("total_size"),
                        flush=True,
                    )
                    time.sleep(config.get("manifest_interval_seconds", 6.2))
            plan = state.plan(
                manifests, config["datasets"], retry_failed=args.retry_failed
            )
            enforce_budget(plan, usage, config)
            plan = {**plan, "manifests": manifests, "categories": config["datasets"]}
            atomic_json(plan_path, plan)
            print(
                "download plan",
                plan["download_bytes"],
                "bytes; reused",
                plan["reused_objects"],
                "objects",
                flush=True,
            )
            report.update(
                download_bytes=plan["download_bytes"],
                reused_objects=plan["reused_objects"],
            )
        else:
            plan = json.loads(plan_path.read_text())
        if args.phase in {"apply", "daily"}:
            if args.offline and plan["download_bytes"]:
                raise ValueError("Offline update cannot download missing versions")
            if remote:
                # Recompute against the frozen manifest; completed objects are reused on resume.
                remaining = state.plan(
                    plan["manifests"],
                    plan["categories"],
                    retry_failed=args.retry_failed,
                )
                enforce_budget(remaining, remote.usage(), config)
                last_print = [0]

                def progress(done, failed):
                    if done + failed >= last_print[0] + 100 or failed:
                        print(
                            "download progress",
                            done,
                            "failed",
                            failed,
                            "planned",
                            len(remaining["todo"]),
                            flush=True,
                        )
                        last_print[0] = done + failed

                report["downloaded_objects"] = state.transfer(
                    remaining, remote, config["workers"], progress
                )
                atomic_json(run / "usage-after.json", remote.usage())
        if args.phase in {"publish", "daily"}:
            from backend.services.engine.data_platform.own_data.current import update_current

            report["update"] = update_current(
                root,
                run,
                plan,
                state,
                Path(config["eastmoney_root"]),
                drop_datasets=config.get("drop_datasets", []),
            )
        report["status"] = "planned" if args.phase == "plan" else "complete"
        if args.phase in {"publish", "daily"}:
            # The authoritative current manifest is in DuckDB. Keep recovery plans
            # only while an update is unfinished, rather than a full snapshot per day.
            plan_path.unlink(missing_ok=True)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(report_path, report)
        state.close()
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()
    print(json.dumps(report, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
