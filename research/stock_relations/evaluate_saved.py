"""New evaluation run for existing saved predictions; never retrain or overwrite."""

import argparse
import json
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from reference import load_reference, sha256
from train import record_run, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not all(
        re.fullmatch(r"[A-Za-z0-9_-]+", x) for x in (args.run_id, args.source_run)
    ):
        raise ValueError("Unsafe run id")
    root = args.workspace.resolve()
    source = root / "experiments/runs" / args.source_run
    run = root / "experiments/runs" / args.run_id
    if (root / "experiments/paper-records" / args.run_id).exists():
        raise FileExistsError(args.run_id)
    run.mkdir(exist_ok=False)
    cfg = json.loads((source / "research-config.json").read_text())
    ref = load_reference(root, cfg)
    from factor_research import Tee

    logfile = (run / "run.log").open("x", buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, logfile), Tee(sys.stderr, logfile)
    # Result data and metadata only; no prices, feature matrices, or weight copies.
    names = (
        "research-config.json",
        "config.json",
        "preprocessing.json",
        "sample-audit.json",
        "training-trace.csv",
        "optimizer-trace.parquet",
        "checkpoint-selection.csv",
        "model-verification.json",
        "predictions.parquet",
        "scores.parquet",
        "ic-summary.csv",
        "daily-ic.parquet",
        "paired-ic.csv",
        "v1-reproduction.csv",
    )
    provenance = json.loads((source / "provenance.json").read_text())
    provenance.update(
        evaluation_source_run=str(source),
        evaluation_only=True,
        evaluation_source_sha256={name: sha256(source / name) for name in names},
        evaluation_code_sha256=sha256(__file__),
    )
    write_json(run / "provenance.json", provenance)
    write_json(
        run / "status.json",
        {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "evaluation_only": True,
        },
    )
    try:
        for name in names:
            shutil.copy2(source / name, run / name)
        for script, folder, argv in (
            (
                Path(__file__).with_name("verify_training.py"),
                "independent-model-verification",
                ["--workspace", str(root), "--run-id", args.run_id],
            ),
        ):
            result = subprocess.run(
                [sys.executable, str(script), *argv], capture_output=True, text=True
            )
            (run / folder).mkdir(exist_ok=True)
            (run / folder / "verification.log").write_text(
                result.stdout + result.stderr
            )
            if result.returncode:
                raise RuntimeError("Independent model verification failed")
        scores = pd.read_parquet(run / "scores.parquet")
        ref.run_portfolios(
            scores, json.loads((run / "config.json").read_text()), provenance, run
        )
        script = (
            root
            / "experiments/paper-records"
            / cfg["reference_run"]
            / "code/verify_model_portfolio_backtest.py"
        )
        result = subprocess.run(
            [sys.executable, str(script), str(run)], capture_output=True, text=True
        )
        (run / "independent-verification").mkdir(exist_ok=True)
        (run / "independent-verification/verification.log").write_text(
            result.stdout + result.stderr
        )
        if result.returncode:
            raise RuntimeError("Independent portfolio verification failed")
        write_json(
            run / "status.json",
            {
                "status": "complete",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "evaluation_only": True,
                "training_source": args.source_run,
            },
        )
    except BaseException:
        (run / "failure.txt").write_text(traceback.format_exc())
        write_json(
            run / "status.json",
            {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "evaluation_only": True,
            },
        )
        record_run(run, cfg, root, "failed")
        raise
    print(record_run(run, cfg, root, "complete") / "paper.md", flush=True)


if __name__ == "__main__":
    main()
