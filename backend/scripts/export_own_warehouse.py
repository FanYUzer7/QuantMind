"""Export pinned native-training and Qlib datasets from the owned warehouse."""

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from backend.shared.stock_utils import StockCodeUtil
from backend.services.engine.data_platform.own_data import features
from backend.services.engine.data_platform.own_data.warehouse import Warehouse, sha256


def write_qlib(root, raw, adjusted, symbols, sessions):
    """Qlib adjusted price / factor = raw price; volume * factor = raw shares."""
    (root / "calendars").mkdir(parents=True)
    (root / "instruments").mkdir()
    (root / "features").mkdir()
    (root / "calendars/day.txt").write_text("\n".join(sessions) + "\n")
    instruments = []
    for symbol in symbols:
        r = (
            raw.loc[raw.symbol == symbol]
            .sort_values("trade_date")
            .reset_index(drop=True)
        )
        a = (
            adjusted.loc[adjusted.symbol == symbol]
            .sort_values("trade_date")
            .reset_index(drop=True)
        )
        if len(r) != len(sessions) or len(a) != len(sessions):
            raise ValueError("Qlib calendar mismatch")
        factor = a.close.to_numpy() / r.close.to_numpy()
        if not np.isfinite(factor).all() or (factor <= 0).any():
            raise ValueError("Invalid Qlib price ratio")
        code = StockCodeUtil.to_qlib(symbol)
        folder = root / "features" / code
        folder.mkdir()
        values = {k: a[k].to_numpy() for k in ["open", "high", "low", "close"]}
        values.update(
            factor=factor,
            volume=r.volume.to_numpy() / factor,
            amount=r.amount.to_numpy(),
            raw_close=r.close.to_numpy(),
            is_st=r.is_st.to_numpy(),
            change=a.close.pct_change(fill_method=None).to_numpy(),
        )
        for key, array in values.items():
            np.r_[0, np.asarray(array, dtype=np.float32)].astype("<f4").tofile(
                folder / (key + ".day.bin")
            )
        instruments.append(f"{code}\t{sessions[0]}\t{sessions[-1]}")
    (root / "instruments/all.txt").write_text("\n".join(instruments) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--calendar", type=Path, required=True)
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--warmup-start", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise SystemExit("Use a new immutable dataset version")
    calendar = json.loads(a.calendar.read_text())
    sessions = features.calendar_sessions(calendar, a.warmup_start, a.end)
    w = Warehouse(a.root)
    try:
        raw = w.daily(
            a.symbols, a.warmup_start, a.end, adjustment="raw", source="offline"
        )
        adjusted = w.daily(
            a.symbols, a.warmup_start, a.end, adjustment="qfq", source="offline"
        )
        for frame in [raw, adjusted]:
            if frame.quality_flags.ne("").any():
                raise ValueError("Quality issues in selected export window")
        raw, adjusted = [
            features.align_panel(f, a.symbols, sessions) for f in [raw, adjusted]
        ]
        frame = features.build_research_features(
            raw, adjusted, a.symbols, sessions, a.start, a.end, research_preview=True
        )
        source_files = w.connection.execute(
            "SELECT table_name,files_json FROM catalog_tables WHERE source_family='offline_daily' ORDER BY table_name"
        ).fetchall()
        source_hashes = {
            str(p): sha256(p) for _, paths in source_files for p in json.loads(paths)
        }
    finally:
        w.close()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.output.parent) as tmp:
        stage = Path(tmp) / "dataset"
        stage.mkdir()
        hashes = {}
        for year, part in frame.groupby(frame.trade_date.dt.year):
            path = stage / f"model_features_{year}.parquet"
            part.to_parquet(path, index=False)
            hashes[path.name] = sha256(path)
        metadata = {
            "dataset_id": a.output.name,
            "research_only": True,
            "production_enabled": False,
            "source_warehouse": str(a.root.resolve()),
            "source_files": source_hashes,
            "features": features.FEATURES,
            "feature_definitions": features.FEATURE_DEFINITIONS,
            "feature_implementation_sha256": sha256(Path(features.__file__)),
            "calendar_sha256": sha256(a.calendar),
            "files": hashes,
            "rows": len(frame),
            "symbols": a.symbols,
            "start": a.start,
            "end": a.end,
            "warmup_start": a.warmup_start,
            "available_at_verified": False,
            "corporate_actions_independently_verified": False,
            "qlib": "Task-scoped temporary conversion; not included in snapshots",
        }
        (stage / "manifest.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2)
        )
        stage.rename(a.output)
    print(
        json.dumps(
            {
                "output": str(a.output),
                "rows": len(frame),
                "features": features.FEATURES,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
