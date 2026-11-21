"""Bounded command-line research over one pinned integrated warehouse release."""

import argparse
import json
from pathlib import Path

from backend.services.engine.data_platform.own_data.warehouse import Warehouse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sql")
    parser.add_argument("--symbols", nargs="+")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--adjust", default="raw", choices=["raw", "qfq", "hfq"])
    parser.add_argument("--fields", nargs="+")
    parser.add_argument(
        "--factor-source",
        default="l1_l2_factors",
        choices=["l1_factors", "l1_l2_factors", "features_daily"],
    )
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    if not 1 <= args.limit <= 10000:
        raise ValueError("limit must be 1..10000")
    if not any((args.root / p).is_file() for p in ("STORE.json", "CURRENT.json")):
        raise ValueError("Integrated source has not been published")
    w = Warehouse(args.root)
    try:
        print(
            json.dumps(
                {
                    "mode": "current" if w.release["release_id"] == "current" else w.release["release_id"],
                    "database": str(w.release["database"]),
                },
                ensure_ascii=False,
            )
        )
        if args.sql:
            statements = w.connection.extract_statements(args.sql)
            if (
                len(statements) != 1
                or str(statements[0].type) != "StatementType.SELECT"
            ):
                raise ValueError("Research SQL accepts one SELECT/WITH statement")
            sql = args.sql.rstrip().rstrip(";")
            frame = w.connection.execute(
                f"SELECT * FROM ({sql}) LIMIT {args.limit}"
            ).fetchdf()
        elif args.symbols:
            if not args.start or not args.end:
                raise ValueError("Specify start and end")
            if args.fields:
                frame = w.factors(
                    args.symbols,
                    args.start,
                    args.end,
                    args.fields,
                    source=args.factor_source,
                )
            else:
                frame = w.daily(
                    args.symbols,
                    args.start,
                    args.end,
                    adjustment=args.adjust,
                    source="integrated",
                )
        else:
            frame = w.tables()
        print(frame.head(args.limit).to_string(index=False))
    finally:
        w.close()


if __name__ == "__main__":
    main()
