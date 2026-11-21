"""Diagnostics for bounded panels. Hypotheses never alter data or pick a source."""

from collections import Counter
from statistics import median


def unique_rows(batch):
    grouped = {}
    for row in batch.rows:
        grouped.setdefault((row["symbol"], row["trade_date"]), []).append(row)
    return {k: v[0] for k, v in grouped.items() if len(v) == 1}, [
        k for k, v in grouped.items() if len(v) > 1
    ]


def scale_hypothesis(
    reference, candidate, canonical_field, raw_field, *, scales=(1, 100, 1000, 10000)
):
    """Compare candidate raw quantities with independently declared source units.

    Evidence requires >=10 observations, >=2 instruments, >=3 dates and >=95%
    agreement within 0.01%. Zero/missing/negative rows provide no scale evidence.
    This is a diagnostic of this panel, not automatic metadata inference.
    """
    a, duplicates_a = unique_rows(reference)
    b, duplicates_b = unique_rows(candidate)
    pairs = []
    for key in sorted(a.keys() & b.keys()):
        x, y = a[key].get(canonical_field), b[key].get(raw_field)
        if x is not None and y is not None and x > 0 and y > 0:
            pairs.append((key, x, y))
    symbols = sorted({key[0] for key, _, _ in pairs})
    days = sorted({key[1] for key, _, _ in pairs})
    eligible = len(pairs) >= 10 and len(symbols) >= 2 and len(days) >= 3
    results = []
    for scale in scales:
        errors = [abs(y * scale - x) / abs(x) for _, x, y in pairs]
        matching = sum(err <= 1e-4 for err in errors)
        results.append(
            {
                "scale": scale,
                "matching_rows": matching,
                "matching_fraction": matching / len(errors) if errors else None,
                "median_relative_error": median(errors) if errors else None,
                "max_relative_error": max(errors) if errors else None,
            }
        )
    winners = [r for r in results if eligible and r["matching_fraction"] >= 0.95]
    return {
        "canonical_field": canonical_field,
        "candidate_raw_field": raw_field,
        "usable_pairs": len(pairs),
        "symbols": symbols,
        "dates": days,
        "duplicate_keys_excluded": len(duplicates_a) + len(duplicates_b),
        "hypotheses": results,
        "status": "sample_supported"
        if len(winners) == 1
        else "insufficient_evidence"
        if not eligible
        else "inconsistent_or_ambiguous",
        "supported_scale": winners[0]["scale"] if len(winners) == 1 else None,
        "automatic_application": False,
        "scope": "these instruments and dates only",
    }


def diagnose_panel(reference, candidate):
    a, duplicates_a = unique_rows(reference)
    b, duplicates_b = unique_rows(candidate)
    keys = sorted(a.keys() & b.keys())
    fields = {}
    for field in ("open", "high", "low", "close", "volume_shares", "amount_cny"):
        pairs = [
            (k, a[k].get(field), b[k].get(field))
            for k in keys
            if a[k].get(field) is not None and b[k].get(field) is not None
        ]
        errors = [abs(y - x) for _, x, y in pairs]
        fields[field] = {
            "paired_rows": len(pairs),
            "equal_within_1e_6": sum(err <= 1e-6 for err in errors),
            "max_absolute_difference": max(errors) if errors else None,
            "mismatch_examples": [
                {"symbol": k[0], "trade_date": k[1], "reference": x, "candidate": y}
                for k, x, y in pairs
                if abs(y - x) > 1e-6
            ][:8],
        }
    adjustments = sorted(
        {(a[k]["price_adjustment"], b[k]["price_adjustment"]) for k in keys}
    )
    comparable = bool(keys) and all(x == y and x != "unknown" for x, y in adjustments)
    kinds = sorted({(a[k]["data_kind"], b[k]["data_kind"]) for k in keys})
    comparable = comparable and all(x == y for x, y in kinds)
    price_errors = {"invalid_ohlc", "missing_or_invalid_ohlc"}
    comparable = comparable and not any(
        price_errors.intersection(r["quality_flags"])
        for k in keys
        for r in (a[k], b[k])
    )
    comparable = comparable and all(
        a[k]["asset_type"] == b[k]["asset_type"] for k in keys
    )
    return {
        "reference": reference.source_id,
        "candidate": candidate.source_id,
        "overlap_keys": len(keys),
        "reference_only": len(a.keys() - b.keys()),
        "candidate_only": len(b.keys() - a.keys()),
        "duplicate_keys": {"reference": duplicates_a, "candidate": duplicates_b},
        "declared_price_bases": adjustments,
        "data_kinds": kinds,
        "declared_price_comparable": comparable,
        "fields": fields,
        "volume_scale": scale_hypothesis(
            reference, candidate, "volume_shares", "volume_raw"
        ),
        "amount_scale": scale_hypothesis(
            reference, candidate, "amount_cny", "amount_raw"
        ),
        "selection_performed": False,
        "note": "Price differences are diagnostic even when bases are unknown/different; equality does not certify adjustment or full-history quality",
    }


def adjustment_relations(raw, qfq, hfq):
    a, _ = unique_rows(raw)
    b, _ = unique_rows(qfq)
    c, _ = unique_rows(hfq)
    keys = sorted(a.keys() & b.keys() & c.keys())
    result = []
    for symbol in sorted({k[0] for k in keys}):
        values = []
        for key in keys:
            if key[0] == symbol and all(
                x[key]["close"] is not None and x[key]["close"] > 0 for x in (a, b, c)
            ):
                values.append(
                    {
                        "date": key[1],
                        "raw_close": a[key]["close"],
                        "qfq_close": b[key]["close"],
                        "hfq_close": c[key]["close"],
                        "qfq_over_hfq": b[key]["close"] / c[key]["close"],
                    }
                )
        result.append(
            {
                "symbol": symbol,
                "observations": values,
                "ratio_range": [
                    min(v["qfq_over_hfq"] for v in values),
                    max(v["qfq_over_hfq"] for v in values),
                ]
                if values
                else None,
            }
        )
    return {
        "per_symbol": result,
        "note": "Observed relationships only: adjustment anchor/date and algorithm still need verification; do not splice different bases",
    }


def assess_feature_input(batch):
    """Conservative admission report; observing prices is not production approval."""
    blockers = Counter()
    if batch.evidence_only:
        blockers["evidence_only"] += 1
    if batch.truncated:
        blockers["truncated"] += 1
    if not batch.rows:
        blockers["empty"] += 1
    for row in batch.rows:
        for flag in row["quality_flags"]:
            blockers[flag] += 1
        if row["volume_shares"] is None or row["amount_cny"] is None:
            blockers["canonical_quantity_unavailable"] += 1
    if batch.issues:
        blockers["reader_issues"] += len(batch.issues)
    return {
        "ready": not blockers,
        "blockers": dict(blockers),
        "scope": "daily price/activity contract only; calendar, membership, ST, labels and feature semantics require separate validation",
        "production_route_changed": False,
    }
