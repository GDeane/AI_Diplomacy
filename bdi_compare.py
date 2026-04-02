"""
BDI Distance Comparison — Compute similarity metrics between prediction and ground truth.

Takes a _bdi.json file (output from semantic_parser.py) and computes:
1. Relationship accuracy: ordinal distance across the 5-level scale per power, averaged
2. Agreement similarity: Jaccard index on (counterparty, type, region) tuples
3. Order similarity: set overlap between predicted and actual order strings

Usage:
    python bdi_compare.py \
        --bdi_file results/my_test_run_13/tom_probes/probe_A_TURKEY_predicts_AUSTRIA_S1902M_20260314_112633_bdi.json

    # Process all BDI files in a directory
    python bdi_compare.py \
        --bdi_dir results/my_test_run_13/tom_probes/
"""

import argparse
import json
import os
import re
from glob import glob
from typing import Dict, List, Set, Tuple

# Ordinal scale for relationships (0-4)
RELATIONSHIP_SCALE = {
    "Enemy": 0,
    "Unfriendly": 1,
    "Neutral": 2,
    "Friendly": 3,
    "Ally": 4,
}
MAX_RELATIONSHIP_DISTANCE = 4  # Enemy <-> Ally


def normalize_order(order: str) -> str:
    """Normalize an order string for comparison.

    Strips whitespace, uppercases, removes 'VIA' convoy annotations,
    and normalizes multiple spaces to single spaces.
    """
    order = order.strip().upper()
    order = re.sub(r"\s+VIA\b", "", order)
    order = re.sub(r"\s+", " ", order)
    return order


def order_to_unit_destination(order: str) -> Tuple[str, str, str]:
    """Extract (unit, action_type, destination) from a Diplomacy order string.

    Returns a tuple for partial-credit matching:
    - unit: e.g., "A BUD"
    - action_type: "move", "hold", "support", "convoy"
    - destination: target location for moves, supported unit's target for supports
    """
    order = normalize_order(order)
    parts = order.split()

    if len(parts) < 2:
        return (order, "unknown", "")

    unit = f"{parts[0]} {parts[1]}"

    if "H" in parts[2:] or "HOLD" in parts[2:]:
        return (unit, "hold", parts[1])

    if "S" in parts:
        s_idx = parts.index("S")
        support_tail = parts[s_idx + 1:]
        if "-" in support_tail:
            dest = support_tail[support_tail.index("-") + 1]
            return (unit, "support", dest)
        elif len(support_tail) >= 2:
            return (unit, "support", support_tail[1])
        return (unit, "support", "")

    if "C" in parts:
        c_idx = parts.index("C")
        convoy_tail = parts[c_idx + 1:]
        if "-" in convoy_tail:
            dest = convoy_tail[convoy_tail.index("-") + 1]
            return (unit, "convoy", dest)
        return (unit, "convoy", "")

    if "-" in parts:
        dash_idx = parts.index("-")
        if dash_idx + 1 < len(parts):
            return (unit, "move", parts[dash_idx + 1])

    return (unit, "hold", parts[1])


# ---------------------------------------------------------------------------
# Relationship distance
# ---------------------------------------------------------------------------

def relationship_distance(gt_rels: Dict[str, str], pred_rels: Dict[str, str]) -> dict:
    """Compute per-power ordinal distance and overall accuracy for relationships.

    Returns:
        {
            "per_power": {"POWER": {"gt": "Ally", "pred": "Friendly", "distance": 1}, ...},
            "mean_distance": float,       # 0 = perfect, 4 = worst
            "normalized_accuracy": float,  # 1 = perfect, 0 = worst (1 - mean_distance/4)
            "exact_matches": int,
            "total": int,
        }
    """
    all_powers = sorted(set(list(gt_rels.keys()) + list(pred_rels.keys())))
    per_power = {}
    total_distance = 0

    for power in all_powers:
        gt_level = gt_rels.get(power, "Neutral")
        pred_level = pred_rels.get(power, "Neutral")
        gt_val = RELATIONSHIP_SCALE.get(gt_level, 2)
        pred_val = RELATIONSHIP_SCALE.get(pred_level, 2)
        dist = abs(gt_val - pred_val)
        per_power[power] = {"gt": gt_level, "pred": pred_level, "distance": dist}
        total_distance += dist

    n = len(all_powers) if all_powers else 1
    mean_dist = total_distance / n
    exact = sum(1 for p in per_power.values() if p["distance"] == 0)

    return {
        "per_power": per_power,
        "mean_distance": round(mean_dist, 4),
        "normalized_accuracy": round(1 - mean_dist / MAX_RELATIONSHIP_DISTANCE, 4),
        "exact_matches": exact,
        "total": n,
    }


# ---------------------------------------------------------------------------
# Agreement similarity
# ---------------------------------------------------------------------------

def agreement_to_tuple(a: dict) -> Tuple[str, str, str]:
    """Convert an agreement dict to a comparable tuple."""
    return (
        a.get("counterparty", "").upper(),
        a.get("type", "").lower(),
        a.get("region", "").upper(),
    )


def agreement_similarity(gt_agreements: List[dict], pred_agreements: List[dict]) -> dict:
    """Compute Jaccard similarity on agreement sets.

    Returns:
        {
            "gt_set": [...],
            "pred_set": [...],
            "intersection": [...],
            "jaccard": float,  # |intersection| / |union|, 1 = perfect
            "precision": float,  # |intersection| / |pred|
            "recall": float,  # |intersection| / |gt|
        }
    """
    gt_set = set(agreement_to_tuple(a) for a in gt_agreements)
    pred_set = set(agreement_to_tuple(a) for a in pred_agreements)

    intersection = gt_set & pred_set
    union = gt_set | pred_set

    jaccard = len(intersection) / len(union) if union else 1.0
    precision = len(intersection) / len(pred_set) if pred_set else (1.0 if not gt_set else 0.0)
    recall = len(intersection) / len(gt_set) if gt_set else (1.0 if not pred_set else 0.0)

    return {
        "gt_set": sorted([list(t) for t in gt_set]),
        "pred_set": sorted([list(t) for t in pred_set]),
        "intersection": sorted([list(t) for t in intersection]),
        "jaccard": round(jaccard, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }


# ---------------------------------------------------------------------------
# Order similarity
# ---------------------------------------------------------------------------

def order_similarity(gt_orders: List[str], pred_orders: List[str]) -> dict:
    """Compute order prediction accuracy with exact and partial matching.

    Exact match: normalized order strings are identical.
    Unit match: the same unit is given an order (regardless of what order).
    Destination match: the same unit is sent to the same destination.

    Returns:
        {
            "gt_orders": [...],
            "pred_orders": [...],
            "exact_matches": [...],
            "exact_jaccard": float,
            "destination_accuracy": float, # fraction of GT (unit, dest) pairs matched
        }
    """
    gt_normalized = set(normalize_order(o) for o in gt_orders)
    pred_normalized = set(normalize_order(o) for o in pred_orders)

    exact_intersection = gt_normalized & pred_normalized
    exact_union = gt_normalized | pred_normalized
    exact_jaccard = len(exact_intersection) / len(exact_union) if exact_union else 1.0

    # Partial matching: destination-level
    gt_parsed = [order_to_unit_destination(o) for o in gt_orders]
    pred_parsed = [order_to_unit_destination(o) for o in pred_orders]

    gt_unit_dest = {(p[0], p[2]) for p in gt_parsed}
    pred_unit_dest = {(p[0], p[2]) for p in pred_parsed}
    dest_overlap = gt_unit_dest & pred_unit_dest
    destination_accuracy = len(dest_overlap) / len(gt_unit_dest) if gt_unit_dest else 1.0

    return {
        "gt_orders": sorted(gt_normalized),
        "pred_orders": sorted(pred_normalized),
        "exact_matches": sorted(exact_intersection),
        "exact_jaccard": round(exact_jaccard, 4),
        "destination_accuracy": round(destination_accuracy, 4),
    }


# ---------------------------------------------------------------------------
# Full comparison
# ---------------------------------------------------------------------------

def split_agreements_by_scope(agreements: List[dict], predictor_power: str) -> Tuple[List[dict], List[dict]]:
    """Split agreements into bilateral (with predictor) and third-party (with others).

    Bilateral agreements are those where the counterparty is the predictor power —
    these are directly knowable from conversation. Third-party agreements involve
    other powers and may only be partially inferrable from board state.
    """
    bilateral = []
    third_party = []
    for a in agreements:
        if a.get("counterparty", "").upper() == predictor_power.upper():
            bilateral.append(a)
        else:
            third_party.append(a)
    return bilateral, third_party


def compare_bdi(gt_bdi: dict, pred_bdi: dict, predictor_power: str = None) -> dict:
    """Compare a single prediction's BDI against ground truth.

    If predictor_power is provided, agreements are split into bilateral
    (with predictor) and third-party (with others) for separate scoring.
    """
    gt_agreements = gt_bdi.get("agreements", [])
    pred_agreements = pred_bdi.get("agreements", [])

    gt_rels = gt_bdi.get("relationships", {})
    pred_rels = pred_bdi.get("relationships", {})

    result = {
        "relationships": relationship_distance(gt_rels, pred_rels),
        "agreements_all": agreement_similarity(gt_agreements, pred_agreements),
        "orders": order_similarity(
            gt_bdi.get("intended_orders", []),
            pred_bdi.get("intended_orders", []),
        ),
    }

    if predictor_power:
        gt_bilateral, gt_third_party = split_agreements_by_scope(gt_agreements, predictor_power)
        pred_bilateral, pred_third_party = split_agreements_by_scope(pred_agreements, predictor_power)
        result["agreements_bilateral"] = agreement_similarity(gt_bilateral, pred_bilateral)
        result["agreements_third_party"] = agreement_similarity(gt_third_party, pred_third_party)

        # Bilateral relationship: target's relationship with predictor specifically
        bilateral_gt = {predictor_power: gt_rels.get(predictor_power, "Neutral")}
        bilateral_pred = {predictor_power: pred_rels.get(predictor_power, "Neutral")}
        result["relationship_bilateral"] = relationship_distance(bilateral_gt, bilateral_pred)

    return result


def process_bdi_file(bdi_file: str) -> dict:
    """Process a single BDI file and compute all comparisons."""
    with open(bdi_file, "r") as f:
        data = json.load(f)

    gt_bdi = data["ground_truth_bdi"]
    predictor_power = data["metadata"].get("predictor_power")
    results = []

    for pred in data["parsed_predictions"]:
        if pred["bdi"] is None:
            results.append({
                "model": pred["model"],
                "run_index": pred["run_index"],
                "error": pred.get("parse_error", "No BDI data"),
                "scores": None,
            })
            continue

        scores = compare_bdi(gt_bdi, pred["bdi"], predictor_power=predictor_power)
        results.append({
            "model": pred["model"],
            "run_index": pred["run_index"],
            "error": None,
            "scores": scores,
        })

    # Compute aggregate scores across successful predictions
    valid_results = [r for r in results if r["scores"] is not None]
    if valid_results:
        def _agg(key_path):
            """Helper to compute mean and per_run for a nested score key."""
            vals = []
            for r in valid_results:
                obj = r["scores"]
                for k in key_path:
                    obj = obj[k]
                vals.append(obj)
            return {"mean": round(sum(vals) / len(vals), 4), "per_run": vals}

        aggregate = {
            "relationship_accuracy": _agg(["relationships", "normalized_accuracy"]),
            "agreement_all_jaccard": _agg(["agreements_all", "jaccard"]),
            "order_exact_jaccard": _agg(["orders", "exact_jaccard"]),
            "order_destination_accuracy": _agg(["orders", "destination_accuracy"]),
        }

        # Add bilateral/third-party splits if available
        if valid_results[0]["scores"].get("agreements_bilateral"):
            aggregate["agreement_bilateral_jaccard"] = _agg(["agreements_bilateral", "jaccard"])
            aggregate["agreement_third_party_jaccard"] = _agg(["agreements_third_party", "jaccard"])
        if valid_results[0]["scores"].get("relationship_bilateral"):
            aggregate["relationship_bilateral_accuracy"] = _agg(["relationship_bilateral", "normalized_accuracy"])
    else:
        aggregate = None

    return {
        "metadata": data["metadata"],
        "per_prediction": results,
        "aggregate": aggregate,
    }


def print_summary(result: dict) -> None:
    """Print a human-readable summary of comparison results."""
    meta = result["metadata"]
    print(f"\n{'='*70}")
    print(f"  {meta['predictor_power']} predicting {meta['target_power']} @ {meta['phase']}")
    print(f"  Probe type: {meta['probe_type']}  |  Models: {', '.join(meta['models'])}")
    print(f"{'='*70}")

    for pred in result["per_prediction"]:
        if pred["error"]:
            print(f"\n  [{pred['model']} #{pred['run_index']}] ERROR: {pred['error']}")
            continue

        s = pred["scores"]
        print(f"\n  [{pred['model']} #{pred['run_index']}]")

        # Relationships
        rel = s["relationships"]
        print(f"    Relationships: accuracy={rel['normalized_accuracy']:.2f}  "
              f"({rel['exact_matches']}/{rel['total']} exact)")
        for power, info in sorted(rel["per_power"].items()):
            marker = "  " if info["distance"] == 0 else f"  (off by {info['distance']})"
            print(f"      {power:10s}  gt={info['gt']:10s}  pred={info['pred']:10s}{marker}")

        # Agreements (all)
        agr = s["agreements_all"]
        print(f"    Agreements (all):      jaccard={agr['jaccard']:.2f}  "
              f"precision={agr['precision']:.2f}  recall={agr['recall']:.2f}")
        if agr["intersection"]:
            print(f"      Matched: {agr['intersection']}")
        gt_only = [a for a in agr["gt_set"] if a not in agr["intersection"]]
        pred_only = [a for a in agr["pred_set"] if a not in agr["intersection"]]
        if gt_only:
            print(f"      Missed:  {gt_only}")
        if pred_only:
            print(f"      Extra:   {pred_only}")

        # Agreements (bilateral vs third-party)
        if "agreements_bilateral" in s:
            bilat = s["agreements_bilateral"]
            print(f"    Agreements (bilateral): jaccard={bilat['jaccard']:.2f}  "
                  f"precision={bilat['precision']:.2f}  recall={bilat['recall']:.2f}  "
                  f"(gt={len(bilat['gt_set'])}, pred={len(bilat['pred_set'])})")
            third = s["agreements_third_party"]
            print(f"    Agreements (3rd-party): jaccard={third['jaccard']:.2f}  "
                  f"precision={third['precision']:.2f}  recall={third['recall']:.2f}  "
                  f"(gt={len(third['gt_set'])}, pred={len(third['pred_set'])})")

        # Orders
        ord_ = s["orders"]
        print(f"    Orders:        exact_jaccard={ord_['exact_jaccard']:.2f}  "
              f"dest_acc={ord_['destination_accuracy']:.2f}")
        if ord_["exact_matches"]:
            print(f"      Exact matches: {ord_['exact_matches']}")
        gt_only = sorted(set(ord_["gt_orders"]) - set(ord_["exact_matches"]))
        pred_only = sorted(set(ord_["pred_orders"]) - set(ord_["exact_matches"]))
        if gt_only:
            print(f"      GT only:   {gt_only}")
        if pred_only:
            print(f"      Pred only: {pred_only}")

    # Aggregate
    if result["aggregate"]:
        agg = result["aggregate"]
        print(f"\n  {'─'*50}")
        print(f"  AGGREGATE (n={len([r for r in result['per_prediction'] if r['scores']])})")
        print(f"    Relationship accuracy (all): {agg['relationship_accuracy']['mean']:.4f}")
        if "relationship_bilateral_accuracy" in agg:
            print(f"    Relationship accuracy (bil): {agg['relationship_bilateral_accuracy']['mean']:.4f}")
        print(f"    Agreement Jaccard (all):     {agg['agreement_all_jaccard']['mean']:.4f}")
        if "agreement_bilateral_jaccard" in agg:
            print(f"    Agreement Jaccard (bilat.):  {agg['agreement_bilateral_jaccard']['mean']:.4f}")
            print(f"    Agreement Jaccard (3rd-p.):  {agg['agreement_third_party_jaccard']['mean']:.4f}")
        print(f"    Order exact Jaccard:         {agg['order_exact_jaccard']['mean']:.4f}")
        print(f"    Order destination accuracy:  {agg['order_destination_accuracy']['mean']:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="BDI Distance Comparison")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bdi_file", type=str, help="Path to a single _bdi.json file")
    group.add_argument("--bdi_dir", type=str, help="Path to directory containing _bdi.json files")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: same as input, with _scores suffix)")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output, only print aggregate")
    args = parser.parse_args()

    if args.bdi_file:
        bdi_files = [args.bdi_file]
    else:
        bdi_files = sorted(glob(os.path.join(args.bdi_dir, "*_bdi.json")))
        if not bdi_files:
            print(f"No _bdi.json files found in {args.bdi_dir}")
            return

    print(f"Processing {len(bdi_files)} BDI file(s)...")

    for bdi_file in bdi_files:
        result = process_bdi_file(bdi_file)

        if not args.quiet:
            print_summary(result)

        # Save scores
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(bdi_file))[0]
            output_path = os.path.join(args.output_dir, f"{base}_scores.json")
        else:
            output_path = bdi_file.replace("_bdi.json", "_scores.json")

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        if args.quiet and result["aggregate"]:
            meta = result["metadata"]
            agg = result["aggregate"]
            bilat_str = f"  agr_bilat={agg['agreement_bilateral_jaccard']['mean']:.3f}" if "agreement_bilateral_jaccard" in agg else ""
            print(f"  {meta['predictor_power']}->{meta['target_power']} @ {meta['phase']}: "
                  f"rel={agg['relationship_accuracy']['mean']:.3f}  "
                  f"agr_all={agg['agreement_all_jaccard']['mean']:.3f}{bilat_str}  "
                  f"ord={agg['order_exact_jaccard']['mean']:.3f}")

        print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
