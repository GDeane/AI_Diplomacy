"""
Generate "guess neutral + random orders" baseline predictions for ToM probes.

For each probe file, creates a _bdi.json with N baseline predictions where:
- All relationships are set to "Neutral"
- Agreements are empty
- Orders are randomly selected (one per unit) from the legal moves

Usage:
    python generate_baseline.py
"""

import json
import glob
import os
import re
import random

ALL_POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]
N_BASELINE = 5
SEED = 42


def parse_legal_orders_from_prompt(prompt: str, target_power: str) -> dict[str, list[str]]:
    """Parse legal orders per unit from the probe prompt text.

    Returns dict mapping unit location to list of legal orders.
    """
    # Find the section listing target's legal orders
    marker = f"{target_power}'s orderable units and legal moves:"
    start = prompt.find(marker)
    if start < 0:
        return {}

    section = prompt[start + len(marker):]
    # Cut off at next major section (starts with a line not indented)
    end_match = re.search(r'\n\S', section)
    if end_match:
        section = section[:end_match.start()]

    # Parse "Unit at XXX:" blocks
    orders_by_unit = {}
    current_unit = None
    for line in section.split('\n'):
        line = line.strip()
        unit_match = re.match(r'Unit at (\w+):', line)
        if unit_match:
            current_unit = unit_match.group(1)
            orders_by_unit[current_unit] = []
        elif line.startswith('- ') and current_unit:
            orders_by_unit[current_unit].append(line[2:].strip())

    return orders_by_unit


def generate_random_orders(orders_by_unit: dict[str, list[str]], rng: random.Random) -> list[str]:
    """Select one random order per unit."""
    selected = []
    for unit_loc, legal_orders in orders_by_unit.items():
        if legal_orders:
            selected.append(rng.choice(legal_orders))
    return selected


def generate_baseline_bdi(probe_path: str, n: int, rng: random.Random) -> dict:
    """Generate a baseline _bdi.json for a given probe file."""
    with open(probe_path) as f:
        probe = json.load(f)

    meta = probe["metadata"]
    target = meta["target_power"]
    predictor = meta["predictor_power"]

    # Get ground truth BDI (using post-phase relationships)
    gt = probe["ground_truth"]
    gt_relationships = gt.get("target_relationships_post_phase", {})
    gt_orders = gt.get("target_orders", [])

    # Parse legal orders from prompt
    orders_by_unit = parse_legal_orders_from_prompt(probe["prompt_used"], target)

    # Build neutral relationships (exclude self)
    neutral_relationships = {p: "Neutral" for p in ALL_POWERS if p != target}

    # Generate N baseline predictions
    predictions = []
    for i in range(n):
        random_orders = generate_random_orders(orders_by_unit, rng)
        predictions.append({
            "model": "baseline_neutral_random",
            "run_index": i,
            "parse_error": None,
            "bdi": {
                "relationships": neutral_relationships,
                "agreements": [],
                "intended_orders": random_orders,
            },
        })

    # We need ground truth BDI in the same format as semantic_parser outputs.
    # For agreements, we'd normally use the LLM parser, but for baseline we can
    # just copy from an existing _bdi.json if available, otherwise leave empty.
    gt_bdi = {
        "relationships": gt_relationships,
        "agreements": [],  # Will be populated from existing _bdi.json if available
        "intended_orders": gt_orders,
    }

    return {
        "metadata": {
            **meta,
            "models": ["baseline_neutral_random"],
            "num_predictions": n,
            "parser_model": "baseline",
            "parsed_at": "baseline",
        },
        "ground_truth_bdi": gt_bdi,
        "parsed_predictions": predictions,
    }


def copy_gt_agreements(baseline_bdi: dict, existing_bdi_path: str):
    """Copy ground truth agreements from an existing _bdi.json (parsed by the real parser)."""
    if os.path.exists(existing_bdi_path):
        with open(existing_bdi_path) as f:
            existing = json.load(f)
        baseline_bdi["ground_truth_bdi"]["agreements"] = existing["ground_truth_bdi"].get("agreements", [])


def main():
    rng = random.Random(SEED)

    # Create baseline output directory
    for run in ["my_test_run_4", "my_test_run_14"]:
        for probe_type in ["probe_A", "probe_B"]:
            os.makedirs(f"results/{run}/tom_probes/{probe_type}/baseline", exist_ok=True)

    # Find all probe files (excluding bdi/scores)
    probe_files = sorted(glob.glob("results/my_test_run_*/tom_probes/probe_*/**/probe_*.json", recursive=True))
    probe_files = [f for f in probe_files if "_bdi" not in f and "_scores" not in f and "/baseline/" not in f]

    # Group by situation (run_dir + phase + predictor + target + probe_type) — use one probe per situation
    seen_situations = set()
    unique_probes = []
    for pf in probe_files:
        with open(pf) as f:
            meta = json.load(f)["metadata"]
        key = (meta["run_dir"].split("/")[-1], meta["phase"], meta["predictor_power"], meta["target_power"], meta["probe_type"])
        if key not in seen_situations:
            seen_situations.add(key)
            unique_probes.append(pf)

    print(f"Generating baselines for {len(unique_probes)} unique situation/probe combinations...")

    for pf in unique_probes:
        with open(pf) as f:
            meta = json.load(f)["metadata"]

        run_dir = os.path.basename(meta["run_dir"])
        probe_type = f"probe_{meta['probe_type']}"
        predictor = meta["predictor_power"]
        target = meta["target_power"]
        phase = meta["phase"]

        # Generate baseline
        baseline = generate_baseline_bdi(pf, N_BASELINE, rng)

        # Copy GT agreements from the real model's _bdi.json
        # Use any existing _bdi.json from the same situation (e.g., gemini's)
        for model_dir in ["gemini", "claude", "gpt"]:
            existing_pattern = f"results/{run_dir}/tom_probes/{probe_type}/{model_dir}/*_{predictor}_predicts_{target}_{phase}_*_bdi.json"
            existing_files = glob.glob(existing_pattern)
            if existing_files:
                copy_gt_agreements(baseline, existing_files[0])
                break

        # Save
        output_dir = f"results/{run_dir}/tom_probes/{probe_type}/baseline"
        output_name = f"{probe_type}_{predictor}_predicts_{target}_{phase}_baseline_bdi.json"
        output_path = os.path.join(output_dir, output_name)

        with open(output_path, "w") as f:
            json.dump(baseline, f, indent=2, ensure_ascii=False)

        print(f"  Saved: {output_path}")

    print(f"\nDone. Generated {len(unique_probes)} baseline _bdi.json files.")
    print("Run bdi_compare on the baseline directories to score them.")


if __name__ == "__main__":
    main()
