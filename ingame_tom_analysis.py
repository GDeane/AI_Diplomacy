"""
In-Game Theory of Mind Analysis — Measure how accurately agents model each other during play.

For each phase, compares every agent's stated relationship about every other agent
against that other agent's actual relationship back. This reveals cases where agents
are successfully deceived (e.g., Turkey believes Austria is an Ally, but Austria
views Turkey as an Enemy).

Usage:
    python ingame_tom_analysis.py --game_file results/my_test_run_14/lmvsgame.json

    # Only analyze specific phases
    python ingame_tom_analysis.py --game_file results/my_test_run_14/lmvsgame.json \
        --phases S1902M F1903M

    # Only analyze specific power pairs
    python ingame_tom_analysis.py --game_file results/my_test_run_14/lmvsgame.json \
        --powers TURKEY AUSTRIA
"""

import argparse
import json
import os

ALL_POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

RELATIONSHIP_SCALE = {
    "Enemy": 0,
    "Unfriendly": 1,
    "Neutral": 2,
    "Friendly": 3,
    "Ally": 4,
}
MAX_DISTANCE = 4


def analyze_phase(state_agents: dict, phase_name: str, power_filter: list = None) -> dict:
    """Analyze ToM accuracy for a single phase.

    For each ordered pair (A, B), compares:
    - A's belief about B (A's relationship toward B)
    - B's actual stance toward A (B's relationship toward A)

    The asymmetry is intentional: A believes B is "Ally", but B actually views A as "Enemy".
    This measures whether A has accurately modeled B's disposition.
    """
    pairs = []
    for believer in ALL_POWERS:
        if believer not in state_agents:
            continue
        believer_rels = state_agents[believer].get("relationships", {})
        for target in ALL_POWERS:
            if target == believer or target not in state_agents:
                continue
            if power_filter and believer not in power_filter and target not in power_filter:
                continue

            # What does believer think about target?
            belief = believer_rels.get(target, "Neutral")
            # What does target actually think about believer?
            target_rels = state_agents[target].get("relationships", {})
            reality = target_rels.get(believer, "Neutral")

            belief_val = RELATIONSHIP_SCALE.get(belief, 2)
            reality_val = RELATIONSHIP_SCALE.get(reality, 2)
            distance = abs(belief_val - reality_val)

            pairs.append({
                "believer": believer,
                "target": target,
                "belief": belief,
                "reality": reality,
                "distance": distance,
                "accurate": distance == 0,
                "deceived": belief_val > reality_val,  # believer thinks target is friendlier than they are
                "suspicious": belief_val < reality_val,  # believer thinks target is less friendly than they are
            })

    n = len(pairs) if pairs else 1
    mean_distance = sum(p["distance"] for p in pairs) / n
    accuracy = 1 - mean_distance / MAX_DISTANCE
    exact_matches = sum(1 for p in pairs if p["accurate"])
    deceived_count = sum(1 for p in pairs if p["deceived"])

    return {
        "phase": phase_name,
        "pairs": pairs,
        "mean_distance": round(mean_distance, 4),
        "normalized_accuracy": round(accuracy, 4),
        "exact_matches": exact_matches,
        "total_pairs": len(pairs),
        "deceived_count": deceived_count,
    }


def find_notable_asymmetries(phase_result: dict, min_distance: int = 2) -> list:
    """Find pairs where belief and reality diverge significantly."""
    return [p for p in phase_result["pairs"] if p["distance"] >= min_distance]


def analyze_game(game_file: str, phase_filter: list = None, power_filter: list = None) -> dict:
    with open(game_file, "r") as f:
        game_data = json.load(f)

    phases = game_data.get("phases", [])
    results = []

    for phase in phases:
        phase_name = phase.get("name", "")
        if phase_filter and phase_name not in phase_filter:
            continue

        state_agents = phase.get("state_agents", {})
        if not state_agents:
            continue

        phase_result = analyze_phase(state_agents, phase_name, power_filter)
        results.append(phase_result)

    # Compute game-wide aggregate
    all_pairs = [p for r in results for p in r["pairs"]]
    if all_pairs:
        n = len(all_pairs)
        game_aggregate = {
            "mean_distance": round(sum(p["distance"] for p in all_pairs) / n, 4),
            "normalized_accuracy": round(1 - sum(p["distance"] for p in all_pairs) / (n * MAX_DISTANCE), 4),
            "exact_matches": sum(1 for p in all_pairs if p["accurate"]),
            "total_pairs": n,
            "deceived_count": sum(1 for p in all_pairs if p["deceived"]),
        }
    else:
        game_aggregate = None

    return {
        "game_file": game_file,
        "phase_filter": phase_filter,
        "power_filter": power_filter,
        "per_phase": results,
        "game_aggregate": game_aggregate,
    }


def print_results(analysis: dict, verbose: bool = False):
    print(f"\n{'='*70}")
    print(f"  In-Game ToM Analysis: {os.path.basename(analysis['game_file'])}")
    if analysis["power_filter"]:
        print(f"  Power filter: {', '.join(analysis['power_filter'])}")
    print(f"{'='*70}")

    for phase_result in analysis["per_phase"]:
        phase = phase_result["phase"]
        print(f"\n  Phase {phase}:  accuracy={phase_result['normalized_accuracy']:.2f}  "
              f"({phase_result['exact_matches']}/{phase_result['total_pairs']} exact)  "
              f"deceived={phase_result['deceived_count']}")

        notable = find_notable_asymmetries(phase_result)
        if notable:
            for p in sorted(notable, key=lambda x: -x["distance"]):
                direction = "DECEIVED" if p["deceived"] else "SUSPICIOUS"
                print(f"    {direction}: {p['believer']} thinks {p['target']}={p['belief']}, "
                      f"but {p['target']} views {p['believer']}={p['reality']}  "
                      f"(off by {p['distance']})")

        if verbose:
            for p in sorted(phase_result["pairs"], key=lambda x: (x["believer"], x["target"])):
                if p["distance"] == 0:
                    continue
                print(f"    {p['believer']:10s} thinks {p['target']:10s}={p['belief']:10s}  "
                      f"actual={p['reality']:10s}  (off by {p['distance']})")

    if analysis["game_aggregate"]:
        agg = analysis["game_aggregate"]
        print(f"\n  {'─'*50}")
        print(f"  GAME AGGREGATE ({len(analysis['per_phase'])} phases)")
        print(f"    Normalized accuracy:  {agg['normalized_accuracy']:.4f}")
        print(f"    Exact matches:        {agg['exact_matches']}/{agg['total_pairs']}")
        print(f"    Deceived instances:   {agg['deceived_count']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="In-Game Theory of Mind Analysis")
    parser.add_argument("--game_file", type=str, required=True, help="Path to lmvsgame.json")
    parser.add_argument("--phases", nargs="+", default=None, help="Only analyze specific phases (e.g., S1902M F1903M)")
    parser.add_argument("--powers", nargs="+", default=None, help="Only analyze pairs involving these powers")
    parser.add_argument("--verbose", action="store_true", help="Show all mismatched pairs, not just notable ones")
    parser.add_argument("--output_file", type=str, default=None, help="Save JSON results to file")
    args = parser.parse_args()

    analysis = analyze_game(args.game_file, phase_filter=args.phases, power_filter=args.powers)
    print_results(analysis, verbose=args.verbose)

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {args.output_file}")
    else:
        # Default: save next to game file
        output_path = args.game_file.replace("lmvsgame.json", "ingame_tom_analysis.json")
        with open(output_path, "w") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {output_path}")


if __name__ == "__main__":
    main()
