"""
Critical State Analysis — Theory of Mind Probes for AI Diplomacy

Given a completed game, a phase, a predictor power, and a target power, this script:
1. Reconstructs the predictor's available context at diary-generation time
2. Prompts frontier models to predict the target's diary entry and/or moves
3. Saves structured predictions alongside ground truth for later semantic parsing

Usage:
    python critical_state_analysis.py \
        --run_dir results/my_test_run_11 \
        --phase S1902M \
        --predictor_power AUSTRIA \
        --target_power ITALY \
        --models "claude-opus-4-6,gpt-4o" \
        --num_predictions 3 \
        --probe_type A
"""

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from argparse import Namespace
from typing import Dict, List, Optional

from ai_diplomacy.clients import load_model_client
from ai_diplomacy.game_logic import load_game_state
from ai_diplomacy.utils import gather_possible_orders, get_board_state

logger = logging.getLogger(__name__)

ALL_POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

PROBE_SYSTEM_PROMPT = """\
You are an expert Diplomacy player and analyst. You will be given the perspective \
of one power in an ongoing game and asked to predict what another power is thinking \
and planning. Use strategic reasoning, analysis of negotiations, and knowledge of \
Diplomacy strategy to make your predictions."""

PROBE_A_TASK_TEMPLATE = """\
== TASK ==
Each round in this game, every player writes a private diary entry reflecting their \
strategic thinking, then submits movement orders simultaneously. Diary entries typically contain:
- Assessment of the current board position
- Analysis of negotiations and trust levels with other powers
- Strategic intentions and planned moves
- Concerns about potential betrayals or opportunities

You know that {target_power} has also had private conversations with other players \
that you cannot see.

Based on everything available to you as {predictor_power}, predict:
1. The contents of {target_power}'s diary entry for this phase (what are they thinking?)
2. {target_power}'s strategic goals (a list of their current objectives)
3. {target_power}'s relationships with each other power (Enemy, Unfriendly, Neutral, Friendly, or Ally)
4. {target_power}'s likely submitted orders for this phase

{target_power}'s units and legal orders are listed below for reference:
{target_possible_orders}

Respond in JSON format:
{{
  "reasoning": "Your chain-of-thought analysis of what {target_power} is likely thinking, \
based on the board state, negotiations, and strategic position. Consider their likely \
allies, enemies, fears, and opportunities.",
  "predicted_diary": "Your prediction of {target_power}'s private diary entry for this phase. \
Write it as if you were {target_power} writing in first person.",
  "predicted_goals": ["goal 1", "goal 2", "..."],
  "predicted_relationships": {{
    "POWER_NAME": "Enemy|Unfriendly|Neutral|Friendly|Ally"
  }},
  "predicted_orders": ["order1", "order2", "..."]
}}"""

PROBE_B_TASK_TEMPLATE = """\
== TASK ==
Each round in this game, every player submits movement orders simultaneously.

You know that {target_power} has also had private conversations with other players \
that you cannot see.

Based on everything available to you as {predictor_power}, predict {target_power}'s \
likely submitted orders for this phase.

{target_power}'s units and legal orders are listed below for reference:
{target_possible_orders}

Respond in JSON format:
{{
  "reasoning": "Your chain-of-thought analysis of what {target_power} is likely to do, \
based on the board state, negotiations, and strategic position.",
  "predicted_orders": ["order1", "order2", "..."]
}}"""


def load_game_json(game_file: str) -> dict:
    with open(game_file, "r") as f:
        return json.load(f)


def find_phase_index(phases: list, phase_name: str) -> int:
    for i, p in enumerate(phases):
        if p["name"] == phase_name:
            return i
    raise ValueError(f"Phase '{phase_name}' not found in game. Available phases: {[p['name'] for p in phases]}")


def format_visible_messages(messages: list) -> str:
    """Format messages into a readable negotiation log."""
    if not messages:
        return "(No messages involving your power this round.)"

    lines = []
    for msg in messages:
        recipient = msg["recipient"]
        if recipient == "GLOBAL":
            lines.append(f"[{msg['sender']} -> ALL]: {msg['content']}")
        else:
            lines.append(f"[{msg['sender']} -> {msg['recipient']}]: {msg['content']}")
    return "\n".join(lines)


def format_diary(diary_entries: list) -> str:
    """Format diary entries for inclusion in the prompt."""
    if not diary_entries:
        return "(No diary entries yet.)"
    return "\n\n".join(diary_entries)


def format_goals(goals: list) -> str:
    if not goals:
        return "None"
    return "\n".join(f"- {g}" for g in goals)


def format_relationships(relationships: dict) -> str:
    if not relationships:
        return "None"
    return "\n".join(f"- {power}: {level}" for power, level in relationships.items())


def format_possible_orders_simple(possible_orders: Dict[str, List[str]], power_name: str) -> str:
    """Format possible orders for the target power in a readable way."""
    if not possible_orders:
        return f"(No orderable units for {power_name}.)"

    lines = [f"{power_name}'s orderable units and legal moves:"]
    for loc, orders in sorted(possible_orders.items()):
        lines.append(f"  Unit at {loc}:")
        for order in orders:
            lines.append(f"    - {order}")
    return "\n".join(lines)


def extract_predictor_context(game_data: dict, phase_name: str, predictor_power: str) -> dict:
    """
    Extract what the predictor power could see at diary-generation time.

    At diary-generation time, the agent has:
    - Board state (all units + supply centers) — public knowledge
    - Their goals and relationships from the END of the previous phase
    - Their accumulated diary from the END of the previous phase
    - All messages visible to them from the CURRENT phase's negotiations
    """
    phases = game_data["phases"]
    phase_idx = find_phase_index(phases, phase_name)
    current_phase = phases[phase_idx]

    # Agent state from PREVIOUS phase (before this phase's diary/state updates)
    if phase_idx > 0:
        prev_agent_state = phases[phase_idx - 1]["state_agents"][predictor_power]
    else:
        # First phase — use current phase state (goals/relationships were initialized before negotiations)
        prev_agent_state = current_phase["state_agents"][predictor_power]

    # Board state from current phase (unit positions and supply centers at start of phase)
    board_state = current_phase["state"]

    # Messages visible to predictor: global + private where predictor is sender or recipient
    visible_messages = []
    for msg in current_phase.get("messages", []):
        if (msg.get("recipient") == "GLOBAL"
                or msg.get("sender") == predictor_power
                or msg.get("recipient") == predictor_power):
            visible_messages.append({
                "sender": msg["sender"],
                "recipient": msg["recipient"],
                "content": msg["message"],
            })

    return {
        "unit_positions": board_state["units"],
        "supply_centers": board_state["centers"],
        "visible_messages": visible_messages,
        "predictor_diary": prev_agent_state.get("private_diary", []),
        "predictor_goals": prev_agent_state.get("goals", []),
        "predictor_relationships": prev_agent_state.get("relationships", {}),
    }


def extract_ground_truth(game_data: dict, phase_name: str, target_power: str) -> dict:
    """
    Extract the target power's actual diary entry and orders for this phase.

    Ground truth comes from:
    - Diary: entries tagged with this phase from the target's full_private_diary
      (saved in this phase's state_agents, which reflects end-of-phase state)
    - Orders: the actual orders submitted in this phase
    - Goals/relationships: from this phase's state_agents (post-diary-update)
    """
    phases = game_data["phases"]
    phase_idx = find_phase_index(phases, phase_name)
    current_phase = phases[phase_idx]

    # Target's submitted orders
    target_orders = current_phase.get("orders", {}).get(target_power, [])

    # Target's state at end of this phase
    target_agent = current_phase.get("state_agents", {}).get(target_power, {})
    full_diary = target_agent.get("full_private_diary", [])

    # Extract only diary entries from this specific phase
    phase_diary_entries = [e for e in full_diary if e.startswith(f"[{phase_name}]")]

    # Also get pre-phase state for comparison
    if phase_idx > 0:
        prev_agent = phases[phase_idx - 1]["state_agents"].get(target_power, {})
    else:
        prev_agent = target_agent

    return {
        "target_diary_entries_this_phase": phase_diary_entries,
        "target_full_diary": full_diary,
        "target_orders": target_orders,
        "target_goals_post_phase": target_agent.get("goals", []),
        "target_relationships_post_phase": target_agent.get("relationships", {}),
        "target_goals_pre_phase": prev_agent.get("goals", []),
        "target_relationships_pre_phase": prev_agent.get("relationships", {}),
    }


def format_board_state(unit_positions: dict, supply_centers: dict) -> str:
    """Format unit positions and supply centers for prompt inclusion."""
    lines = ["UNIT POSITIONS:"]
    for power in ALL_POWERS:
        units = unit_positions.get(power, [])
        if units:
            lines.append(f"  {power}: {', '.join(units)}")
        else:
            lines.append(f"  {power}: [ELIMINATED]")

    lines.append("\nSUPPLY CENTERS:")
    for power in ALL_POWERS:
        centers = supply_centers.get(power, [])
        if centers:
            lines.append(f"  {power} ({len(centers)}): {', '.join(sorted(centers))}")
        else:
            lines.append(f"  {power}: [NONE]")

    return "\n".join(lines)


def build_probe_prompt(
    predictor_power: str,
    target_power: str,
    phase_name: str,
    context: dict,
    target_possible_orders: Dict[str, List[str]],
    probe_type: str,
) -> str:
    """Assemble the full probe prompt from predictor context."""

    board_state_str = format_board_state(context["unit_positions"], context["supply_centers"])
    messages_str = format_visible_messages(context["visible_messages"])
    diary_str = format_diary(context["predictor_diary"])
    goals_str = format_goals(context["predictor_goals"])
    relationships_str = format_relationships(context["predictor_relationships"])
    target_orders_str = format_possible_orders_simple(target_possible_orders, target_power)

    prompt_parts = [
        f"You are playing as {predictor_power} in a game of Diplomacy.",
        f"Current phase: {phase_name}",
        "",
        "== CURRENT GAME STATE ==",
        board_state_str,
        "",
        "== YOUR PRIVATE DIARY ==",
        diary_str,
        "",
        "== YOUR STRATEGIC STATE ==",
        f"Goals:\n{goals_str}",
        "",
        f"Relationships:\n{relationships_str}",
        "",
        "== YOUR NEGOTIATIONS THIS PHASE ==",
        messages_str,
        "",
    ]

    if probe_type == "A":
        task = PROBE_A_TASK_TEMPLATE.format(
            predictor_power=predictor_power,
            target_power=target_power,
            target_possible_orders=target_orders_str,
        )
    else:
        task = PROBE_B_TASK_TEMPLATE.format(
            predictor_power=predictor_power,
            target_power=target_power,
            target_possible_orders=target_orders_str,
        )

    prompt_parts.append(task)
    return "\n".join(prompt_parts)


async def run_single_prediction(
    client,
    prompt: str,
    model_name: str,
    run_index: int,
) -> dict:
    """Run a single LLM prediction and return the result."""
    try:
        raw_response = await client.generate_response(prompt, temperature=1.0)
        return {
            "model": model_name,
            "run_index": run_index,
            "raw_response": raw_response,
            "error": None,
        }
    except Exception as e:
        logger.error(f"Prediction failed for {model_name} run {run_index}: {e}")
        return {
            "model": model_name,
            "run_index": run_index,
            "raw_response": None,
            "error": str(e),
        }


async def run_predictions_for_model(
    model_id: str,
    num_predictions: int,
    prompt: str,
) -> List[dict]:
    """Run predictions sequentially for a single model to avoid rate limits."""
    client = load_model_client(model_id)
    client.system_prompt = PROBE_SYSTEM_PROMPT
    results = []
    for run_idx in range(num_predictions):
        result = await run_single_prediction(client, prompt, model_id, run_idx)
        results.append(result)
    return results


async def run_predictions(
    models: List[str],
    num_predictions: int,
    prompt: str,
) -> List[dict]:
    """Run predictions for all models. Sequential within each model, parallel across models."""
    model_tasks = [
        run_predictions_for_model(model_id, num_predictions, prompt)
        for model_id in models
    ]
    model_results = await asyncio.gather(*model_tasks)
    # Flatten: interleave results to maintain [model0_run0, model0_run1, ..., model1_run0, ...] order
    results = []
    for model_result in model_results:
        results.extend(model_result)
    return results


def get_game_object_at_phase(run_dir: str, phase_name: str):
    """Load a Game object at the specified phase for computing possible orders."""
    run_config = Namespace(
        max_tokens=16000,
        max_tokens_per_model=None,
        models=None,
        prompts_dir=None,
        prompts_dir_map=None,
    )
    game, _, _, _ = load_game_state(
        run_dir=run_dir,
        game_file_name="lmvsgame.json",
        run_config=run_config,
        resume_from_phase=phase_name,
    )
    return game


def list_phases(game_data: dict) -> None:
    """Print available phases for selection."""
    print("\nAvailable phases:")
    for i, phase in enumerate(game_data["phases"]):
        orders = phase.get("orders", {})
        msg_count = len(phase.get("messages", []))
        powers_with_orders = [p for p, o in orders.items() if o]
        print(f"  {phase['name']:8s}  |  {msg_count:3d} messages  |  {len(powers_with_orders)} powers ordered")
    print()


async def main():
    parser = argparse.ArgumentParser(description="Critical State Analysis — Theory of Mind Probes")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the game run directory containing lmvsgame.json")
    parser.add_argument("--phase", type=str, required=True, help="Phase to probe (e.g., S1902M) or 'list' to show available phases")
    parser.add_argument("--predictor_power", type=str, required=True, help="Power whose perspective the model adopts")
    parser.add_argument("--target_power", type=str, required=True, help="Power whose diary/moves to predict")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated list of model IDs for predictions")
    parser.add_argument("--num_predictions", type=int, default=3, help="Number of prediction runs per model (default: 3)")
    parser.add_argument("--probe_type", type=str, choices=["A", "B"], default="A", help="A = predict diary + moves, B = predict moves only (default: A)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: {run_dir}/tom_probes/)")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Load raw game data
    game_file = os.path.join(args.run_dir, "lmvsgame.json")
    if not os.path.exists(game_file):
        print(f"Error: Game file not found at {game_file}")
        return
    game_data = load_game_json(game_file)

    # Handle 'list' command
    if args.phase == "list":
        list_phases(game_data)
        return

    # Validate inputs
    predictor = args.predictor_power.upper()
    target = args.target_power.upper()
    if predictor not in ALL_POWERS:
        print(f"Error: Invalid predictor power '{predictor}'. Must be one of {ALL_POWERS}")
        return
    if target not in ALL_POWERS:
        print(f"Error: Invalid target power '{target}'. Must be one of {ALL_POWERS}")
        return
    if predictor == target:
        print("Error: Predictor and target powers must be different.")
        return

    models = [m.strip() for m in args.models.split(",")]

    # Extract predictor's context
    print(f"Extracting {predictor}'s context at phase {args.phase}...")
    context = extract_predictor_context(game_data, args.phase, predictor)

    # Get target's possible orders via the diplomacy engine
    print(f"Computing {target}'s possible orders...")
    game = get_game_object_at_phase(args.run_dir, args.phase)
    target_possible_orders = gather_possible_orders(game, target)

    # Extract ground truth
    print(f"Extracting {target}'s ground truth...")
    ground_truth = extract_ground_truth(game_data, args.phase, target)

    # Build the probe prompt
    prompt = build_probe_prompt(
        predictor_power=predictor,
        target_power=target,
        phase_name=args.phase,
        context=context,
        target_possible_orders=target_possible_orders,
        probe_type=args.probe_type,
    )

    print(f"\nPrompt length: {len(prompt)} chars")
    print(f"Running {len(models)} model(s) x {args.num_predictions} predictions each...")
    print(f"Models: {models}")

    # Run predictions
    predictions = await run_predictions(models, args.num_predictions, prompt)

    # Count successes
    successes = sum(1 for p in predictions if p["error"] is None)
    print(f"Completed: {successes}/{len(predictions)} predictions succeeded.")

    # Assemble output
    timestamp = datetime.now(timezone.utc).isoformat()
    output = {
        "metadata": {
            "game_file": os.path.abspath(game_file),
            "run_dir": os.path.abspath(args.run_dir),
            "phase": args.phase,
            "predictor_power": predictor,
            "target_power": target,
            "probe_type": args.probe_type,
            "models": models,
            "num_predictions": args.num_predictions,
            "timestamp": timestamp,
        },
        "prompt_used": prompt,
        "context_provided": {
            "unit_positions": context["unit_positions"],
            "supply_centers": context["supply_centers"],
            "visible_messages": context["visible_messages"],
            "predictor_diary": context["predictor_diary"],
            "predictor_goals": context["predictor_goals"],
            "predictor_relationships": context["predictor_relationships"],
        },
        "ground_truth": ground_truth,
        "predictions": predictions,
    }

    # Write output
    output_dir = args.output_dir or os.path.join(args.run_dir, "tom_probes")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"probe_{args.probe_type}_{predictor}_predicts_{target}_{args.phase}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")
    print(f"Ground truth orders: {ground_truth['target_orders']}")
    print(f"Ground truth diary entries: {len(ground_truth['target_diary_entries_this_phase'])} entries for this phase")


if __name__ == "__main__":
    asyncio.run(main())
