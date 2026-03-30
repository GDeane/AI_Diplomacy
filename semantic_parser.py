"""
Semantic Parser — Extract BDI-JSON from probe results for quantitative comparison.

Takes a probe output JSON file, extracts a standardized BDI-JSON representation
from both ground truth and each prediction, then saves the parsed output.

BDI-JSON Schema:
{
  "relationships": {"POWER": "Enemy|Unfriendly|Neutral|Friendly|Ally", ...},
  "agreements": [
    {"counterparty": "POWER", "type": "dmz|bounce|split|support|move|non_aggression|convoy", "region": "LOC"}
  ],
  "intended_orders": ["A BUD - SER", ...]
}

Usage:
    python semantic_parser.py \
        --probe_file results/my_test_run_13/tom_probes/probe_A_TURKEY_predicts_AUSTRIA_S1902M_20260313_160223.json \
        --parser_model gemini-2.5-flash

    # Process all probes in a directory
    python semantic_parser.py \
        --probe_dir results/my_test_run_13/tom_probes/ \
        --parser_model gemini-2.5-flash
"""

import argparse
import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from glob import glob
from typing import Dict, List, Optional

from ai_diplomacy.clients import load_model_client

logger = logging.getLogger(__name__)

ALL_POWERS = ["AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"]

AGREEMENT_TYPES = ["dmz", "bounce", "split", "support", "move", "non_aggression", "convoy"]

AGREEMENT_EXTRACTION_PROMPT = """\
You are a structured information extractor for the board game Diplomacy.

Given a diary entry or negotiation summary written by a player, extract all \
agreements, deals, or arrangements the author describes having with other powers.

Agreement types:
- "dmz": Agreement to keep a region demilitarized (neither side moves in)
- "bounce": Agreement to intentionally bounce/collide in a region
- "split": Agreement to divide regions between two powers
- "support": Agreement where one power supports another's move
- "move": Agreement about where a power will move a specific unit
- "non_aggression": General agreement not to attack each other
- "convoy": Agreement to convoy a unit

For each agreement, extract:
- "counterparty": The other power involved (e.g., "RUSSIA", "ITALY")
- "type": One of the types listed above
- "region": The specific region/province involved (use standard 3-letter Diplomacy abbreviations \
like GAL, BLA, TYR, BUL, RUM, etc.). For "split" type, use a comma-separated string of regions.

IMPORTANT:
- Only extract agreements that the author explicitly describes or references.
- Do NOT infer agreements that are not stated.
- If no agreements are mentioned, return an empty list.
- Use uppercase 3-letter abbreviations for regions (e.g., GAL not Galicia, BLA not Black Sea).
- Use uppercase for power names (e.g., RUSSIA not Russia).

Respond with ONLY a JSON array of agreement objects:
[
  {{"counterparty": "POWER", "type": "dmz|bounce|split|support|move|non_aggression|convoy", "region": "LOC"}},
  ...
]

If no agreements are found, respond with: []

TEXT TO PARSE:
{text}"""

RELATIONSHIP_EXTRACTION_PROMPT = """\
You are a structured information extractor for the board game Diplomacy.

Given a diary entry or negotiation summary written by a player ({author_power}), \
extract how the author perceives their relationship with each other power.

Use ONLY these relationship levels:
- "Enemy": Actively hostile, planning attacks against
- "Unfriendly": Distrustful, wary, potential adversary
- "Neutral": No strong feelings either way
- "Friendly": Positive relations, some cooperation
- "Ally": Strong alliance, active coordination

IMPORTANT:
- Base your assessment on what the text explicitly states or strongly implies.
- If a power is not mentioned at all, assign "Neutral".
- Consider the overall tone and specific language used about each power.

Respond with ONLY a JSON object mapping each power to a relationship level:
{{
  "AUSTRIA": "Neutral",
  "ENGLAND": "Neutral",
  "FRANCE": "Neutral",
  "GERMANY": "Neutral",
  "ITALY": "Neutral",
  "RUSSIA": "Neutral",
  "TURKEY": "Neutral"
}}

(Exclude {author_power} from the output — do not include the author's relationship with themselves.)

TEXT TO PARSE:
{text}"""


def parse_json_from_response(raw: str) -> Optional[dict | list]:
    """Try to extract JSON from an LLM response with multiple fallback strategies."""
    if not raw:
        return None

    # Try direct parse
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting JSON block from markdown
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding first { or [ to last } or ]
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = raw.find(start_char)
        end = raw.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass

    # Last resort: try json_repair if available
    try:
        import json_repair
        return json_repair.loads(raw)
    except Exception:
        pass

    return None


async def extract_agreements(client, text: str) -> List[dict]:
    """Use LLM to extract agreements from diary/summary text."""
    if not text or not text.strip():
        return []

    prompt = AGREEMENT_EXTRACTION_PROMPT.format(text=text)
    try:
        response = await client.generate_response(prompt, temperature=0.0)
        parsed = parse_json_from_response(response)
        if isinstance(parsed, list):
            # Validate and normalize agreement objects
            valid = []
            for a in parsed:
                if isinstance(a, dict) and "counterparty" in a and "type" in a:
                    agreement = {
                        "counterparty": a["counterparty"].upper(),
                        "type": a["type"].lower(),
                        "region": a.get("region", "").upper() if a.get("region") else "",
                    }
                    if agreement["type"] not in AGREEMENT_TYPES:
                        logger.warning(f"Unknown agreement type '{agreement['type']}', keeping as-is")
                    valid.append(agreement)
            return valid
        else:
            logger.warning(f"Agreement extraction returned non-list: {type(parsed)}")
            return []
    except Exception as e:
        logger.error(f"Agreement extraction failed: {e}")
        return []


async def extract_relationships(client, text: str, author_power: str) -> Dict[str, str]:
    """Use LLM to extract perceived relationships from diary/summary text."""
    if not text or not text.strip():
        return {p: "Neutral" for p in ALL_POWERS if p != author_power}

    prompt = RELATIONSHIP_EXTRACTION_PROMPT.format(text=text, author_power=author_power)
    try:
        response = await client.generate_response(prompt, temperature=0.0)
        parsed = parse_json_from_response(response)
        if isinstance(parsed, dict):
            # Normalize and validate
            valid_levels = {"Enemy", "Unfriendly", "Neutral", "Friendly", "Ally"}
            result = {}
            for power in ALL_POWERS:
                if power == author_power:
                    continue
                level = parsed.get(power, "Neutral")
                if level not in valid_levels:
                    logger.warning(f"Invalid relationship level '{level}' for {power}, defaulting to Neutral")
                    level = "Neutral"
                result[power] = level
            return result
        else:
            logger.warning(f"Relationship extraction returned non-dict: {type(parsed)}")
            return {p: "Neutral" for p in ALL_POWERS if p != author_power}
    except Exception as e:
        logger.error(f"Relationship extraction failed: {e}")
        return {p: "Neutral" for p in ALL_POWERS if p != author_power}


async def parse_ground_truth(client, probe_data: dict) -> dict:
    """
    Extract BDI-JSON from ground truth data.

    Relationships come from post-phase state (after the target's negotiation diary
    entry, where it formulates intent and updates relationships for this phase).
    Orders come from the structured field.
    Agreements are extracted from diary text via LLM.
    """
    gt = probe_data["ground_truth"]
    target_power = probe_data["metadata"]["target_power"]

    # Relationships: use post-phase state, which reflects the target's updated stance
    # after writing its negotiation diary (the same step where it decides on orders).
    # Pre-phase relationships are stale carryover from the previous phase and may not
    # reflect the target's actual intent (e.g., still "Ally" while planning a stab).
    relationships = gt.get("target_relationships_post_phase", {})

    # Orders: use actual submitted orders directly
    intended_orders = gt.get("target_orders", [])

    # Agreements: extract from diary text via LLM
    diary_entries = gt.get("target_diary_entries_this_phase", [])
    diary_text = "\n\n".join(diary_entries) if diary_entries else ""
    agreements = await extract_agreements(client, diary_text)

    return {
        "relationships": relationships,
        "agreements": agreements,
        "intended_orders": intended_orders,
    }


def parse_prediction_structured_fields(raw_response: str) -> dict:
    """Extract structured fields from a prediction's raw LLM response."""
    parsed = parse_json_from_response(raw_response)
    if not isinstance(parsed, dict):
        return {
            "predicted_diary": "",
            "predicted_orders": [],
            "predicted_relationships": None,
            "predicted_goals": None,
            "reasoning": "",
        }
    return {
        "predicted_diary": parsed.get("predicted_diary", ""),
        "predicted_orders": parsed.get("predicted_orders", []),
        "predicted_relationships": parsed.get("predicted_relationships"),
        "predicted_goals": parsed.get("predicted_goals"),
        "reasoning": parsed.get("reasoning", ""),
    }


async def parse_single_prediction(client, prediction: dict, target_power: str) -> dict:
    """
    Extract BDI-JSON from a single prediction.

    If the prediction already has structured relationships, use those directly.
    Otherwise, extract from the predicted diary text via LLM.
    Orders come from the predicted_orders field.
    Agreements are always extracted from predicted diary text via LLM.
    """
    if prediction.get("error"):
        return {
            "model": prediction["model"],
            "run_index": prediction["run_index"],
            "parse_error": f"Prediction had error: {prediction['error']}",
            "bdi": None,
        }

    fields = parse_prediction_structured_fields(prediction["raw_response"])

    # Relationships: use structured field if available, otherwise extract from diary
    if fields["predicted_relationships"] and isinstance(fields["predicted_relationships"], dict):
        relationships = fields["predicted_relationships"]
    else:
        relationships = await extract_relationships(client, fields["predicted_diary"], target_power)

    # Orders: use structured field directly
    intended_orders = fields["predicted_orders"] if isinstance(fields["predicted_orders"], list) else []

    # Agreements: always extract from diary text via LLM
    agreements = await extract_agreements(client, fields["predicted_diary"])

    return {
        "model": prediction["model"],
        "run_index": prediction["run_index"],
        "parse_error": None,
        "bdi": {
            "relationships": relationships,
            "agreements": agreements,
            "intended_orders": intended_orders,
        },
    }


async def process_probe_file(probe_file: str, parser_model: str) -> dict:
    """Process a single probe file: parse ground truth and all predictions into BDI-JSON."""
    logger.info(f"Processing: {probe_file}")

    with open(probe_file, "r") as f:
        probe_data = json.load(f)

    client = load_model_client(parser_model)
    client.system_prompt = "You are a precise structured information extractor. Always respond with valid JSON only."

    target_power = probe_data["metadata"]["target_power"]

    # Parse ground truth
    logger.info("Parsing ground truth...")
    gt_bdi = await parse_ground_truth(client, probe_data)

    # Parse all predictions in parallel
    logger.info(f"Parsing {len(probe_data['predictions'])} predictions...")
    prediction_tasks = [
        parse_single_prediction(client, pred, target_power)
        for pred in probe_data["predictions"]
    ]
    parsed_predictions = await asyncio.gather(*prediction_tasks)

    # Count successes
    successes = sum(1 for p in parsed_predictions if p["bdi"] is not None)
    logger.info(f"Parsed {successes}/{len(parsed_predictions)} predictions successfully.")

    result = {
        "metadata": {
            **probe_data["metadata"],
            "parser_model": parser_model,
            "parsed_at": datetime.now(timezone.utc).isoformat(),
        },
        "ground_truth_bdi": gt_bdi,
        "parsed_predictions": parsed_predictions,
    }

    return result


async def main():
    parser = argparse.ArgumentParser(description="Semantic Parser — Extract BDI-JSON from probe results")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--probe_file", type=str, help="Path to a single probe output JSON file")
    group.add_argument("--probe_dir", type=str, help="Path to directory containing probe JSON files")
    parser.add_argument("--parser_model", type=str, default="gemini-2.5-flash", help="Model to use for text parsing (default: gemini-2.5-flash)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: same directory as input, with _bdi suffix)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Collect probe files to process
    if args.probe_file:
        probe_files = [args.probe_file]
    else:
        probe_files = sorted(glob(os.path.join(args.probe_dir, "probe_*.json")))
        # Exclude already-parsed BDI files
        probe_files = [f for f in probe_files if "_bdi" not in os.path.basename(f)]
        if not probe_files:
            print(f"No probe files found in {args.probe_dir}")
            return

    print(f"Found {len(probe_files)} probe file(s) to process.")

    for probe_file in probe_files:
        result = await process_probe_file(probe_file, args.parser_model)

        # Determine output path
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(probe_file))[0]
            output_path = os.path.join(args.output_dir, f"{base}_bdi.json")
        else:
            output_path = probe_file.replace(".json", "_bdi.json")

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"Saved: {output_path}")

        # Print summary
        gt = result["ground_truth_bdi"]
        print(f"  Ground truth: {len(gt['agreements'])} agreements, {len(gt['intended_orders'])} orders")
        for p in result["parsed_predictions"]:
            if p["bdi"]:
                print(f"  Prediction [{p['model']} #{p['run_index']}]: "
                      f"{len(p['bdi']['agreements'])} agreements, {len(p['bdi']['intended_orders'])} orders")
            else:
                print(f"  Prediction [{p['model']} #{p['run_index']}]: PARSE FAILED — {p['parse_error']}")


if __name__ == "__main__":
    asyncio.run(main())
