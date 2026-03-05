# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI Diplomacy extends the open-source [Diplomacy](https://github.com/diplomacy/diplomacy) board game engine with LLM-powered agents. Each of the 7 powers (AUSTRIA, ENGLAND, FRANCE, GERMANY, ITALY, RUSSIA, TURKEY) is controlled by an autonomous AI agent that negotiates, plans, and submits orders via LLM calls.

## Setup and Dependencies

- **Python 3.13+** with [uv](https://github.com/astral-sh/uv) for dependency management
- Install: `uv sync` then `source .venv/bin/activate`
- API keys go in `.env` (OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY, OPENROUTER_API_KEY, TOGETHER_API_KEY)
- Ruff line-length: 150

## Common Commands

```bash
# Run a single game
python lm_game.py --max_year 1910 --num_negotiation_rounds 3 --models "gpt-4o"

# Run with simple prompts (default) and planning
python lm_game.py --max_year 1910 --planning_phase --num_negotiation_rounds 2

# Resume an interrupted game
python lm_game.py --run_dir results/game_run_001 --resume_from_phase S1902M

# Run batch experiments
python experiment_runner.py --experiment_dir results/exp001 --iterations 10 --parallel 10 --max_year 1905 --models "gpt-4o"

# Tests
pytest tests/                           # project-level tests
pytest diplomacy/tests/                 # diplomacy engine tests
pytest diplomacy/tests/test_game.py     # single test file

# Animation (separate TypeScript/Three.js project)
cd ai_animation && npm install && npm run dev    # dev server at localhost:5173
cd ai_animation && npm test                       # vitest unit tests
cd ai_animation && npm run test:e2e               # playwright e2e tests
```

## Architecture

### Core Game Loop (`lm_game.py`)

The main orchestrator runs an async loop over game phases:
1. **Negotiations** (movement phases only) - agents exchange messages via `conduct_negotiations()`
2. **Planning** (optional) - agents set strategic directives via `planning_phase()`
3. **Diary entries** - agents generate negotiation diary entries
4. **Order generation** - parallel async LLM calls via `get_valid_orders()`, with diary consolidation running concurrently
5. **Phase processing** - diplomacy engine resolves orders
6. **State updates** - agents update goals/relationships based on outcomes
7. **Save** - game state serialized to `lmvsgame.json` after each phase

### Key Modules (`ai_diplomacy/`)

| File | Purpose |
|---|---|
| `agent.py` | `DiplomacyAgent` class - goals, relationships, diary, LLM-driven state updates |
| `clients.py` | `BaseModelClient` + implementations for OpenAI, Anthropic, Gemini, DeepSeek, OpenRouter, Together |
| `prompt_constructor.py` | Assembles game state + agent state + strategic context into LLM prompts |
| `possible_order_context.py` | BFS pathfinding, threat/opportunity identification, XML context for orders |
| `game_history.py` | Phase-by-phase tracking of messages, orders, and results |
| `negotiations.py` | Multi-round negotiation orchestration |
| `planning.py` | Strategic planning phase |
| `diary_logic.py` | Diary consolidation (yearly summarization to prevent context overflow) |
| `formatter.py` | Two-step prompt formatting via Gemini Flash |
| `narrative.py` | Phase summary narrative generation |
| `game_logic.py` | Save/load game state, initialization |
| `initialization.py` | Agent creation and initial LLM calls for goals/relationships |
| `utils.py` | Order validation, prompt loading, LLM logging, model assignment |

### Prompt System

- Prompts live in `ai_diplomacy/prompts/` (full) and `ai_diplomacy/prompts_simple/` (default)
- Additional variants: `prompts_simple_optim_v1.0/`, `prompts/prompts_hold_reduction_v1/`, `prompts/prompts_benchmark/`
- Per-power system prompts (e.g., `france_system_prompt.txt`) and task-specific templates (e.g., `order_instructions_movement_phase.txt`, `conversation_instructions.txt`)
- `--simple_prompts` (default True) uses `prompts_simple/`; `--prompts_dir` overrides

### Model ID Syntax

```
<client_prefix:>model_name[@base_url][#api_key]
```
Prefixes: `openai`, `openai-requests`, `openai-responses`, `anthropic`, `gemini`, `deepseek`, `openrouter`, `together`

### Configuration (`config.py`)

Singleton `config` object (Pydantic `BaseSettings`). Loads `.env` automatically. API keys are validated at access time - warns on startup if missing, raises `ValueError` if used when empty.

### Diplomacy Engine (`diplomacy/`)

Forked from the open-source diplomacy project. Provides `Game` class, map, order processing, DAIDE protocol, client/server networking. Has its own test suite under `diplomacy/tests/`.

### Experiment Runner (`experiment_runner.py`)

Orchestrates parallel `lm_game.py` runs via process pool. Analysis modules in `experiment_runner/analysis/` (summary, critical_state, statistical_game_analysis, compare_stats).

### Analysis Pipeline (`analysis/`)

Separate data analysis modules for processing game results: `p1_make_longform_orders_data.py` -> `p2_make_convo_data.py` -> `p3_make_phase_data.py`, plus `statistical_game_analysis.py`.

### Animation (`ai_animation/`)

TypeScript + Three.js app for 3D game visualization. Has its own `CLAUDE.md` with detailed structure. Built with Vite, tested with Vitest + Playwright.

## Game Output Structure

Each game run (under `results/`) produces:
- `lmvsgame.json` - complete game data with phase summaries and agent states
- `llm_responses.csv` - all LLM interactions
- `overview.jsonl` - error stats, model assignments, run config
- `general_game.log` - execution logs

## Key Patterns

- All LLM calls are async; the codebase uses `asyncio.gather()` extensively for parallel execution
- JSON parsing from LLM responses uses multiple fallbacks: `json.loads` -> `json_repair` -> `json5` -> `ast.literal_eval` -> regex extraction
- Powers are always in alphabetical order: AUSTRIA, ENGLAND, FRANCE, GERMANY, ITALY, RUSSIA, TURKEY
- The `PowerEnum` in `models.py` handles typos/aliases via `_missing_` override
- Game state is saved after every phase for resume capability
