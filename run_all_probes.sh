#!/bin/bash
# Run all ToM probes: 5 predictions × 3 models × 2 probe types × 5 situations = 150 predictions
# Each model runs sequentially within critical_state_analysis.py (to avoid 429s),
# but different models for the same situation run in parallel.

set -e

# Model mapping: short name -> full model ID
declare -A MODELS
MODELS[gemini]="gemini-3.1-pro-preview"
MODELS[claude]="openrouter-anthropic/claude-opus-4.6"
MODELS[gpt]="openrouter-openai/gpt-5.4"

N=5  # predictions per model

# Define situations: run_dir|phase|predictor|target
SITUATIONS=(
    "results/my_test_run_4|F1901M|AUSTRIA|ITALY"
    "results/my_test_run_14|F1903M|AUSTRIA|ITALY"
    "results/my_test_run_14|S1902M|ENGLAND|FRANCE"
    "results/my_test_run_14|S1902M|TURKEY|AUSTRIA"
    "results/my_test_run_14|S1904M|ITALY|AUSTRIA"
)

run_probe() {
    local run_dir="$1" phase="$2" predictor="$3" target="$4" probe_type="$5" model_short="$6"
    local model_id="${MODELS[$model_short]}"
    local output_dir="${run_dir}/tom_probes/probe_${probe_type}/${model_short}"

    echo "[START] ${probe_type} | ${model_short} | ${predictor}->${target} @ ${phase} (${run_dir})"
    python critical_state_analysis.py \
        --run_dir "$run_dir" \
        --phase "$phase" \
        --predictor_power "$predictor" \
        --target_power "$target" \
        --models "$model_id" \
        --num_predictions "$N" \
        --probe_type "$probe_type" \
        --output_dir "$output_dir" \
        2>&1 | tail -3
    echo "[DONE]  ${probe_type} | ${model_short} | ${predictor}->${target} @ ${phase}"
}

export -f run_probe

# Run each situation
for sit in "${SITUATIONS[@]}"; do
    IFS='|' read -r run_dir phase predictor target <<< "$sit"
    echo ""
    echo "========================================"
    echo "  ${predictor} predicts ${target} @ ${phase} (${run_dir})"
    echo "========================================"

    for probe_type in A B; do
        echo ""
        echo "--- Probe ${probe_type} ---"
        # Run 3 models in parallel, each doing N sequential predictions
        for model_short in gemini claude gpt; do
            run_probe "$run_dir" "$phase" "$predictor" "$target" "$probe_type" "$model_short" &
        done
        wait  # Wait for all 3 models to finish before next probe type
        echo "--- Probe ${probe_type} complete ---"
    done
done

echo ""
echo "========================================"
echo "  All probes complete!"
echo "========================================"
