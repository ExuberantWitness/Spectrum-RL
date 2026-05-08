#!/bin/bash
# Experiment grid for Integrated EW HRL
# Usage: bash artifacts/run_experiments.sh

set -e

SEEDS=(42 123 456 789 1024)
STEPS=200000
RESULTS_DIR="artifacts/results"
MODEL_DIR="artifacts/models"

echo "================================================"
echo "Integrated EW HRL Experiment Grid"
echo "Seeds: ${SEEDS[@]}"
echo "Steps per run: $STEPS"
echo "================================================"

mkdir -p "$RESULTS_DIR" "$MODEL_DIR"

# Run multi-seed HRL training
for seed in "${SEEDS[@]}"; do
    echo ""
    echo "--- Running HRL with seed=$seed ---"
    python -m artifacts.main \
        --mode full_experiment \
        --seed $seed \
        --total_steps $STEPS \
        --results_dir "${RESULTS_DIR}/seed_${seed}" \
        --model_dir "${MODEL_DIR}/seed_${seed}"
done

echo ""
echo "================================================"
echo "Aggregating results across seeds..."
python -c "
import json, os, numpy as np

all_results = {}
for seed in [42, 123, 456, 789, 1024]:
    path = f'artifacts/results/seed_{seed}/hrl_results.json'
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
            # Get final evaluation metrics
            if 'metrics_history' in data:
                final = data['metrics_history'][-1]
                all_results[f'seed_{seed}'] = final.get('eval_reward_mean', 0)

if all_results:
    values = list(all_results.values())
    print(f'Multi-seed results:')
    print(f'  Mean: {np.mean(values):.3f}')
    print(f'  Std:  {np.std(values):.3f}')
    print(f'  Min:  {np.min(values):.3f}')
    print(f'  Max:  {np.max(values):.3f}')

    summary = {
        'seeds': all_results,
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
    }
    with open('artifacts/results/multi_seed_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('Summary saved to artifacts/results/multi_seed_summary.json')
"
echo "Done!"
