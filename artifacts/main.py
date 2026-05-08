"""Main entry point for Integrated EW HRL experiments.

Usage:
    python -m artifacts.main --mode train --seed 42
    python -m artifacts.main --mode evaluate --checkpoint models/hrl_step200000.pt
    python -m artifacts.main --mode baselines
    python -m artifacts.main --mode full_experiment
"""

import argparse
import sys
import os
import numpy as np
import torch
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ExperimentConfig
from env import IntegratedRadarEnv
from training import HRLTrainer, CurriculumScheduler
from baselines import IndependentRLAgent, FixedPolicyAgent, RandomBaseline
from eval import Evaluator, Visualizer


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_hrl(config: ExperimentConfig) -> dict:
    """Train the three-layer HRL agent."""
    set_seed(config.seed)
    trainer = HRLTrainer(config)
    results = trainer.train()
    trainer.save_results()
    trainer.save_checkpoint()
    return results


def evaluate_hrl(config: ExperimentConfig, checkpoint_path: str) -> dict:
    """Evaluate a trained HRL agent."""
    trainer = HRLTrainer(config)

    ckpt = torch.load(checkpoint_path, map_location=trainer.device)
    trainer.strategic.load_state_dict(ckpt["strategic"])
    trainer.tactical_ppo.policy.load_state_dict(ckpt["tactical_policy"])
    trainer.executive_ppo.policy.load_state_dict(ckpt["executive_policy"])

    metrics = trainer.evaluate()
    print(f"\nEvaluation Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics


def run_baselines(config: ExperimentConfig) -> dict:
    """Run all baseline methods."""
    set_seed(config.seed)
    results = {}

    print("\n" + "=" * 60)
    print("Running Random Baseline")
    random_agent = RandomBaseline(config)
    results["Random"] = random_agent.evaluate(num_episodes=50)
    print(f"  Random: {results['Random']['mean_reward']:.4f}")

    print("\n" + "=" * 60)
    print("Running Fixed Policy Baselines")
    fixed_agent = FixedPolicyAgent(config)
    for mode in ["uniform", "balanced", "detect_heavy", "jam_heavy"]:
        results[f"Fixed-{mode}"] = fixed_agent.evaluate(num_episodes=50, mode=mode)
        print(f"  Fixed-{mode}: {results[f'Fixed-{mode}']['mean_reward']:.4f}")

    print("\n" + "=" * 60)
    print("Running Independent RL Baseline")
    ind_rl = IndependentRLAgent(config)
    ind_results = ind_rl.train(total_steps=100_000)
    results["IndependentRL"] = {
        "mean_reward": ind_results["metrics"][-1]["mean_reward"]
        if ind_results["metrics"] else 0,
    }
    print(f"  IndependentRL: {results['IndependentRL']['mean_reward']:.4f}")

    return results


def full_experiment(config: ExperimentConfig) -> dict:
    """Run the complete experiment pipeline."""
    print("\n" + "=" * 70)
    print("FULL EXPERIMENT: Three-Layer HRL for Integrated EW")
    print("  Detection (探) + Reconnaissance (侦) + Jamming (干) + Communication (通)")
    print("=" * 70)

    # 1. Run baselines
    print("\n", "=" * 60)
    print("PHASE 1: Baselines")
    baseline_results = run_baselines(config)
    Evaluator.save_results(baseline_results,
                           os.path.join(config.results_dir, "baselines.json"))

    # 2. Train HRL
    print("\n", "=" * 60)
    print("PHASE 2: HRL Training with Curriculum Learning")
    hrl_results = train_hrl(config)
    Evaluator.save_results(hrl_results,
                           os.path.join(config.results_dir, "hrl_results.json"))

    # 3. Compare and visualize
    print("\n", "=" * 60)
    print("PHASE 3: Analysis & Visualization")
    viz = Visualizer(config.results_dir)

    # Prepare comparison data
    all_rewards = {}
    for name, result in baseline_results.items():
        all_rewards[name] = result.get("mean_reward", 0)

    if hrl_results.get("metrics_history"):
        final_eval = hrl_results["metrics_history"][-1]
        all_rewards["HRL (Ours)"] = final_eval.get("eval_reward_mean",
            np.mean([m["eval_reward_mean"] for m in hrl_results["metrics_history"][-3:]]))

    # Generate visualizations
    viz.plot_comparison_bars(all_rewards, save_path=os.path.join(
        config.results_dir, "comparison.png"))
    viz.plot_training_curve(
        hrl_results.get("metrics_history", []),
        save_path=os.path.join(config.results_dir, "training_curve.png"),
    )
    viz.generate_report(
        {"eval_reward_mean": all_rewards.get("HRL (Ours)", 0)},
        {k: {"mean_reward": v} for k, v in all_rewards.items() if k != "HRL (Ours)"},
    )

    # 4. Ablation study
    print("\n", "=" * 60)
    print("PHASE 4: Ablation Study")
    ablation = run_ablation(config)
    Evaluator.save_results(ablation,
                           os.path.join(config.results_dir, "ablation.json"))
    viz.plot_ablation(ablation, save_path=os.path.join(
        config.results_dir, "ablation.png"))

    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {config.results_dir}")
    print(f"Models saved to: {config.model_dir}")
    print("=" * 70)

    return {
        "baselines": baseline_results,
        "hrl": hrl_results,
        "ablation": ablation,
        "comparison": all_rewards,
    }


def run_ablation(config: ExperimentConfig) -> dict:
    """Run ablation study: remove each HRL layer and measure impact."""
    results = {}

    # Full HRL (reference)
    set_seed(config.seed)
    trainer_full = HRLTrainer(config)
    full_results = trainer_full.train()
    results["Full HRL"] = full_results["metrics_history"][-1]["eval_reward_mean"]

    # Without strategic layer (flat SAC)
    print("  Running: No Strategic Layer (flat SAC)...")
    from baselines.independent_rl import IndependentRLAgent
    agent = IndependentRLAgent(config)
    flat_results = agent.train(total_steps=config.total_steps // 2)
    results["No Strategic Layer"] = flat_results["metrics"][-1]["mean_reward"]

    # Without curriculum learning
    print("  Running: No Curriculum (random init)...")
    config_no_cl = ExperimentConfig()
    config_no_cl.curriculum.steps_per_stage = config.total_steps
    set_seed(config.seed)
    trainer_nocl = HRLTrainer(config_no_cl)
    nocl_results = trainer_nocl.train()
    results["No Curriculum"] = nocl_results["metrics_history"][-1]["eval_reward_mean"]

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Three-Layer HRL for Integrated EW (探侦干通一体化)")
    parser.add_argument("--mode", type=str, default="train",
                        choices=["train", "evaluate", "baselines", "full_experiment"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_steps", type=int, default=200_000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results_dir", type=str, default="artifacts/results")
    parser.add_argument("--model_dir", type=str, default="artifacts/models")

    args = parser.parse_args()

    config = ExperimentConfig()
    config.seed = args.seed
    config.total_steps = args.total_steps
    config.device = args.device if torch.cuda.is_available() else "cpu"
    config.results_dir = args.results_dir
    config.model_dir = args.model_dir

    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)

    print(f"Experiment Configuration:")
    print(f"  Mode: {args.mode}")
    print(f"  Seed: {config.seed}")
    print(f"  Total Steps: {config.total_steps}")
    print(f"  Device: {config.device}")
    print(f"  Results: {config.results_dir}")
    print()

    if args.mode == "train":
        train_hrl(config)
    elif args.mode == "evaluate":
        if not args.checkpoint:
            print("Error: --checkpoint required for evaluate mode")
            sys.exit(1)
        evaluate_hrl(config, args.checkpoint)
    elif args.mode == "baselines":
        run_baselines(config)
    elif args.mode == "full_experiment":
        result = full_experiment(config)


if __name__ == "__main__":
    main()
