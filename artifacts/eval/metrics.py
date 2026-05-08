"""Evaluation metrics for EW HRL experiments."""

import numpy as np
import json
from typing import Dict, List, Tuple
from pathlib import Path


class Evaluator:
    """Compute and aggregate evaluation metrics across experiments."""

    @staticmethod
    def compute_detection_metrics(results: List[Dict]) -> Dict:
        pd_values = [r.get("avg_pd", 0) for r in results]
        return {
            "mean_pd": np.mean(pd_values),
            "std_pd": np.std(pd_values),
            "min_pd": np.min(pd_values),
            "max_pd": np.max(pd_values),
        }

    @staticmethod
    def compute_recon_metrics(results: List[Dict]) -> Dict:
        coverage = [r.get("coverage", 0) for r in results]
        precision = [r.get("precision", 0) for r in results]
        f1 = [r.get("f1", 0) for r in results]
        return {
            "mean_coverage": np.mean(coverage),
            "std_coverage": np.std(coverage),
            "mean_precision": np.mean(precision),
            "mean_f1": np.mean(f1),
        }

    @staticmethod
    def compute_jamming_metrics(results: List[Dict]) -> Dict:
        effectiveness = [r.get("effectiveness", 0) for r in results]
        jsr = [r.get("avg_jsr_db", -100) for r in results]
        return {
            "mean_effectiveness": np.mean(effectiveness),
            "std_effectiveness": np.std(effectiveness),
            "mean_jsr_db": np.mean(jsr),
        }

    @staticmethod
    def compute_comm_metrics(results: List[Dict]) -> Dict:
        rates = [r.get("rate", 0) for r in results]
        sinr = [r.get("sinr_db", -100) for r in results]
        return {
            "mean_rate_mbps": np.mean(rates),
            "std_rate_mbps": np.std(rates),
            "mean_sinr_db": np.mean(sinr),
        }

    @staticmethod
    def compute_resource_efficiency(results: List[Dict]) -> Dict:
        """Compute six-dimensional resource utilization."""
        freq_usage = [r.get("freq_usage", 0) for r in results]
        power_usage = [r.get("power_usage", 0) for r in results]
        return {
            "mean_freq_efficiency": np.mean(freq_usage),
            "mean_power_efficiency": np.mean(power_usage),
        }

    @classmethod
    def compute_all(cls, results: List[Dict]) -> Dict:
        return {
            "detection": cls.compute_detection_metrics(results),
            "reconnaissance": cls.compute_recon_metrics(results),
            "jamming": cls.compute_jamming_metrics(results),
            "communication": cls.compute_comm_metrics(results),
            "resource": cls.compute_resource_efficiency(results),
        }

    @staticmethod
    def compare_methods(
        methods: Dict[str, List[float]],  # method_name -> list of episode rewards
    ) -> str:
        """Generate comparison table in markdown."""
        rows = ["| Method | Mean Reward | Std Reward | Max | Min |"]
        rows.append("|--------|------------|------------|-----|-----|")

        for name, rewards in methods.items():
            mean_r = np.mean(rewards)
            std_r = np.std(rewards)
            max_r = np.max(rewards)
            min_r = np.min(rewards)
            rows.append(
                f"| {name} | {mean_r:.3f} | {std_r:.3f} | {max_r:.3f} | {min_r:.3f} |"
            )

        # Best method
        best = max(methods, key=lambda k: np.mean(methods[k]))
        rows.append(f"\n**Best method: {best}**")

        return "\n".join(rows)

    @staticmethod
    def save_results(results: Dict, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        with open(path, "w") as f:
            json.dump(convert(results), f, indent=2)
