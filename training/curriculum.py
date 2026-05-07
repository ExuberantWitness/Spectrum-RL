"""Curriculum learning scheduler for progressive EW training."""

import numpy as np
from typing import List, Dict


class CurriculumScheduler:
    """Manages progressive training stages: simple → complex scenarios."""

    STAGES = [
        {
            "name": "detect_only",
            "description": "Detection + basic communication, no jamming",
            "active_functions": [0, 3],  # detect, comm
            "num_opponents": 0,
            "num_targets": 2,
            "target_hostile_ratio": 0.0,
            "jam_power_max_db": 0.0,
            "success_threshold": 0.75,
            "min_steps": 20_000,
        },
        {
            "name": "detect_jam",
            "description": "Detection + jamming, basic opponent",
            "active_functions": [0, 2],  # detect, jam
            "num_opponents": 1,
            "num_targets": 3,
            "target_hostile_ratio": 0.5,
            "jam_power_max_db": 50.0,
            "success_threshold": 0.65,
            "min_steps": 30_000,
        },
        {
            "name": "detect_recon_jam",
            "description": "Three-function integration",
            "active_functions": [0, 1, 2],  # detect, recon, jam
            "num_opponents": 2,
            "num_targets": 4,
            "target_hostile_ratio": 0.5,
            "jam_power_max_db": 60.0,
            "success_threshold": 0.60,
            "min_steps": 30_000,
        },
        {
            "name": "full_integrated",
            "description": "All four functions with intelligent opponents",
            "active_functions": [0, 1, 2, 3],  # all
            "num_opponents": 2,
            "num_targets": 4,
            "target_hostile_ratio": 0.5,
            "jam_power_max_db": 70.0,
            "success_threshold": None,  # No early exit from final stage
            "min_steps": 50_000,
        },
    ]

    def __init__(self, steps_per_stage: int = 50_000):
        self.steps_per_stage = steps_per_stage
        self.current_stage = 0
        self.stage_step = 0
        self.stage_history: List[Dict] = []
        self._rolling_reward = []

    def get_config(self) -> dict:
        """Get environment config for current stage."""
        return self.STAGES[self.current_stage].copy()

    def update(self, reward: float) -> bool:
        """Update based on step reward. Returns True if stage advanced."""
        self.stage_step += 1
        self._rolling_reward.append(reward)
        if len(self._rolling_reward) > 1000:
            self._rolling_reward.pop(0)

        cfg = self.STAGES[self.current_stage]

        # Check if we should advance
        if self.current_stage < len(self.STAGES) - 1:
            min_steps = cfg["min_steps"]
            threshold = cfg["success_threshold"]

            if self.stage_step >= min_steps and threshold is not None:
                recent_reward = np.mean(self._rolling_reward[-500:])
                max_reward = self._max_possible_reward()
                normalized = recent_reward / max(max_reward, 1e-8)

                if normalized >= threshold:
                    self._advance_stage()
                    return True

        # Fixed step-based advancement as fallback
        if self.stage_step >= self.steps_per_stage \
                and self.current_stage < len(self.STAGES) - 1:
            self._advance_stage()
            return True

        return False

    def _advance_stage(self):
        self.stage_history.append({
            "stage": self.current_stage,
            "name": self.STAGES[self.current_stage]["name"],
            "steps": self.stage_step,
            "final_reward": np.mean(self._rolling_reward[-500:]) if self._rolling_reward else 0,
        })
        self.current_stage += 1
        self.stage_step = 0
        self._rolling_reward = []

    def _max_possible_reward(self) -> float:
        """Compute the maximum possible per-step reward for the current stage."""
        # Weights match _get_function_weights in radar_env.py
        stage = self.current_stage
        if stage == 0:  # detect_only: w=[1.0, 0.0, 0.0, 0.2]
            return 1.0 + 0.2 + 0.1  # detect + comm + balance
        elif stage == 1:  # detect_jam: w=[0.6, 0.0, 0.4, 0.0]
            return 0.6 + 0.4 + 0.1  # detect + jam + balance
        elif stage == 2:  # detect_recon_jam: w=[0.4, 0.3, 0.3, 0.0]
            return 0.4 + 0.3 + 0.3 + 0.1  # detect + recon + jam + balance
        else:  # full_integrated: w=[0.25, 0.25, 0.25, 0.25]
            return 1.0 + 0.1  # all four + balance

    def get_progress(self) -> dict:
        return {
            "stage": self.current_stage,
            "stage_name": self.STAGES[self.current_stage]["name"] if self.current_stage < len(self.STAGES) else "done",
            "stage_step": self.stage_step,
            "history": self.stage_history,
        }
