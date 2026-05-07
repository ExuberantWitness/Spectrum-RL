"""Fixed rule-based policy baseline for EW resource allocation."""

import numpy as np
from typing import Dict

from config import ExperimentConfig
from env.radar_env import IntegratedRadarEnv


class FixedPolicyAgent:
    """Rule-based policy with fixed resource allocation heuristics."""

    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self.env = IntegratedRadarEnv(config=config.env)
        self.act_dim = self.env.action_space.shape[0]

        # Fixed resource split
        self.split_modes = {
            "uniform": np.array([0.25, 0.25, 0.25, 0.25]),
            "detect_heavy": np.array([0.5, 0.15, 0.2, 0.15]),
            "jam_heavy": np.array([0.15, 0.15, 0.55, 0.15]),
            "balanced": np.array([0.3, 0.2, 0.3, 0.2]),
        }

        # Beam scanning patterns
        self.scan_modes = {
            "raster": lambda t, n: (t % n) * 360.0 / n - 180.0,
            "sequential": lambda t, n: ((t // 4) % n) * 360.0 / n - 180.0,
            "targeted": lambda t, n: 0.0,  # boresight
        }

    def get_action(self, obs: np.ndarray, step: int, mode: str = "balanced") -> np.ndarray:
        """Generate action based on fixed heuristics."""
        action = np.zeros(self.act_dim, dtype=np.float32)
        c = self.cfg.env
        idx = 0

        # Frequency: uniform spread
        nf = c.num_freq_channels
        freq_pattern = np.zeros(nf, dtype=np.float32)
        active_ch = min(32, nf)
        freq_pattern[:active_ch] = 1.0
        action[idx:idx + nf] = freq_pattern
        idx += nf

        # Beams: raster scan
        nb = c.num_beams
        scan_mode = self.scan_modes["raster"]
        for i in range(nb):
            action[idx + i] = scan_mode(step + i, nb) / 180.0
        idx += nb
        for i in range(nb):
            action[idx + i] = (i * 10.0 - 20.0) / 45.0  # elevation tilt
        idx += nb

        # Power: split by mode
        power_split = self.split_modes[mode]
        for i in range(4):
            action[idx + i] = power_split[i] * 2.0 - 1.0  # map to [-1, 1]
        idx += 4

        # Waveform: fixed selection per function
        nw = c.num_waveform_types
        for func in range(4):
            wf_start = idx + func * nw
            action[wf_start + (func % nw)] = 1.0  # one-hot
        idx += 4 * nw

        # Time: equal split
        for i in range(4):
            action[idx + i] = 0.25
        idx += 4

        # Code: fixed per function
        nc = c.num_code_types
        for func in range(4):
            code_start = idx + func * nc
            action[code_start + (func % nc)] = 1.0
        idx += 4 * nc

        return np.clip(action, -1.0, 1.0)

    def evaluate(self, num_episodes: int = 10, mode: str = "balanced") -> Dict:
        rewards = []
        for ep in range(num_episodes):
            obs, _ = self.env.reset()
            ep_reward = 0.0
            step = 0
            done = False
            while not done:
                action = self.get_action(obs, step, mode)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                ep_reward += reward
                step += 1
                done = terminated or truncated
            rewards.append(ep_reward)
        return {"mean_reward": np.mean(rewards), "std_reward": np.std(rewards)}
