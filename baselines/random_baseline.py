"""Random action baseline."""

import numpy as np
from typing import Dict

from config import ExperimentConfig
from env.radar_env import IntegratedRadarEnv


class RandomBaseline:
    """Random action baseline for lower-bound performance."""

    def __init__(self, config: ExperimentConfig):
        self.env = IntegratedRadarEnv(config=config.env)
        self.act_dim = self.env.action_space.shape[0]

    def evaluate(self, num_episodes: int = 10) -> Dict:
        rewards = []
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                action = np.random.uniform(-1, 1, self.act_dim).astype(np.float32)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                ep_reward += reward
                done = terminated or truncated
            rewards.append(ep_reward)
        return {"mean_reward": np.mean(rewards), "std_reward": np.std(rewards)}
