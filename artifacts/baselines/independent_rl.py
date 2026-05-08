"""Independent RL baseline: Single-level SAC for each function separately."""

import numpy as np
import torch
from typing import Dict
import os
import json

from config import ExperimentConfig
from env.radar_env import IntegratedRadarEnv
from training.sac_trainer import SACTrainer
from agents.buffers import ReplayBuffer


class IndependentRLAgent:
    """Trains four independent SAC agents, one per function, with fixed resource split."""

    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        self.env = IntegratedRadarEnv(config=config.env)
        obs_dim = self.env.observation_space.shape[0]

        # Each function gets a subset of the action space
        act_dim_total = self.env.action_space.shape[0]
        act_dim_per_func = act_dim_total // 4

        self.func_names = ["detect", "recon", "jam", "comm"]
        self.sac_agents = {}
        self.buffers = {}

        for name in self.func_names:
            self.sac_agents[name] = SACTrainer(
                obs_dim, act_dim_per_func,
                hidden=config.hrl.executive_hidden,
                lr=config.hrl.executive_lr,
                device=self.device,
            )
            self.buffers[name] = ReplayBuffer(
                config.hrl.buffer_capacity, obs_dim, act_dim_per_func,
            )

    def train(self, total_steps: int = 100_000) -> Dict:
        print("Training Independent RL Baseline...")
        obs, _ = self.env.reset()
        metrics = []

        for step in range(total_steps):
            # Each SAC agent selects its part of the action
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            actions = []

            for i, name in enumerate(self.func_names):
                sub_action, _ = self.sac_agents[name].select_action(
                    obs_tensor, deterministic=False,
                )
                actions.append(sub_action.flatten())

            full_action = np.concatenate(actions)
            if len(full_action) < self.env.action_space.shape[0]:
                full_action = np.pad(full_action,
                    (0, self.env.action_space.shape[0] - len(full_action)))

            next_obs, reward, terminated, truncated, info = self.env.step(full_action)

            # Push to each agent's buffer (reward shared equally for now)
            for i, name in enumerate(self.func_names):
                self.buffers[name].push(
                    obs, actions[i], reward / 4.0, next_obs, terminated,
                )

            obs = next_obs

            if terminated or truncated:
                obs, _ = self.env.reset()

            # Update
            if self.buffers["detect"].size > self.cfg.hrl.batch_size:
                for name in self.func_names:
                    batch = self.buffers[name].sample(
                        self.cfg.hrl.batch_size, self.device,
                    )
                    self.sac_agents[name].update(batch)

            if step % 5000 == 0:
                eval_metrics = self.evaluate(5)
                eval_metrics["step"] = step
                metrics.append(eval_metrics)
                print(f"Step {step}: reward={eval_metrics['mean_reward']:.3f}")

        return {"metrics": metrics}

    def evaluate(self, num_episodes: int = 10) -> Dict:
        rewards = []
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            ep_reward = 0.0
            done = False
            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                actions = []
                for name in self.func_names:
                    sub_action, _ = self.sac_agents[name].select_action(
                        obs_tensor, deterministic=True,
                    )
                    actions.append(sub_action.flatten())
                full_action = np.concatenate(actions)
                if len(full_action) < self.env.action_space.shape[0]:
                    full_action = np.pad(full_action,
                        (0, self.env.action_space.shape[0] - len(full_action)))
                obs, reward, terminated, truncated, _ = self.env.step(full_action)
                ep_reward += reward
                done = terminated or truncated
            rewards.append(ep_reward)
        return {"mean_reward": np.mean(rewards), "std_reward": np.std(rewards)}
