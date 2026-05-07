"""Soft Actor-Critic implementation for HRL sub-policies."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Tuple
import copy


class SACTrainer:
    """SAC trainer for a single policy layer."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        device: str = "cpu",
    ):
        from agents.networks import StochasticMLP, QNetwork

        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.actor = StochasticMLP(state_dim, action_dim, hidden).to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = QNetwork(state_dim, action_dim, hidden).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = nn.Parameter(torch.zeros(1, device=device))
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)

        self.train_step = 0

    def select_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            action, log_pi, _ = self.actor.sample(state)
            if deterministic:
                mu, _ = self.actor.forward(state)
                action = torch.tanh(mu)
                log_pi = torch.zeros(action.shape[0], 1, device=self.device)
            return action.squeeze(0).cpu().numpy(), log_pi.squeeze(0).cpu().numpy()

    def update(
        self, batch: Tuple[torch.Tensor, ...]
    ) -> dict:
        obs, actions, rewards, next_obs, dones = batch
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)

        # --- Critic update ---
        with torch.no_grad():
            next_action, next_log_pi, _ = self.actor.sample(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = rewards + self.gamma * (1 - dones) * q_next

        q1, q2 = self.critic(obs, actions)
        critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optim.step()

        # --- Actor update ---
        new_action, log_pi, _ = self.actor.sample(obs)
        q1_new, q2_new = self.critic(obs, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_pi - q_new).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optim.step()

        # --- Alpha update ---
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp().item()
        self.alpha = max(min(self.alpha, 10.0), 0.01)  # clamp for stability

        # --- Target update ---
        with torch.no_grad():
            for p, p_targ in zip(self.critic.parameters(),
                                  self.critic_target.parameters()):
                p_targ.data.mul_(1 - self.tau).add_(self.tau * p.data)

        self.train_step += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "alpha_loss": alpha_loss.item(),
        }

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha,
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "alpha_optim": self.alpha_optim.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha.data = ckpt["log_alpha"].data
