"""PPO trainer adapted from CleanRL's battle-tested implementation.

https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from typing import Tuple


def layer_init(layer, std=2.0**0.5, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOPolicy(nn.Module):
    """CleanRL-style agent with actor (mean head + learned log_std) and critic."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.action_dim = action_dim

        # Critic: state -> value
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )

        # Actor mean: state -> action_mean
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, action_dim), std=0.01),
        )

        # Learned log_std (single parameter per action dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        # Return: action, log_prob (sum over dims), entropy (sum over dims), value
        return (
            action,
            probs.log_prob(action).sum(-1),
            probs.entropy().sum(-1),
            self.critic(x),
        )

    def get_deterministic_action(self, x):
        """Return mean action (tanh-clamped) without sampling."""
        action_mean = self.actor_mean(x)
        return torch.tanh(action_mean)


class PPOTrainer:
    """PPO trainer wrapping a PPOPolicy, using CleanRL's update logic."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        norm_adv: bool = True,
        clip_vloss: bool = True,
        target_kl: float = None,
        update_epochs: int = 10,
        device: str = "cpu",
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.norm_adv = norm_adv
        self.clip_vloss = clip_vloss
        self.target_kl = target_kl
        self.update_epochs = update_epochs
        self.action_dim = action_dim

        self.policy = PPOPolicy(state_dim, action_dim, hidden).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.train_step = 0

    def select_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return action, log_prob, value."""
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)
            if deterministic:
                action = self.policy.get_deterministic_action(state)
                log_prob = torch.zeros(state.shape[0], device=self.device)
                value = self.policy.get_value(state)
            else:
                action, log_prob, _, value = self.policy.get_action_and_value(state)
            return (
                action.squeeze(0).cpu().numpy(),
                log_prob.squeeze(0).cpu().numpy(),
                value.squeeze(0).cpu().numpy(),
            )

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
    ) -> dict:
        """CleanRL PPO update loop."""
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        logprobs = logprobs.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        values = values.to(self.device)

        # NaN guard
        if torch.isnan(obs).any() or torch.isnan(actions).any():
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0,
                    "approx_kl": 0.0, "clipfrac": 0.0}

        batch_size = obs.shape[0]
        minibatch_size = max(batch_size // 32, 64)
        b_inds = np.arange(batch_size)
        clipfracs = []

        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.policy.get_action_and_value(
                    obs[mb_inds], actions[mb_inds],
                )
                logratio = newlogprob - logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    )

                mb_advantages = advantages[mb_inds]
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss (CleanRL: max of pg_loss1 and pg_loss2)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                    v_clipped = values[mb_inds] + torch.clamp(
                        newvalue - values[mb_inds],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        self.train_step += 1
        return {
            "actor_loss": pg_loss.item(),
            "critic_loss": v_loss.item(),
            "entropy": entropy_loss.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": np.mean(clipfracs),
        }

    def save(self, path: str):
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
