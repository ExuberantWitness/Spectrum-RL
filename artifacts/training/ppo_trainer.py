"""PPO trainer adapted from CleanRL with improvements for high-dim action spaces.

Key improvements over original:
- State-dependent log_std (not global parameter) for adaptive exploration
- Per-dimension PPO clipping (fixes 82% clipfrac for high-dim actions)
- LayerNorm for training stability
- Proper entropy coefficient and target KL for early stopping
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from typing import Tuple, Optional


def layer_init(layer, std=2.0**0.5, bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOPolicy(nn.Module):
    """CleanRL-style agent with state-dependent log_std and LayerNorm."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 256,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.use_layernorm = use_layernorm

        # Critic: state -> value
        critic_layers = [
            layer_init(nn.Linear(state_dim, hidden)),
            nn.LayerNorm(hidden) if use_layernorm else nn.Identity(),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.LayerNorm(hidden) if use_layernorm else nn.Identity(),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        ]
        self.critic = nn.Sequential(*critic_layers)

        # Actor mean: state -> action_mean
        actor_layers = [
            layer_init(nn.Linear(state_dim, hidden)),
            nn.LayerNorm(hidden) if use_layernorm else nn.Identity(),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.LayerNorm(hidden) if use_layernorm else nn.Identity(),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, action_dim), std=0.01),
        ]
        self.actor_mean = nn.Sequential(*actor_layers)

        # State-dependent log_std head (parallel to actor_mean, shares first layers)
        logstd_layers = [
            layer_init(nn.Linear(state_dim, hidden)),
            nn.LayerNorm(hidden) if use_layernorm else nn.Identity(),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.LayerNorm(hidden) if use_layernorm else nn.Identity(),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, action_dim), std=0.01),
        ]
        self.actor_logstd_head = nn.Sequential(*logstd_layers)
        # Initialize log_std bias to -0.5 (std ≈ 0.6) for the output layer
        nn.init.constant_(self.actor_logstd_head[-1].bias, -0.5)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd_head(x)
        # Clamp log_std to prevent extreme values
        action_logstd = torch.clamp(action_logstd, -5.0, 2.0)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            z = probs.rsample()
            action = torch.tanh(z)
        else:
            z = torch.atanh(torch.clamp(action, -0.999, 0.999))
        # Squashed Gaussian log-prob: per-dimension (no sum)
        log_prob_per_dim = probs.log_prob(z)  # (B, action_dim)
        log_prob_per_dim -= torch.log(1 - action.pow(2) + 1e-6)
        return (
            action,
            log_prob_per_dim,  # per-dim, shape (B, action_dim) — caller sums if needed
            probs.entropy(),   # per-dim, shape (B, action_dim)
            self.critic(x),
        )

    def get_deterministic_action(self, x):
        """Return mean action (tanh-clamped) without sampling."""
        action_mean = self.actor_mean(x)
        return torch.tanh(action_mean)


class PPOTrainer:
    """PPO trainer with per-dimension clipping for high-dim action spaces."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        norm_adv: bool = True,
        clip_vloss: bool = True,
        target_kl: Optional[float] = 0.015,
        update_epochs: int = 10,
        device: str = "cpu",
        use_layernorm: bool = True,
        per_dim_clip: bool = True,
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
        self.per_dim_clip = per_dim_clip

        # Ensure hidden size is sufficient for the action dimension
        if hidden < action_dim * 2:
            hidden = action_dim * 2

        self.policy = PPOPolicy(state_dim, action_dim, hidden, use_layernorm).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.train_step = 0

    def select_action(
        self, state: torch.Tensor, deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return action, per-dim log_prob, value."""
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)
            if deterministic:
                action_mean = self.policy.actor_mean(state)
                action = torch.tanh(action_mean)
                logstd = self.policy.actor_logstd_head(state)
                logstd = torch.clamp(logstd, -5.0, 2.0)
                probs = Normal(action_mean, torch.exp(logstd))
                log_prob_per_dim = probs.log_prob(action_mean)
                log_prob_per_dim -= torch.log(1 - action.pow(2) + 1e-6)
                value = self.policy.get_value(state)
            else:
                action, log_prob_per_dim, _, value = self.policy.get_action_and_value(state)
            return (
                action.squeeze(0).cpu().numpy(),
                log_prob_per_dim.squeeze(0).cpu().numpy(),
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
        """PPO update with TRUE per-dimension clipping.

        logprobs must be per-dim shape (B, action_dim), NOT summed to scalars.
        Ratio, clip, and loss are computed per-dim then averaged over (batch, dims).
        """
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        logprobs = logprobs.to(self.device)
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)
        values = values.to(self.device)

        # NaN guard
        if torch.isnan(obs).any() or torch.isnan(actions).any() or torch.isnan(logprobs).any():
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0,
                    "approx_kl": 0.0, "clipfrac": 0.0}

        batch_size = obs.shape[0]
        minibatch_size = max(batch_size // 8, 64)
        b_inds = np.arange(batch_size)
        clipfracs = []

        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob_per_dim, entropy_per_dim, newvalue = \
                    self.policy.get_action_and_value(obs[mb_inds], actions[mb_inds])

                # True per-dimension ratio: (mb, D) vs (mb, D)
                old_logprob_per_dim = logprobs[mb_inds]  # (mb, D)
                logratio_per_dim = newlogprob_per_dim - old_logprob_per_dim
                ratio_per_dim = logratio_per_dim.exp()  # (mb, D)

                # Per-dimension clip: clip each dim independently, then mean over (B, D)
                mb_advantages = advantages[mb_inds]  # (mb,)
                if self.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                pg_loss1 = -mb_advantages.unsqueeze(-1) * ratio_per_dim
                pg_loss2 = -mb_advantages.unsqueeze(-1) * torch.clamp(
                    ratio_per_dim, 1.0 - self.clip_coef, 1.0 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()  # mean over (B, D)

                with torch.no_grad():
                    # Total KL: sum per-dim KL, mean over batch
                    approx_kl = ((ratio_per_dim - 1) - logratio_per_dim).sum(-1).mean()
                    clipfracs.append(
                        ((ratio_per_dim - 1.0).abs() > self.clip_coef)
                        .float().mean().item()  # mean over (B, D)
                    )

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

                entropy_loss = entropy_per_dim.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None and approx_kl > self.target_kl * self.action_dim:
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
