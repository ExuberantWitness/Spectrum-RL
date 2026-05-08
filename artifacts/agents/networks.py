"""Neural network architectures for three-layer HRL agent."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from typing import Tuple


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, n_layers=2, act=nn.ReLU):
        super().__init__()
        layers = []
        for i in range(n_layers):
            d_in = in_dim if i == 0 else hidden
            d_out = out_dim if i == n_layers - 1 else hidden
            layers.append(nn.Linear(d_in, d_out))
            if i < n_layers - 1:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class StochasticMLP(nn.Module):
    """MLP with separate mean/log_std heads for SAC."""

    def __init__(self, in_dim, out_dim, hidden=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, out_dim)
        self.log_std = nn.Linear(hidden, out_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.apply(init_weights)
        # Small weights for mean head so policy starts near zero
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.constant_(self.mean.bias, 0.0)
        # Initialize log_std to produce std ≈ 0.5 (moderate exploration)
        nn.init.constant_(self.log_std.weight, 0.0)
        nn.init.constant_(self.log_std.bias, -0.7)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(x)
        mu = self.mean(h)
        log_std = torch.clamp(self.log_std(h), self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(x)
        std = log_std.exp()
        dist = Normal(mu, std)
        z = dist.rsample()
        log_pi = dist.log_prob(z).sum(-1, keepdim=True)
        # Squashed Gaussian
        action = torch.tanh(z)
        log_pi -= torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        return action, log_pi, mu


class QNetwork(nn.Module):
    """Twin Q-network for SAC."""

    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.q1 = MLP(state_dim + action_dim, 1, hidden)
        self.q2 = MLP(state_dim + action_dim, 1, hidden)
        self.apply(init_weights)

    def forward(self, state, action) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class StrategicNetwork(nn.Module):
    """Strategic layer: Options-based policy over function activation patterns.

    Outputs an option selection and termination probability.
    """

    def __init__(self, obs_dim: int, num_options: int = 8, hidden: int = 256):
        super().__init__()
        self.num_options = num_options
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # Option policy: π_Ω(ω|s)
        self.option_head = nn.Linear(hidden, num_options)
        # Termination: β(s, ω) for each option
        self.term_head = nn.Linear(hidden + num_options, 1)
        # Value function V(s)
        self.value = nn.Linear(hidden, 1)
        self.apply(init_weights)

    def forward(self, obs: torch.Tensor):
        h = self.encoder(obs)
        option_logits = self.option_head(h)
        value = self.value(h)
        return option_logits, value, h

    def get_option(self, obs: torch.Tensor, deterministic: bool = False):
        logits, value, h = self.forward(obs)
        if deterministic:
            option = logits.argmax(-1)
        else:
            dist = Categorical(logits=logits)
            option = dist.sample()
        return option, logits, value, h

    def get_termination(self, obs: torch.Tensor, option_onehot: torch.Tensor):
        h = self.encoder(obs)
        term_input = torch.cat([h, option_onehot], dim=-1)
        term_logit = self.term_head(term_input)
        return torch.sigmoid(term_logit)


class TacticalNetwork(nn.Module):
    """Tactical layer: Multi-agent resource allocation among four functions.

    Input: strategic option + env state
    Output: resource subspace allocation for each function
    """

    def __init__(self, obs_dim: int, num_options: int = 8, hidden: int = 256):
        super().__init__()
        in_dim = obs_dim + num_options
        self.encoder = MLP(in_dim, hidden, hidden)
        # Output: resource budget for each of 4 functions × 6 dimensions
        self.resource_head = StochasticMLP(hidden, 4 * 6, hidden)
        # Q-networks
        self.q = QNetwork(in_dim, 4 * 6, hidden)
        self.apply(init_weights)

    def forward(self, obs: torch.Tensor, option_onehot: torch.Tensor):
        x = torch.cat([obs, option_onehot], dim=-1)
        h = self.encoder(x)
        return h

    def get_action(self, obs, option_onehot, deterministic=False):
        x = torch.cat([obs, option_onehot], dim=-1)
        action, log_pi, mu = self.resource_head.sample(x)
        if deterministic:
            action = torch.tanh(mu)
        return action, log_pi

    def get_q(self, obs, option_onehot, action):
        x = torch.cat([obs, option_onehot], dim=-1)
        return self.q(x, action)


class ExecutiveNetwork(nn.Module):
    """Executive layer: Fine-grained continuous parameter control.

    Input: tactical resource allocation + env state
    Output: specific parameter values for frequency, time, power, waveform, code
    """

    def __init__(self, obs_dim: int, resource_dim: int = 24, hidden: int = 256):
        super().__init__()
        in_dim = obs_dim + resource_dim
        self.encoder = MLP(in_dim, hidden, hidden)
        # Action: concrete parameter settings
        self.action_dim = 64 + 16 + 4 + 32 + 4 + 16  # match env action space
        self.policy = StochasticMLP(hidden, self.action_dim, hidden)
        self.q = QNetwork(in_dim, self.action_dim, hidden)
        self.apply(init_weights)

    def get_action(self, obs, resource_alloc, deterministic=False):
        x = torch.cat([obs, resource_alloc], dim=-1)
        action, log_pi, mu = self.policy.sample(x)
        if deterministic:
            action = torch.tanh(mu)
        return action, log_pi

    def get_q(self, obs, resource_alloc, action):
        x = torch.cat([obs, resource_alloc], dim=-1)
        return self.q(x, action)


class OpponentEncoder(nn.Module):
    """Encode opponent behavior patterns from observation history."""

    def __init__(self, obs_dim: int, hidden: int = 128, num_intents: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden, batch_first=True, num_layers=2)
        self.intent_classifier = nn.Linear(hidden, num_intents)
        self.strategy_encoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.apply(init_weights)

    def forward(self, obs_seq: torch.Tensor):
        """obs_seq: (batch, seq_len, obs_dim)"""
        lstm_out, (h_n, _) = self.lstm(obs_seq)
        last_hidden = h_n[-1]  # (batch, hidden)
        intent_logits = self.intent_classifier(last_hidden)
        strategy_embed = self.strategy_encoder(last_hidden)
        return intent_logits, strategy_embed


class AttentionEncoder(nn.Module):
    """Multi-head attention over the six resource dimensions and four functions."""

    def __init__(self, dim: int = 128, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=dim * 4,
            batch_first=True, dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embed = nn.Parameter(torch.randn(1, 24, dim))  # 4 funcs × 6 dims
        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        """x: (batch, seq_len, dim)"""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x + self.pos_embed[:, :x.size(1), :]
        return self.transformer(x)
