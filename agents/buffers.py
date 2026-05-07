"""Experience replay buffers for HRL training."""

import numpy as np
import torch
from typing import Tuple, List


class ReplayBuffer:
    """Standard experience replay buffer."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str = "cpu") -> Tuple:
        idx = np.random.randint(0, self.size, size=min(batch_size, self.size))
        return (
            torch.FloatTensor(self.obs[idx]).to(device),
            torch.FloatTensor(self.actions[idx]).to(device),
            torch.FloatTensor(self.rewards[idx]).to(device),
            torch.FloatTensor(self.next_obs[idx]).to(device),
            torch.FloatTensor(self.dones[idx]).to(device),
        )

    def __len__(self) -> int:
        return self.size


class HRLReplayBuffer:
    """Hierarchical replay buffer with separate storage for each layer."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        option_dim: int,
        resource_dim: int = 24,
    ):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        # Strategic layer
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.options = np.zeros((capacity,), dtype=np.int64)
        self.option_logits = np.zeros((capacity, option_dim), dtype=np.float32)
        self.strategic_reward = np.zeros((capacity, 1), dtype=np.float32)

        # Tactical layer
        self.resource_alloc = np.zeros((capacity, resource_dim), dtype=np.float32)
        self.tactical_reward = np.zeros((capacity, 1), dtype=np.float32)

        # Executive layer
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)

        # Shared
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        # Option transition tracking
        self.option_lengths = np.zeros((capacity,), dtype=np.int64)
        self._current_option_start = 0

    def push_strategic(
        self,
        obs: np.ndarray,
        option: int,
        option_logits: np.ndarray,
        next_obs: np.ndarray,
        reward: float,
        done: bool,
        option_length: int,
    ):
        self.obs[self.ptr] = obs
        self.options[self.ptr] = option
        self.option_logits[self.ptr] = option_logits
        self.next_obs[self.ptr] = next_obs
        self.strategic_reward[self.ptr] = reward
        self.dones[self.ptr] = done
        self.option_lengths[self.ptr] = option_length

    def push_tactical(
        self,
        resource_alloc: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.resource_alloc[self.ptr] = resource_alloc
        self.tactical_reward[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done

    def push_executive(
        self,
        action: np.ndarray,
        reward: float,
    ):
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward

    def advance(self):
        """Advance buffer pointer after all per-step data has been pushed."""
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update_strategic_rewards(self, indices, rewards):
        """Update strategic rewards at given buffer indices (for option-level returns)."""
        for idx, r in zip(indices, rewards):
            self.strategic_reward[idx] = r

    def sample_strategic(self, batch_size: int, device: str = "cpu") -> Tuple:
        idx = np.random.randint(0, self.size, size=min(batch_size, self.size))
        return (
            torch.FloatTensor(self.obs[idx]).to(device),
            torch.LongTensor(self.options[idx]).to(device),
            torch.FloatTensor(self.strategic_reward[idx]).to(device),
            torch.FloatTensor(self.next_obs[idx]).to(device),
            torch.FloatTensor(self.dones[idx]).to(device),
        )

    def sample_tactical(self, batch_size: int, device: str = "cpu") -> Tuple:
        """Sample tactical-layer transitions: state=[obs, option_onehot], action=resource_alloc."""
        idx = np.random.randint(0, self.size, size=min(batch_size, self.size))
        option_dim = self.option_logits.shape[1]
        option_onehot = np.zeros((len(idx), option_dim), dtype=np.float32)
        for i, j in enumerate(idx):
            option_onehot[i, self.options[j]] = 1.0

        tactical_state = np.concatenate([self.obs[idx], option_onehot], axis=1)
        tactical_next_state = np.concatenate([self.next_obs[idx], option_onehot], axis=1)
        return (
            torch.FloatTensor(tactical_state).to(device),
            torch.FloatTensor(self.resource_alloc[idx]).to(device),
            torch.FloatTensor(self.tactical_reward[idx]).to(device),
            torch.FloatTensor(tactical_next_state).to(device),
            torch.FloatTensor(self.dones[idx]).to(device),
        )

    def sample_executive(self, batch_size: int, device: str = "cpu") -> Tuple:
        """Sample executive-layer transitions: state=[obs, resource_alloc], action=final_action."""
        idx = np.random.randint(0, self.size, size=min(batch_size, self.size))
        exec_state = np.concatenate([self.obs[idx], self.resource_alloc[idx]], axis=1)
        exec_next_state = np.concatenate([self.next_obs[idx], self.resource_alloc[idx]], axis=1)
        return (
            torch.FloatTensor(exec_state).to(device),
            torch.FloatTensor(self.actions[idx]).to(device),
            torch.FloatTensor(self.rewards[idx]).to(device),
            torch.FloatTensor(exec_next_state).to(device),
            torch.FloatTensor(self.dones[idx]).to(device),
        )

    def __len__(self) -> int:
        return self.size


class RolloutBuffer:
    """On-policy rollout buffer for PPO (tactical + executive layers).

    Collects transitions during rollout, computes GAE, feeds PPO update,
    then clears for the next rollout.  Strategic transitions are flushed
    to a persistent HRLReplayBuffer on option termination.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        act_dim: int,
        option_dim: int,
        resource_dim: int = 24,
    ):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        # Tactical layer: state=[obs|option_onehot], action=resource_alloc
        self.tac_states = np.zeros((capacity, obs_dim + option_dim), dtype=np.float32)
        self.tac_actions = np.zeros((capacity, resource_dim), dtype=np.float32)
        self.tac_log_probs = np.zeros((capacity, 1), dtype=np.float32)
        self.tac_values = np.zeros((capacity, 1), dtype=np.float32)

        # Executive layer: state=[obs|resource_alloc], action=final_action
        self.exec_states = np.zeros((capacity, obs_dim + resource_dim), dtype=np.float32)
        self.exec_actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.exec_log_probs = np.zeros((capacity, 1), dtype=np.float32)
        self.exec_values = np.zeros((capacity, 1), dtype=np.float32)

        # Shared
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        # Strategic tracking (flushed to persistent buffer on option termination)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.options = np.zeros((capacity,), dtype=np.int64)
        self.option_logits = np.zeros((capacity, option_dim), dtype=np.float32)
        self.option_step_rewards = []
        self.option_ptr_indices = []

    def push(
        self,
        obs: np.ndarray,
        option: int,
        option_logits: np.ndarray,
        option_onehot: np.ndarray,
        resource_alloc: np.ndarray,
        final_action: np.ndarray,
        tac_log_prob: np.ndarray,
        tac_value: np.ndarray,
        exec_log_prob: np.ndarray,
        exec_value: np.ndarray,
        reward: float,
        done: bool,
    ):
        self.tac_states[self.ptr] = np.concatenate([obs, option_onehot])
        self.tac_actions[self.ptr] = resource_alloc
        self.tac_log_probs[self.ptr] = tac_log_prob
        self.tac_values[self.ptr] = tac_value

        self.exec_states[self.ptr] = np.concatenate([obs, resource_alloc])
        self.exec_actions[self.ptr] = final_action
        self.exec_log_probs[self.ptr] = exec_log_prob
        self.exec_values[self.ptr] = exec_value

        self.obs[self.ptr] = obs
        self.options[self.ptr] = option
        self.option_logits[self.ptr] = option_logits

        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def compute_gae(self, gamma: float, gae_lambda: float, values: np.ndarray,
                    bootstrap_value: float = 0.0):
        n = self.size
        advantages = np.zeros((n, 1), dtype=np.float32)
        returns_arr = np.zeros((n, 1), dtype=np.float32)
        gae_val = 0.0
        for t in reversed(range(n)):
            if self.dones[t, 0] > 0.5:
                next_value = 0.0
            elif t == n - 1:
                next_value = bootstrap_value
            else:
                next_value = values[t + 1, 0]
            delta = self.rewards[t, 0] + gamma * next_value - values[t, 0]
            gae_val = delta + gamma * gae_lambda * (1.0 - self.dones[t, 0]) * gae_val
            advantages[t, 0] = gae_val
            returns_arr[t, 0] = gae_val + values[t, 0]
        return advantages, returns_arr

    def flush_strategic(self, gamma: float, buffer):
        if not self.option_ptr_indices:
            self.option_step_rewards = []
            self.option_ptr_indices = []
            return
        returns = []
        G = 0.0
        for r in reversed(self.option_step_rewards):
            G = r + gamma * G
            returns.append(G)
        returns = list(reversed(returns))
        for idx, ret in zip(self.option_ptr_indices, returns):
            next_idx = min(idx + 1, self.size - 1)
            buffer.push_strategic(
                self.obs[idx].copy(),
                self.options[idx],
                self.option_logits[idx].copy(),
                self.obs[next_idx].copy(),
                ret,
                self.dones[idx, 0] > 0.5,
                len(self.option_ptr_indices),
            )
            buffer.advance()
        self.option_step_rewards = []
        self.option_ptr_indices = []

    def clear(self):
        self.ptr = 0
        self.size = 0

    def full(self) -> bool:
        return self.size >= self.capacity

    def __len__(self) -> int:
        return self.size
