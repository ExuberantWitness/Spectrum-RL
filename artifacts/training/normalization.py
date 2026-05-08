"""Running mean/std normalization for observations and rewards.

Adapted from CleanRL's gym wrappers and OpenAI baselines.
"""

import numpy as np


class RunningMeanStd:
    """Tracks running mean and std via Welford's online algorithm."""

    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

    @property
    def std(self):
        return np.sqrt(self.var)


class ObservationNormalizer:
    """Normalizes observations with running mean/std, clipping extreme values."""

    def __init__(self, shape, clip_obs=10.0):
        self.rms = RunningMeanStd(shape=shape)
        self.clip_obs = clip_obs

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        self.rms.update(obs.reshape(1, -1) if obs.ndim == 1 else obs[:1])
        normalized = self.rms.normalize(obs)
        return np.clip(normalized, -self.clip_obs, self.clip_obs)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize without updating running stats (for evaluation)."""
        normalized = self.rms.normalize(obs)
        return np.clip(normalized, -self.clip_obs, self.clip_obs)


class RewardNormalizer:
    """Normalizes rewards by running std of discounted returns."""

    def __init__(self, gamma=0.99, epsilon=1e-8):
        self.returns_rms = RunningMeanStd(shape=())
        self.gamma = gamma
        self.epsilon = epsilon
        self.discounted_return = 0.0

    def __call__(self, reward: float, done: bool = False) -> float:
        self.discounted_return = self.discounted_return * self.gamma + reward
        self.returns_rms.update(np.array([self.discounted_return]))
        if done:
            self.discounted_return = 0.0
        return reward / (self.returns_rms.std + self.epsilon)
