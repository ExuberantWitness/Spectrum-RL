"""Two-layer HRL trainer: Coordinator + 4 Function Policies (all PPO).

Architecture:
  Coordinator (PPO, 24-dim budget, updated every 8-16 steps)
    ├── Detection Policy   (PPO, 34-dim, every step)
    ├── Recon Policy       (PPO, 34-dim, every step)
    ├── Jamming Policy     (PPO, 34-dim, every step)
    └── Communication Policy (PPO, 34-dim, every step)

Each function policy sees obs + coordinator budget, outputs its part of the
136-dim action, and gets its own per-function reward for GAE computation.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import copy
import os
import json
import time

import wandb

from config import ExperimentConfig
from env.radar_env import IntegratedRadarEnv
from agents.buffers import RolloutBuffer
from .ppo_trainer import PPOTrainer
from .curriculum import CurriculumScheduler
from .normalization import ObservationNormalizer, RewardNormalizer


# Function action dimensions: freq(16) + beam_az(2) + beam_el(2) + power(1) +
#   waveform(8) + time(1) + code(4) = 34 each
FUNC_ACTION_DIM = 34
COORD_ACTION_DIM = 24  # 6 resources x 4 functions

FUNC_NAMES = ["detect", "recon", "jam", "comm"]


class CoordinatedTrainer:
    """Coordinator + 4 Function Policies architecture (all PPO)."""

    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        # Create environment
        self.env = IntegratedRadarEnv(
            config=config.env,
            curriculum_stage=0,
        )
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        assert act_dim == FUNC_ACTION_DIM * 4, \
            f"Action dim {act_dim} != 4 x {FUNC_ACTION_DIM}"

        # Observation and reward normalization
        self.obs_normalizer = ObservationNormalizer(shape=(obs_dim,), clip_obs=10.0)
        self.reward_normalizer = RewardNormalizer(gamma=config.hrl.gamma)

        # Coordinator PPO: obs -> 24-dim budget
        coord_hidden = max(config.hrl.strategic_hidden, COORD_ACTION_DIM * 3)
        self.coord_ppo = PPOTrainer(
            state_dim=obs_dim,
            action_dim=COORD_ACTION_DIM,
            hidden=coord_hidden,
            lr=config.hrl.strategic_lr,
            gamma=config.hrl.gamma,
            gae_lambda=config.hrl.gae_lambda,
            clip_coef=config.hrl.clip_epsilon,
            ent_coef=0.05,
            target_kl=0.015,
            update_epochs=config.hrl.update_epochs,
            device=self.device,
        )

        # 4 Function PPO policies: obs + budget -> 34-dim sub-action
        func_hidden = max(config.hrl.executive_hidden, FUNC_ACTION_DIM * 3)
        func_state_dim = obs_dim + COORD_ACTION_DIM
        self.func_ppos = {}
        self.func_optimizers = []
        for name in FUNC_NAMES:
            ppo = PPOTrainer(
                state_dim=func_state_dim,
                action_dim=FUNC_ACTION_DIM,
                hidden=func_hidden,
                lr=config.hrl.executive_lr,
                gamma=config.hrl.gamma,
                gae_lambda=config.hrl.gae_lambda,
                clip_coef=config.hrl.clip_epsilon,
                ent_coef=0.05,
                target_kl=0.015,
                update_epochs=config.hrl.update_epochs,
                device=self.device,
            )
            self.func_ppos[name] = ppo

        # Rollout buffer (on-policy for all policies)
        self.rollout_length = getattr(config.hrl, 'rollout_length', 2048)
        buffer_cap = self.rollout_length + 200
        self.rollout = _CoordRolloutBuffer(
            buffer_cap, obs_dim, COORD_ACTION_DIM, FUNC_ACTION_DIM,
        )

        # Curriculum
        self.curriculum = CurriculumScheduler(
            steps_per_stage=config.curriculum.steps_per_stage,
        )

        self.total_steps = 0
        self.episodes = 0
        self.metrics_history = []
        self._update_count = 0

        # Preserve original env config for eval
        self._eval_env_config = copy.deepcopy(config.env)

    def train(self) -> Dict:
        """Main training loop."""
        print(f"\n{'='*60}")
        print("Starting Coordinated HRL Training (Coordinator + 4 Function PPOs)")
        print(f"Total steps: {self.cfg.total_steps}")
        print(f"Rollout length: {self.rollout_length}")
        print(f"Coordinator: {COORD_ACTION_DIM}-dim budget, updated every "
              f"{self.cfg.hrl.option_duration_min}-{self.cfg.hrl.option_duration_max} steps")
        print(f"Function policies: {FUNC_ACTION_DIM}-dim each x 4 = {self.act_dim}-dim total")
        print(f"{'='*60}\n")

        wandb.init(
            project="integrated-ew-hrl",
            name=f"coord_ppo_s{self.cfg.seed}",
            config={
                "total_steps": self.cfg.total_steps,
                "seed": self.cfg.seed,
                "architecture": "Coordinator + 4 Function PPOs",
                "coord_dim": COORD_ACTION_DIM,
                "func_dim": FUNC_ACTION_DIM,
                "coord_hidden": self.coord_ppo.policy.actor_mean[0].out_features
                    if hasattr(self.coord_ppo.policy.actor_mean[0], 'out_features')
                    else self.cfg.hrl.strategic_hidden,
                **{f"{n}_hidden": self.func_ppos[n].policy.actor_mean[0].out_features
                    if hasattr(self.func_ppos[n].policy.actor_mean[0], 'out_features')
                    else self.cfg.hrl.executive_hidden
                    for n in FUNC_NAMES},
                "device": str(self.device),
            },
            tags=["coordinated", "ppo", "electronic-warfare", "function-decomposition"],
        )

        obs, _ = self.env.reset()
        obs = self.obs_normalizer(obs)
        self.env.set_curriculum_stage(0)
        episode_reward = 0.0
        episode_step = 0

        # Coordinator persistence
        coord_budget = np.zeros(COORD_ACTION_DIM, dtype=np.float32)
        coord_log_prob = np.zeros(COORD_ACTION_DIM, dtype=np.float32)
        coord_value = 0.0
        coord_step_counter = 0
        coord_duration = self.cfg.hrl.option_duration_min  # resample every N steps

        start_time = time.time()

        while self.total_steps < self.cfg.total_steps:
            # --- Coordinator: sample budget every coord_duration steps ---
            if coord_step_counter >= coord_duration or self.total_steps == 0:
                coord_step_counter = 0
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                coord_action, coord_log_prob, coord_value = \
                    self.coord_ppo.select_action(obs_t, deterministic=False)
                coord_budget = coord_action.copy()
            coord_step_counter += 1

            # --- Function policies: each outputs its 34-dim sub-action ---
            func_state = np.concatenate([obs, coord_budget])
            func_state_t = torch.FloatTensor(func_state).unsqueeze(0).to(self.device)

            func_actions = []
            func_log_probs = []
            func_values = []
            for name in FUNC_NAMES:
                act, lp, val = self.func_ppos[name].select_action(
                    func_state_t, deterministic=False,
                )
                func_actions.append(act)
                func_log_probs.append(lp)
                func_values.append(val)

            # Compose full 136-dim action: concatenate 4 x 34
            full_action = np.concatenate(func_actions, axis=-1)

            # --- Environment step ---
            next_obs, reward, terminated, truncated, info = self.env.step(full_action)
            next_obs = self.obs_normalizer(next_obs)
            scaled_reward = self.reward_normalizer(reward, terminated)

            # Per-function rewards from env info
            func_rewards = info.get("function_rewards", {})
            detect_r = func_rewards.get("detect", 0.0)
            recon_r = func_rewards.get("recon", 0.0)
            jam_r = func_rewards.get("jam", 0.0)
            comm_r = func_rewards.get("comm", 0.0)

            episode_reward += reward
            episode_step += 1
            self.total_steps += 1

            # Store in rollout buffer
            self.rollout.push(
                obs.copy(),
                coord_budget.copy(), coord_log_prob, coord_value,
                func_actions, func_log_probs, func_values,
                scaled_reward,
                np.array([detect_r, recon_r, jam_r, comm_r], dtype=np.float32),
                terminated,
            )
            # Track coordinator step
            self.rollout.coord_step_flags.append(coord_step_counter == 1)

            # Step-level logging
            if self.total_steps % 100 == 0:
                wandb.log({
                    "step/reward": reward,
                    "step/curriculum_stage": self.curriculum.current_stage,
                    "step/detect_r": detect_r,
                    "step/recon_r": recon_r,
                    "step/jam_r": jam_r,
                    "step/comm_r": comm_r,
                }, step=self.total_steps)

            obs = next_obs

            # --- PPO Update when rollout buffer is full ---
            if self.rollout.full():
                with torch.no_grad():
                    # Bootstrap values for coordinator
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    bs_coord = self.coord_ppo.policy.get_value(obs_t).item()

                    # Bootstrap values for function policies
                    func_state_boot = np.concatenate([obs, coord_budget])
                    fs_t = torch.FloatTensor(func_state_boot).unsqueeze(0).to(self.device)
                    bs_funcs = [
                        self.func_ppos[n].policy.get_value(fs_t).item()
                        for n in FUNC_NAMES
                    ]

                self._ppo_update(bs_coord, bs_funcs)

            # Curriculum advancement
            stage_advanced = self.curriculum.update(reward)
            if stage_advanced:
                new_stage = self.curriculum.current_stage
                self.env.set_curriculum_stage(new_stage)
                print(f"\n>>> Advanced to curriculum stage {new_stage}: "
                      f"{CurriculumScheduler.STAGES[new_stage]['name']}")
                wandb.log({
                    "curriculum/stage": new_stage,
                    "curriculum/name": CurriculumScheduler.STAGES[new_stage]['name'],
                }, step=self.total_steps)

            # Episode end
            if terminated or truncated:
                obs, _ = self.env.reset()
                obs = self.obs_normalizer(obs)
                self.episodes += 1

                wandb.log({
                    "episode/reward": episode_reward,
                    "episode/length": episode_step,
                    "episode/count": self.episodes,
                }, step=self.total_steps)

                episode_step = 0
                coord_step_counter = coord_duration  # force resample

                if self.episodes % 10 == 0:
                    elapsed = time.time() - start_time
                    stage = self.curriculum.current_stage
                    print(f"Ep {self.episodes:5d} | Steps {self.total_steps:7d} | "
                          f"Reward {episode_reward:7.2f} | Stage {stage} | "
                          f"Time {elapsed:.0f}s")

                episode_reward = 0.0

            # Evaluation
            if self.total_steps % self.cfg.eval_interval == 0:
                eval_metrics = self.evaluate()
                eval_metrics["train_step"] = self.total_steps
                self.metrics_history.append(eval_metrics)
                print(f"\n--- Eval @ {self.total_steps} steps ---")
                for k, v in eval_metrics.items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                wandb.log({"eval/" + k: v for k, v in eval_metrics.items()
                           if isinstance(v, (int, float))}, step=self.total_steps)

            # Save checkpoint
            if self.total_steps % self.cfg.save_interval == 0:
                self.save_checkpoint()

        # Final update with remaining data
        n = len(self.rollout)
        if n > self.cfg.hrl.batch_size:
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                bs_coord = self.coord_ppo.policy.get_value(obs_t).item()
                fs_t = torch.FloatTensor(
                    np.concatenate([obs, coord_budget])
                ).unsqueeze(0).to(self.device)
                bs_funcs = [
                    self.func_ppos[n].policy.get_value(fs_t).item()
                    for n in FUNC_NAMES
                ]
            self._ppo_update(bs_coord, bs_funcs)

        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        wandb.log({"training/duration_h": elapsed / 3600.0}, step=self.total_steps)
        wandb.finish()
        return {
            "total_steps": self.total_steps, "episodes": self.episodes,
            "metrics_history": self.metrics_history,
        }

    def _ppo_update(self, bootstrap_coord: float, bootstrap_funcs: list):
        """PPO update for coordinator and all function policies."""
        self._update_count += 1
        n = len(self.rollout)

        # --- Coordinator update (over persistence windows) ---
        coord_rewards = self.rollout.coord_rewards[:n]
        coord_values = self.rollout.coord_values[:n]
        coord_advantages, coord_returns = _compute_gae(
            coord_rewards, coord_values,
            self.cfg.hrl.gamma, self.cfg.hrl.gae_lambda,
            self.rollout.dones[:n], bootstrap_coord,
        )
        coord_losses = self.coord_ppo.update(
            torch.FloatTensor(self.rollout.coord_states[:n]),
            torch.FloatTensor(self.rollout.coord_actions[:n]),
            torch.FloatTensor(self.rollout.coord_log_probs[:n]),
            torch.FloatTensor(coord_advantages),
            torch.FloatTensor(coord_returns),
            torch.FloatTensor(coord_values),
        )

        # --- Function policy updates (per-function rewards, per-dim log_probs) ---
        func_losses = {}
        func_obs = torch.FloatTensor(self.rollout.func_states[:n])
        for i, name in enumerate(FUNC_NAMES):
            func_rewards = self.rollout.func_rewards[:n, i]
            func_actions_t = torch.FloatTensor(self.rollout.func_actions[:n, i, :])
            func_log_probs_t = torch.FloatTensor(self.rollout.func_log_probs[:n, i, :])
            func_values_t = torch.FloatTensor(self.rollout.func_values[:n, i])

            func_advantages, func_returns = _compute_gae(
                func_rewards, func_values_t.numpy(),
                self.cfg.hrl.gamma, self.cfg.hrl.gae_lambda,
                self.rollout.dones[:n], bootstrap_funcs[i],
            )
            losses = self.func_ppos[name].update(
                func_obs,
                func_actions_t,
                func_log_probs_t,
                torch.FloatTensor(func_advantages),
                torch.FloatTensor(func_returns),
                func_values_t,
            )
            func_losses[name] = losses

        # Logging
        log_dict = {
            "loss/coord_actor": coord_losses["actor_loss"],
            "loss/coord_critic": coord_losses["critic_loss"],
            "loss/coord_entropy": coord_losses["entropy"],
            "loss/coord_clipfrac": coord_losses["clipfrac"],
            "ppo/update": self._update_count,
        }
        for name in FUNC_NAMES:
            log_dict[f"loss/{name}_actor"] = func_losses[name]["actor_loss"]
            log_dict[f"loss/{name}_critic"] = func_losses[name]["critic_loss"]
            log_dict[f"loss/{name}_clipfrac"] = func_losses[name]["clipfrac"]
        wandb.log(log_dict, step=self.total_steps)

        self.rollout.clear()

    def evaluate(self) -> Dict:
        """Run evaluation with deterministic actions on a fresh env."""
        eval_env = IntegratedRadarEnv(
            config=copy.deepcopy(self._eval_env_config),
            curriculum_stage=self.curriculum.current_stage,
        )
        eval_rewards = []

        for _ in range(self.cfg.num_eval_episodes):
            obs, _ = eval_env.reset()
            obs = self.obs_normalizer.normalize(obs)
            ep_reward = 0.0
            done = False
            coord_step = 0

            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

                # Coordinator (deterministic, resampled every coord_duration steps)
                if coord_step % self.cfg.hrl.option_duration_min == 0:
                    coord_action, _, _ = self.coord_ppo.select_action(
                        obs_t, deterministic=True,
                    )
                coord_step += 1

                # Function policies (deterministic)
                func_state = np.concatenate([obs, coord_action])
                fs_t = torch.FloatTensor(func_state).unsqueeze(0).to(self.device)

                func_actions = []
                for name in FUNC_NAMES:
                    act, _, _ = self.func_ppos[name].select_action(
                        fs_t, deterministic=True,
                    )
                    func_actions.append(act)

                full_action = np.concatenate(func_actions, axis=-1)
                obs, reward, terminated, truncated, _ = eval_env.step(full_action)
                obs = self.obs_normalizer.normalize(obs)
                ep_reward += reward
                done = terminated or truncated

            eval_rewards.append(ep_reward)

        return {
            "eval_reward_mean": float(np.mean(eval_rewards)),
            "eval_reward_std": float(np.std(eval_rewards)),
        }

    def save_checkpoint(self):
        os.makedirs(self.cfg.model_dir, exist_ok=True)
        path = os.path.join(self.cfg.model_dir, f"coord_step{self.total_steps}.pt")
        state = {
            "coord_policy": self.coord_ppo.policy.state_dict(),
            "total_steps": self.total_steps,
            "episodes": self.episodes,
            "curriculum_stage": self.curriculum.current_stage,
            "metrics": self.metrics_history,
        }
        for name in FUNC_NAMES:
            state[f"{name}_policy"] = self.func_ppos[name].policy.state_dict()
        torch.save(state, path)
        print(f"Checkpoint saved: {path}")

    def save_results(self):
        os.makedirs(self.cfg.results_dir, exist_ok=True)
        path = os.path.join(self.cfg.results_dir, "metrics.json")
        with open(path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"Results saved: {path}")


# ---------------------------------------------------------------------------
# Rollout buffer for coordinated architecture
# ---------------------------------------------------------------------------

class _CoordRolloutBuffer:
    """On-policy rollout buffer for Coordinator + 4 Function Policies.

    Stores:
      - Coordinator: states, actions, log_probs, values, scalar rewards
      - Functions: states (obs+budget), actions(4x34), log_probs(4), values(4),
        per-function rewards(4)
    """

    def __init__(self, capacity: int, obs_dim: int, coord_dim: int, func_dim: int):
        self.capacity = capacity
        self.ptr = 0

        # Coordinator arrays
        self.coord_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.coord_actions = np.zeros((capacity, coord_dim), dtype=np.float32)
        self.coord_log_probs = np.zeros((capacity, coord_dim), dtype=np.float32)
        self.coord_values = np.zeros((capacity,), dtype=np.float32)
        self.coord_rewards = np.zeros((capacity,), dtype=np.float32)

        # Function arrays (4 functions, per-dim log_probs)
        self.func_states = np.zeros((capacity, obs_dim + coord_dim), dtype=np.float32)
        self.func_actions = np.zeros((capacity, 4, func_dim), dtype=np.float32)
        self.func_log_probs = np.zeros((capacity, 4, func_dim), dtype=np.float32)
        self.func_values = np.zeros((capacity, 4), dtype=np.float32)
        self.func_rewards = np.zeros((capacity, 4), dtype=np.float32)

        # Shared
        self.dones = np.zeros((capacity,), dtype=np.float32)

        # Bookkeeping: which steps start a new coordinator window
        self.coord_step_flags = []

    def push(
        self,
        obs: np.ndarray,
        coord_action: np.ndarray,
        coord_log_prob: np.ndarray,   # per-dim, shape (coord_dim,)
        coord_value: float,
        func_actions: list,
        func_log_probs: list,         # list of 4 per-dim arrays, each (func_dim,)
        func_values: list,
        reward: float,
        func_rewards: np.ndarray,
        done: bool,
    ):
        if self.ptr >= self.capacity:
            return

        self.coord_states[self.ptr] = obs
        self.coord_actions[self.ptr] = coord_action
        self.coord_log_probs[self.ptr] = coord_log_prob
        self.coord_values[self.ptr] = coord_value
        self.coord_rewards[self.ptr] = reward

        self.func_states[self.ptr] = np.concatenate([obs, coord_action])
        for i in range(4):
            self.func_actions[self.ptr, i] = func_actions[i]
            self.func_log_probs[self.ptr, i] = func_log_probs[i]
            self.func_values[self.ptr, i] = func_values[i]
        self.func_rewards[self.ptr] = func_rewards

        self.dones[self.ptr] = float(done)
        self.ptr += 1

    def __len__(self):
        return self.ptr

    def full(self):
        return self.ptr >= self.capacity

    def clear(self):
        self.ptr = 0
        self.coord_step_flags = []


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float,
    gae_lambda: float,
    dones: np.ndarray,
    bootstrap_value: float,
):
    """Compute GAE advantages and returns."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    returns = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        if dones[t] > 0.5:
            next_value = 0.0
        elif t == n - 1:
            next_value = bootstrap_value
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
        advantages[t] = gae
        returns[t] = gae + values[t]
    return advantages, returns
