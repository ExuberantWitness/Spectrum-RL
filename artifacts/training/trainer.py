"""Three-layer HRL trainer with PPO and curriculum learning."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, Tuple
import copy
import os
import json
import time
import wandb

from config import ExperimentConfig
from env.radar_env import IntegratedRadarEnv
from agents.networks import StrategicNetwork
from agents.buffers import HRLReplayBuffer, RolloutBuffer
from .ppo_trainer import PPOTrainer
from .curriculum import CurriculumScheduler
from .normalization import ObservationNormalizer, RewardNormalizer


class HRLTrainer:
    """Orchestrates three-layer HRL training with PPO for sub-policies."""

    def __init__(self, config: ExperimentConfig):
        self.cfg = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        # Create environment (train + separate eval to avoid state corruption)
        self.env = IntegratedRadarEnv(
            config=config.env,
            curriculum_stage=0,
        )
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.resource_dim = 24  # 4 functions x 6 dimensions

        # Observation and reward normalization
        self.obs_normalizer = ObservationNormalizer(shape=(obs_dim,), clip_obs=10.0)
        self.reward_normalizer = RewardNormalizer(gamma=config.hrl.gamma)

        # Strategic network (Options-based, discrete actions -> REINFORCE + baseline)
        self.strategic = StrategicNetwork(
            obs_dim, config.hrl.strategic_options, config.hrl.strategic_hidden,
        ).to(self.device)
        self.strategic_optim = optim.Adam(
            self.strategic.parameters(), lr=config.hrl.strategic_lr,
        )

        # Persistent buffer for strategic transitions (off-policy style, mixed options)
        self.strategic_buffer = HRLReplayBuffer(
            config.hrl.buffer_capacity, obs_dim, act_dim,
            config.hrl.strategic_options, self.resource_dim,
        )

        # On-policy rollout buffer for PPO (tactical + executive)
        self.rollout_length = getattr(config.hrl, 'rollout_length', 2048)
        self.rollout = RolloutBuffer(
            self.rollout_length + 200,  # extra space for episode boundaries
            obs_dim, act_dim, config.hrl.strategic_options, self.resource_dim,
        )

        # Curriculum scheduler
        self.curriculum = CurriculumScheduler(
            steps_per_stage=config.curriculum.steps_per_stage,
        )

        # PPO trainers for tactical and executive layers
        self.tactical_ppo = PPOTrainer(
            obs_dim + config.hrl.strategic_options,
            self.resource_dim,
            hidden=config.hrl.tactical_hidden,
            lr=config.hrl.tactical_lr,
            gamma=config.hrl.gamma,
            gae_lambda=config.hrl.gae_lambda,
            clip_coef=config.hrl.clip_epsilon,
            update_epochs=config.hrl.update_epochs,
            device=self.device,
        )
        self.executive_ppo = PPOTrainer(
            obs_dim + self.resource_dim,
            act_dim,
            hidden=config.hrl.executive_hidden,
            lr=config.hrl.executive_lr,
            gamma=config.hrl.gamma,
            gae_lambda=config.hrl.gae_lambda,
            clip_coef=config.hrl.clip_epsilon,
            update_epochs=config.hrl.update_epochs,
            device=self.device,
        )

        self.total_steps = 0
        self.episodes = 0
        self.metrics_history = []
        self._update_count = 0

        # Preserve original env config for eval (curriculum mutates self.cfg.env)
        self._eval_env_config = copy.deepcopy(config.env)

    def train(self) -> Dict:
        """Main training loop with PPO rollout-update cycle."""
        print(f"\n{'='*60}")
        print("Starting Three-Layer HRL Training (PPO)")
        print(f"Total steps: {self.cfg.total_steps}")
        print(f"Rollout length: {self.rollout_length}")
        print(f"Curriculum stages: {len(CurriculumScheduler.STAGES)}")
        print(f"{'='*60}\n")

        wandb.init(
            project="integrated-ew-hrl",
            name=f"hrl_ppo_s{self.cfg.seed}",
            config={
                "total_steps": self.cfg.total_steps,
                "seed": self.cfg.seed,
                "architecture": "Three-Layer HRL (Strategic->Tactical->Executive) with PPO",
                "algorithm": "PPO (tactical+executive) + REINFORCE (strategic)",
                "strategic_lr": self.cfg.hrl.strategic_lr,
                "tactical_lr": self.cfg.hrl.tactical_lr,
                "executive_lr": self.cfg.hrl.executive_lr,
                "n_options": self.cfg.hrl.strategic_options,
                "rollout_length": self.rollout_length,
                "reward_scale": self.cfg.hrl.reward_scale,
                "curriculum_stages": self.cfg.curriculum.stages,
                "steps_per_stage": self.cfg.curriculum.steps_per_stage,
                "device": str(self.device),
            },
            tags=["hrl", "ppo", "electronic-warfare", "phased-array-radar", "curriculum-learning"],
        )

        obs, _ = self.env.reset()
        obs = self.obs_normalizer(obs)
        self.env.set_curriculum_stage(0)
        episode_reward = 0.0
        episode_step = 0

        # Option tracking
        current_option = None
        option_onehot = np.zeros(self.cfg.hrl.strategic_options, dtype=np.float32)
        option_start_step = 0
        option_probs = np.zeros(self.cfg.hrl.strategic_options, dtype=np.float32)

        # Last resource_alloc for executive state construction
        last_resource_alloc = np.zeros(self.resource_dim, dtype=np.float32)
        # Tactical persistence: resample every N steps for hierarchical time separation
        tac_step_counter = 0
        tac_duration = self.cfg.hrl.option_duration_min  # tactical slower than executive

        start_time = time.time()

        while self.total_steps < self.cfg.total_steps:
            # --- Strategic layer: select option ---
            option_duration = episode_step - option_start_step
            should_replan = (
                current_option is None
                or option_duration >= self.cfg.hrl.option_duration_max
            )

            # β termination: early replanning after min duration
            if not should_replan and option_duration >= self.cfg.hrl.option_duration_min \
                    and self.strategic_buffer.size > 100:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                term_prob = self.strategic.get_termination(
                    obs_tensor,
                    torch.FloatTensor(option_onehot).unsqueeze(0).to(self.device),
                ).item()
                should_replan = np.random.random() < term_prob

            if should_replan or current_option is None:
                # Flush previous option's strategic transitions
                self.rollout.flush_strategic(
                    self.cfg.hrl.gamma, self.strategic_buffer,
                )

                # Select new option
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    option, logits, value, _ = self.strategic.get_option(
                        obs_tensor, deterministic=False,
                    )
                    current_option = option[0].item() if option.dim() > 0 else option.item()
                    option_probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                    option_onehot = np.zeros(self.cfg.hrl.strategic_options, dtype=np.float32)
                    option_onehot[current_option] = 1.0
                option_start_step = episode_step

            # --- Tactical layer (PPO): allocate resources (persists for K steps) ---
            should_resample_tac = (tac_step_counter >= tac_duration) or should_replan \
                or current_option is None
            if should_resample_tac:
                tac_step_counter = 0
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                opt_tensor = torch.FloatTensor(option_onehot).unsqueeze(0).to(self.device)
                tac_state = torch.cat([obs_tensor, opt_tensor], dim=-1)
                resource_action, tac_log_prob, tac_value = self.tactical_ppo.select_action(
                    tac_state, deterministic=False,
                )
                cached_resource_action = resource_action.copy()
                cached_tac_log_prob = tac_log_prob
                cached_tac_value = tac_value
            else:
                resource_action = cached_resource_action
                tac_log_prob = cached_tac_log_prob
                tac_value = cached_tac_value
            tac_step_counter += 1

            # --- Executive layer (PPO): fine-grained parameters ---
            res_tensor = torch.FloatTensor(resource_action).unsqueeze(0).to(self.device)
            obs_tensor2 = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            exec_state = torch.cat([obs_tensor2, res_tensor], dim=-1)
            final_action, exec_log_prob, exec_value = self.executive_ppo.select_action(
                exec_state, deterministic=False,
            )

            # --- Environment step ---
            next_obs, reward, terminated, truncated, info = self.env.step(final_action)
            next_obs = self.obs_normalizer(next_obs)
            scaled_reward = self.reward_normalizer(reward, terminated)
            episode_reward += reward
            episode_step += 1
            self.total_steps += 1

            # Store in rollout buffer
            self.rollout.push(
                obs.copy(), current_option, option_probs.copy(), option_onehot.copy(),
                resource_action.copy(), final_action.copy(),
                tac_log_prob, tac_value,
                exec_log_prob, exec_value,
                scaled_reward, terminated,
            )
            # Track for option-level returns
            self.rollout.option_step_rewards.append(scaled_reward)
            self.rollout.option_ptr_indices.append(self.rollout.ptr - 1)

            last_resource_alloc = resource_action

            # Step-level wandb logging (every 100 steps)
            if self.total_steps % 100 == 0:
                wandb.log({"step/reward": reward,
                           "step/curriculum_stage": self.curriculum.current_stage},
                          step=self.total_steps)

            obs = next_obs

            # --- PPO Update when rollout buffer is full ---
            if self.rollout.full():
                # Compute bootstrap values from current networks
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    opt_t = torch.FloatTensor(option_onehot).unsqueeze(0).to(self.device)
                    res_t = torch.FloatTensor(last_resource_alloc).unsqueeze(0).to(self.device)

                    tac_state_t = torch.cat([obs_t, opt_t], dim=-1)
                    bootstrap_tac = self.tactical_ppo.policy.get_value(tac_state_t).item()

                    exec_state_t = torch.cat([obs_t, res_t], dim=-1)
                    bootstrap_exec = self.executive_ppo.policy.get_value(exec_state_t).item()

                self._ppo_update(bootstrap_tac, bootstrap_exec)

            # --- Strategic update from persistent buffer ---
            if self.strategic_buffer.size > self.cfg.hrl.batch_size * 2:
                strat_batch = self.strategic_buffer.sample_strategic(
                    self.cfg.hrl.batch_size, self.device,
                )
                self._update_strategic(strat_batch)

            # Curriculum advancement
            stage_advanced = self.curriculum.update(reward)
            if stage_advanced:
                new_stage = self.curriculum.current_stage
                self.env.set_curriculum_stage(new_stage)
                print(f"\n>>> Advanced to curriculum stage {new_stage}: "
                      f"{CurriculumScheduler.STAGES[new_stage]['name']}")
                wandb.log({"curriculum/stage": new_stage,
                           "curriculum/name": CurriculumScheduler.STAGES[new_stage]['name']},
                          step=self.total_steps)

            # Episode end
            if terminated or truncated:
                self.rollout.flush_strategic(
                    self.cfg.hrl.gamma, self.strategic_buffer,
                )
                obs, _ = self.env.reset()
                obs = self.obs_normalizer(obs)
                self.episodes += 1

                wandb.log({
                    "episode/reward": episode_reward,
                    "episode/length": episode_step,
                    "episode/count": self.episodes,
                }, step=self.total_steps)

                current_option = None
                episode_step = 0

                if self.episodes % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"Ep {self.episodes:5d} | Steps {self.total_steps:7d} | "
                          f"Reward {episode_reward:7.2f} | "
                          f"Stage {self.curriculum.current_stage} | "
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

        # Final update with remaining rollout data
        if len(self.rollout) > self.cfg.hrl.batch_size:
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                opt_t = torch.FloatTensor(option_onehot).unsqueeze(0).to(self.device)
                res_t = torch.FloatTensor(last_resource_alloc).unsqueeze(0).to(self.device)
                tac_state_t = torch.cat([obs_t, opt_t], dim=-1)
                btac = self.tactical_ppo.policy.get_value(tac_state_t).item()
                exec_state_t = torch.cat([obs_t, res_t], dim=-1)
                bexec = self.executive_ppo.policy.get_value(exec_state_t).item()
            self._ppo_update(btac, bexec)

        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        wandb.log({"training/duration_h": elapsed / 3600.0}, step=self.total_steps)
        wandb.finish()
        return {"total_steps": self.total_steps, "episodes": self.episodes,
                "metrics_history": self.metrics_history}

    def _ppo_update(self, bootstrap_tac: float = 0.0, bootstrap_exec: float = 0.0):
        """CleanRL-style PPO update for tactical and executive layers."""
        self._update_count += 1
        n = len(self.rollout)

        # --- Tactical layer ---
        tac_advantages, tac_returns = self.rollout.compute_gae(
            self.cfg.hrl.gamma,
            self.cfg.hrl.gae_lambda,
            self.rollout.tac_values[:n],
            bootstrap_value=bootstrap_tac,
        )

        tac_losses = self.tactical_ppo.update(
            torch.FloatTensor(self.rollout.tac_states[:n]),
            torch.FloatTensor(self.rollout.tac_actions[:n]),
            torch.FloatTensor(self.rollout.tac_log_probs[:n]).squeeze(-1),
            torch.FloatTensor(tac_advantages).squeeze(-1),
            torch.FloatTensor(tac_returns).squeeze(-1),
            torch.FloatTensor(self.rollout.tac_values[:n]).squeeze(-1),
        )

        # --- Executive layer ---
        exec_advantages, exec_returns = self.rollout.compute_gae(
            self.cfg.hrl.gamma,
            self.cfg.hrl.gae_lambda,
            self.rollout.exec_values[:n],
            bootstrap_value=bootstrap_exec,
        )

        exec_losses = self.executive_ppo.update(
            torch.FloatTensor(self.rollout.exec_states[:n]),
            torch.FloatTensor(self.rollout.exec_actions[:n]),
            torch.FloatTensor(self.rollout.exec_log_probs[:n]).squeeze(-1),
            torch.FloatTensor(exec_advantages).squeeze(-1),
            torch.FloatTensor(exec_returns).squeeze(-1),
            torch.FloatTensor(self.rollout.exec_values[:n]).squeeze(-1),
        )

        # Log PPO losses
        wandb.log({
            "loss/tac_actor": tac_losses["actor_loss"],
            "loss/tac_critic": tac_losses["critic_loss"],
            "loss/tac_entropy": tac_losses["entropy"],
            "loss/tac_kl": tac_losses["approx_kl"],
            "loss/tac_clipfrac": tac_losses["clipfrac"],
            "loss/exec_actor": exec_losses["actor_loss"],
            "loss/exec_critic": exec_losses["critic_loss"],
            "loss/exec_entropy": exec_losses["entropy"],
            "loss/exec_kl": exec_losses["approx_kl"],
            "loss/exec_clipfrac": exec_losses["clipfrac"],
            "ppo/update": self._update_count,
        }, step=self.total_steps)

        self.rollout.clear()

    def _update_strategic(self, batch: Tuple):
        obs, options, returns, next_obs, dones, old_probs = batch

        option_logits, values, _ = self.strategic(obs)
        new_probs = torch.softmax(option_logits, dim=-1)
        selected_new_probs = new_probs.gather(1, options.unsqueeze(-1))
        selected_old_probs = old_probs.gather(1, options.unsqueeze(-1))

        # Importance sampling ratio for off-policy correction
        is_ratio = selected_new_probs / (selected_old_probs + 1e-8)
        is_ratio = torch.clamp(is_ratio, 0.5, 2.0)  # clip for stability

        log_probs = torch.log_softmax(option_logits, dim=-1)
        selected_log_probs = log_probs.gather(1, options.unsqueeze(-1))

        # Advantage: raw returns minus value baseline, normalize advantage only
        advantage = returns - values.detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        policy_loss = -(is_ratio * selected_log_probs * advantage).mean()
        value_loss = nn.MSELoss()(values, returns)

        total_loss = policy_loss + 0.5 * value_loss

        entropy = -(new_probs * torch.log(new_probs + 1e-8)).sum(-1).mean()
        total_loss -= 0.01 * entropy

        self.strategic_optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.strategic.parameters(), 10.0)
        self.strategic_optim.step()

        if self._update_count % 20 == 0:
            wandb.log({
                "loss/strat_total": total_loss.item(),
                "loss/strat_policy": policy_loss.item(),
                "loss/strat_value": value_loss.item(),
                "loss/strat_entropy": entropy.item(),
            }, step=self.total_steps)

    def evaluate(self) -> Dict:
        """Run evaluation episodes with deterministic actions (fresh env, no state leak)."""
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

            while not done:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

                # Get option (deterministic)
                option, _, _, _ = self.strategic.get_option(
                    obs_tensor, deterministic=True,
                )
                opt_idx = option.item() if option.dim() > 1 else option.item()
                option_onehot = np.zeros(self.cfg.hrl.strategic_options, dtype=np.float32)
                option_onehot[opt_idx] = 1.0
                opt_tensor = torch.FloatTensor(option_onehot).unsqueeze(0).to(self.device)

                # Tactical (deterministic)
                tac_state = torch.cat([obs_tensor, opt_tensor], dim=-1)
                res_action, _, _ = self.tactical_ppo.select_action(
                    tac_state, deterministic=True,
                )

                # Executive (deterministic)
                res_tensor = torch.FloatTensor(res_action).unsqueeze(0).to(self.device)
                exec_state = torch.cat([obs_tensor, res_tensor], dim=-1)
                action, _, _ = self.executive_ppo.select_action(
                    exec_state, deterministic=True,
                )

                obs, reward, terminated, truncated, _ = eval_env.step(action)
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
        path = os.path.join(self.cfg.model_dir, f"hrl_step{self.total_steps}.pt")
        torch.save({
            "strategic": self.strategic.state_dict(),
            "tactical_policy": self.tactical_ppo.policy.state_dict(),
            "executive_policy": self.executive_ppo.policy.state_dict(),
            "total_steps": self.total_steps,
            "episodes": self.episodes,
            "curriculum_stage": self.curriculum.current_stage,
            "metrics": self.metrics_history,
        }, path)
        print(f"Checkpoint saved: {path}")

    def save_results(self):
        os.makedirs(self.cfg.results_dir, exist_ok=True)
        path = os.path.join(self.cfg.results_dir, "metrics.json")
        with open(path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
        print(f"Results saved: {path}")
