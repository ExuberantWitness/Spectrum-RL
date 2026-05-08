"""Integrated Phased Array Radar Environment with four functions:
Detection, Reconnaissance, Jamming, Communication (探/侦/干/通).

Six-dimensional resource space: Frequency, Time, Space, Power, Waveform, Code.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional

from .channel_models import RadarChannel, PropagationModel
from .target_models import TargetModel, OpponentModel


class IntegratedRadarEnv(gym.Env):
    """Gym environment for integrated EW radar control."""

    metadata = {"render_modes": ["human"]}

    # Function indices
    FUNC_DETECT = 0
    FUNC_RECON = 1
    FUNC_JAM = 2
    FUNC_COMM = 3

    def __init__(self, config=None, curriculum_stage: int = 0):
        super().__init__()
        if config is None:
            from config import EnvConfig
            config = EnvConfig()

        self.cfg = config
        self.curriculum_stage = curriculum_stage
        self._setup_spaces()
        self._setup_models()

    def _setup_spaces(self):
        c = self.cfg
        # Snapshot num_targets for fixed observation dimension.
        # Curriculum may change c.num_targets later via set_curriculum_stage(),
        # but neural networks require a constant input size throughout training.
        self._obs_num_targets = c.num_targets
        # Observation: [spectrum_occupancy(N_freq), threat_levels(N_beams),
        #               target_params(4*_obs_num_targets), function_status(4),
        #               resource_usage(6), time_step(1)]
        obs_dim = (
            c.num_freq_channels
            + c.num_beams * 2
            + 4 * self._obs_num_targets
            + 4
            + 6
            + 1
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action: [freq_allocation(N_freq), beam_direction(N_beams),
        #          power_levels(4 functions), waveform_selection(4),
        #          time_allocation(4), code_selection(4)]
        act_dim = (
            c.num_freq_channels
            + c.num_beams * 2  # azimuth + elevation
            + 4
            + 4 * c.num_waveform_types
            + 4
            + 4 * c.num_code_types
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

    def _setup_models(self):
        c = self.cfg
        # Use env's own RNG for episode-varying scenarios (not fixed c.seed)
        rng = self.np_random if hasattr(self, 'np_random') and self.np_random is not None \
            else np.random.default_rng(c.seed)
        self.channel = RadarChannel(
            carrier_freq_ghz=c.carrier_freq_ghz,
            bandwidth_mhz=c.bandwidth_mhz,
        )
        self.propagation = PropagationModel(
            num_channels=c.num_freq_channels,
            seed=c.seed,
        )

        self.radar_pos = np.zeros(3, dtype=np.float32)

        # Create targets
        self.targets = []
        for i in range(c.num_targets):
            angle = 2 * np.pi * i / c.num_targets
            dist = 50 + rng.random() * 100  # km
            pos = np.array([
                dist * np.cos(angle), dist * np.sin(angle), rng.uniform(-5, 5)
            ])
            vel = np.array([
                rng.uniform(-300, 300), rng.uniform(-300, 300), rng.uniform(-50, 50)
            ])
            rcs = rng.uniform(*c.target_rcs_range)
            self.targets.append(TargetModel(
                target_id=i, init_position=pos, init_velocity=vel,
                rcs_dbsm=rcs, is_hostile=i < c.num_targets // 2,
            ))

        # Create opponents
        self.opponents = []
        for i in range(c.num_opponents):
            angle = np.pi + 2 * np.pi * i / c.num_opponents
            pos = np.array([80 * np.cos(angle), 80 * np.sin(angle), rng.uniform(-3, 3)])
            intent = rng.choice(OpponentModel.INTENT_TYPES)
            self.opponents.append(OpponentModel(
                opponent_id=i, position=pos,
                max_power_db=rng.uniform(*c.opponent_power_range),
                intent=intent,
            ))

        self.time_step = 0
        self._function_history = np.zeros(4, dtype=np.float32)

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._setup_models()
        self.time_step = 0
        self._function_history.fill(0)
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.time_step += 1

        # Parse action into six resource dimensions
        resources = self._parse_action(action)
        self._last_resources = resources

        # Simulate four functions
        detect_result = self._simulate_detection(resources)
        recon_result = self._simulate_reconnaissance(resources)
        jam_result = self._simulate_jamming(resources)
        comm_result = self._simulate_communication(resources)

        # Opponent reactions
        for opp in self.opponents:
            opp.update_intent()

        # Update target states
        for tgt in self.targets:
            tgt.step(1.0)
            tgt.update_relative(self.radar_pos, self.cfg.carrier_freq_ghz)

        # Update propagation
        self.propagation.step()

        # Compute reward
        reward = self._compute_reward(
            detect_result, recon_result, jam_result, comm_result
        )

        # Check termination
        terminated = self.time_step >= self.cfg.max_steps
        truncated = False

        # Update function history (exponential moving average)
        alpha = 0.1
        self._function_history[0] = (1 - alpha) * self._function_history[0] + alpha * detect_result.get("success", 0.0)
        self._function_history[1] = (1 - alpha) * self._function_history[1] + alpha * recon_result.get("coverage", 0.0)
        self._function_history[2] = (1 - alpha) * self._function_history[2] + alpha * jam_result.get("effectiveness", 0.0)
        self._function_history[3] = (1 - alpha) * self._function_history[3] + alpha * comm_result.get("rate", 0.0)

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _parse_action(self, action: np.ndarray) -> dict:
        """Parse flat action vector into six resource dimensions."""
        c = self.cfg
        idx = 0

        # Frequency allocation (N_freq channels, softmax to distribution)
        nf = c.num_freq_channels
        freq_raw = action[idx:idx + nf]
        freq_alloc = self._softmax(freq_raw)
        idx += nf

        # Beam direction (azimuth for each of N_beams + elevation)
        nb = c.num_beams
        beam_az = action[idx:idx + nb] * 180.0  # [-180, 180] deg
        idx += nb
        beam_el = action[idx:idx + nb] * 45.0  # [-45, 45] deg
        idx += nb

        # Power levels for 4 functions (independent sigmoid, not softmax — so total can < P_max)
        power_raw = action[idx:idx + 4]
        max_power_linear = 10 ** (self.cfg.max_power_db / 10.0)
        # Each function gets 0-40% of P_max independently, total ≤ P_max not guaranteed
        power_frac = 1.0 / (1.0 + np.exp(-power_raw * 2.0)) * 0.4
        power_alloc = power_frac * max_power_linear
        power_db = 10 * np.log10(np.clip(power_alloc, 1e-10, None))
        idx += 4

        # Waveform selection (4 functions × N_waveform logits)
        nw = c.num_waveform_types
        wf_logits = action[idx:idx + 4 * nw].reshape(4, nw)
        waveform_probs = np.array([self._softmax(wf_logits[i]) for i in range(4)])
        idx += 4 * nw

        # Time allocation (4 functions)
        time_raw = action[idx:idx + 4]
        time_alloc = self._softmax(time_raw) * c.num_time_slots
        idx += 4

        # Code selection (4 functions × N_code logits)
        nc = c.num_code_types
        code_logits = action[idx:idx + 4 * nc].reshape(4, nc)
        code_probs = np.array([self._softmax(code_logits[i]) for i in range(4)])
        idx += 4 * nc

        return {
            "freq_alloc": freq_alloc,
            "beam_az": beam_az,
            "beam_el": beam_el,
            "power_db": power_db,
            "waveform_probs": waveform_probs,
            "time_alloc": time_alloc,
            "code_probs": code_probs,
        }

    def _simulate_detection(self, resources: dict) -> dict:
        """Simulate radar target detection performance."""
        c = self.cfg
        total_pd = 0.0

        for tgt in self.targets:
            # Find closest beam to target (with angle wrap-around)
            best_ang_err = 180.0
            for i in range(c.num_beams):
                raw_err = abs(resources["beam_az"][i] - tgt.angle_deg)
                ang_err = min(raw_err, 360.0 - raw_err)
                if ang_err < best_ang_err:
                    best_ang_err = ang_err

            if best_ang_err < 10.0:
                beam_gain_loss = (best_ang_err / 10.0) ** 2 * 3.0
            else:
                beam_gain_loss = 20.0  # sidelobe

            effective_gain = c.antenna_gain_db - beam_gain_loss

            # Average power over allocated frequencies
            tgt_freq_bin = int(
                (tgt.doppler_hz / 5000.0 + 0.5) * c.num_freq_channels
            )
            tgt_freq_bin = np.clip(tgt_freq_bin, 0, c.num_freq_channels - 1)
            freq_factor = resources["freq_alloc"][tgt_freq_bin] * c.num_freq_channels

            pwr_db = resources["power_db"][self.FUNC_DETECT]
            snr = self.channel.radar_range_equation(
                pwr_db + 10 * np.log10(freq_factor),
                effective_gain,
                tgt.rcs_dbsm,
                tgt.range_km,
                c.system_loss_db,
            )
            pd = self.channel.detection_probability(snr)
            # Use continuous expected Pd (not stochastic binary) for low-variance reward
            tgt.detected = pd > 0.5
            total_pd += pd

        n_targets = max(c.num_targets, 1)
        return {
            "total_targets": n_targets,
            "avg_pd": total_pd / n_targets,
            "success": total_pd / n_targets,  # continuous: mean Pd, not binary count
        }

    def _simulate_reconnaissance(self, resources: dict) -> dict:
        """Simulate electronic reconnaissance (spectrum sensing)."""
        c = self.cfg
        channel_gains = self.propagation.step()

        # Apply jamming interference on occupied channels
        occupied_mask = np.zeros(c.num_freq_channels)
        for opp in self.opponents:
            rel = opp.position - self.radar_pos
            rng_km = float(np.linalg.norm(rel))
            opp_action = opp.get_action(self.radar_pos, np.zeros(0))
            occupied_mask += opp_action["freq_channels"] * (
                1.0 / max(rng_km / 50.0, 1.0)
            )

        occupied_mask = np.clip(occupied_mask, 0, 1)
        channel_gains = self.propagation.apply_interference(
            channel_gains, occupied_mask,
            10 * np.log10(10 ** (self.cfg.opponent_power_range[1] / 10.0)),
        )

        # Recon quality: how well we sense each occupied channel
        pwr_db = resources["power_db"][self.FUNC_RECON]
        freq_sensing = resources["freq_alloc"] * pwr_db

        # Threshold ~ noise floor: only sense channels with meaningful SNR
        noise_floor = 10 * np.log10(self.channel.k * self.channel.t0 * self.channel.bw)
        detected_signal = (channel_gains + freq_sensing) > (noise_floor - 10.0)
        true_occupancy = occupied_mask > 0.3
        tp = np.sum(detected_signal & true_occupancy)
        fp = np.sum(detected_signal & ~true_occupancy)
        fn = np.sum(~detected_signal & true_occupancy)

        coverage = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)

        return {
            "coverage": coverage,
            "precision": precision,
            "f1": 2 * coverage * precision / max(coverage + precision, 1e-8),
            "n_channels_sensed": int(np.sum(detected_signal)),
        }

    def _simulate_jamming(self, resources: dict) -> dict:
        """Simulate electronic jamming against opponents."""
        c = self.cfg
        total_jsr = 0.0
        total_effectiveness = 0.0

        for opp in self.opponents:
            rel = opp.position - self.radar_pos
            opp_range = float(np.linalg.norm(rel))
            opp_action = opp.get_action(self.radar_pos, np.zeros(0))

            # Jam on opponent's active channels
            opp_channels = opp_action["freq_channels"]
            our_jam_power = np.sum(resources["freq_alloc"] * opp_channels)
            jam_pwr_db = resources["power_db"][self.FUNC_JAM] + \
                10 * np.log10(np.clip(our_jam_power, 1e-10, None))

            # J/S needed for effective jamming
            jsr_db = jam_pwr_db - opp_action["power_db"]
            jsr_db += c.antenna_gain_db - c.system_loss_db
            jsr_db -= 20 * np.log10(max(opp_range / 50.0, 1.0))

            # Continuous effectiveness: sigmoid around 3 dB threshold
            effectiveness = 1.0 / (1.0 + np.exp(-(jsr_db - 3.0)))
            total_effectiveness += effectiveness
            total_jsr += jsr_db

        n_opp = max(c.num_opponents, 1)
        return {
            "n_jammed": int(total_effectiveness >= c.num_opponents * 0.5),
            "total_opponents": n_opp,
            "avg_jsr_db": total_jsr / n_opp,
            "effectiveness": total_effectiveness / n_opp,
        }

    def _simulate_communication(self, resources: dict) -> dict:
        """Simulate communication performance."""
        c = self.cfg

        # Comm uses dedicated frequency channels
        comm_start = 3 * c.num_freq_channels // 4
        comm_freqs = resources["freq_alloc"][comm_start:]
        comm_pwr_db = resources["power_db"][self.FUNC_COMM]

        # Check for interference on comm channels
        interfered_power = -np.inf
        for opp in self.opponents:
            opp_action = opp.get_action(self.radar_pos, np.zeros(0))
            opp_comm_interference = np.sum(
                opp_action["freq_channels"][comm_start:]
            )
            if opp_comm_interference > 0:
                rel = opp.position - self.radar_pos
                opp_range = float(np.linalg.norm(rel))
                interfered_power = max(
                    interfered_power,
                    opp_action["power_db"] - 20 * np.log10(max(opp_range, 1.0)),
                )

        # SINR per channel
        noise_db = 10 * np.log10(self.channel.k * self.channel.t0 * self.channel.bw)
        sinr_per_ch = comm_pwr_db + 10 * np.log10(np.clip(comm_freqs, 1e-10, None))

        if interfered_power > -np.inf:
            sinr_per_ch = 10 * np.log10(
                np.maximum(
                    10 ** (sinr_per_ch / 10.0) - 10 ** (interfered_power / 10.0),
                    1e-10,
                )
            )

        sinr_per_ch -= noise_db + c.system_loss_db

        # Channel capacity (Shannon, per channel)
        capacity = self.channel.channel_capacity(sinr_per_ch)
        effective_channels = np.sum(comm_freqs > 0.01)
        data_rate = np.sum(capacity) * self.channel.bw / 1e6  # total Mbps

        return {
            "sinr_db": float(np.mean(sinr_per_ch)),
            "capacity_bps_hz": float(np.mean(capacity)),
            "rate": float(np.clip(data_rate, 0, 1000)),
            "channels_used": int(effective_channels),
        }

    def _compute_reward(
        self,
        detect: dict,
        recon: dict,
        jam: dict,
        comm: dict,
    ) -> float:
        """Compute weighted reward based on curriculum stage."""
        w = self._get_function_weights()

        r_detect = detect["success"] * w[0]
        r_recon = recon["coverage"] * w[1]
        # Continuous jamming reward (sigmoid around 3 dB threshold instead of binary)
        r_jam = (1.0 / (1.0 + np.exp(-(jam["avg_jsr_db"] - 3.0)))) * w[2]
        r_comm = np.tanh(comm["rate"] / 100.0) * w[3]

        # Balance bonus: only active functions (weight > 0)
        active_scores = []
        active_weights = []
        for score, weight in zip(
            [detect["success"], recon["coverage"], jam.get("effectiveness", 0.0),
             min(comm["rate"] / 200.0, 1.0)],
            w
        ):
            if weight > 0:
                active_scores.append(score)
                active_weights.append(weight)
        if len(active_scores) > 1:
            balance = 1.0 - np.std(active_scores)
        else:
            balance = 0.0

        # Stronger penalty for resource overuse
        res = self._last_resources if hasattr(self, '_last_resources') else {}
        power_db = res.get("power_db", np.array([40.0, 30.0, 35.0, 25.0]))
        total_power = 10 * np.log10(np.sum(10 ** (power_db / 10.0)) + 1e-10)
        power_penalty = max(0.0, total_power - self.cfg.max_power_db) * 0.1

        reward = r_detect + r_recon + r_jam + r_comm + 0.1 * balance - power_penalty
        self._last_component_rewards = {
            "detect": float(r_detect),
            "recon": float(r_recon),
            "jam": float(r_jam),
            "comm": float(r_comm),
            "balance": float(0.1 * balance),
            "power_penalty": float(power_penalty),
        }
        return float(reward)

    def _get_function_weights(self) -> np.ndarray:
        """Get function importance weights based on curriculum stage."""
        stage = self.curriculum_stage
        if stage == 0:  # detect only
            return np.array([1.0, 0.0, 0.0, 0.2])
        elif stage == 1:  # detect + jam
            return np.array([0.6, 0.0, 0.4, 0.0])
        elif stage == 2:  # detect + recon + jam
            return np.array([0.4, 0.3, 0.3, 0.0])
        else:  # full integrated
            return np.array([0.25, 0.25, 0.25, 0.25])

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        c = self.cfg
        obs_parts = []

        # Spectrum occupancy: first 32 ch = channel power (dB), last 32 ch = opponent activity
        channel_gains = self.propagation._channel_state
        spectrum = 10 * np.log10(np.abs(channel_gains) ** 2 + 1e-10)
        opp_occupancy = np.zeros(c.num_freq_channels)
        for opp in self.opponents:
            opp_occupancy += opp.get_action(self.radar_pos, np.zeros(0))["freq_channels"]
        half = c.num_freq_channels // 2
        spectrum_obs = np.concatenate([spectrum.real[:half], opp_occupancy[:half]])

        obs_parts.append(spectrum_obs)

        # Threat levels per beam direction
        threats = np.zeros(c.num_beams * 2, dtype=np.float32)
        for i, tgt in enumerate(self.targets):
            beam_idx = int((tgt.angle_deg + 180.0) / 360.0 * c.num_beams)
            beam_idx = np.clip(beam_idx, 0, c.num_beams - 1)
            if tgt.is_hostile and tgt.range_km < 200:
                threats[beam_idx] += 1.0 / max(tgt.range_km / 50.0, 1.0)
            threats[c.num_beams + beam_idx] += (
                1.0 / max(tgt.range_km / 50.0, 1.0)
            )
        obs_parts.append(threats)

        # Target parameters (fixed-size slots; unused slots stay zero)
        target_params = np.zeros(4 * self._obs_num_targets, dtype=np.float32)
        for i, tgt in enumerate(self.targets):
            target_params[4 * i] = tgt.range_km / c.max_range_km
            target_params[4 * i + 1] = tgt.angle_deg / 180.0
            target_params[4 * i + 2] = tgt.doppler_hz / 5000.0
            target_params[4 * i + 3] = float(tgt.is_hostile)
        obs_parts.append(target_params)

        # Function status (EMA of recent performance)
        func_status = self._function_history.copy()
        obs_parts.append(func_status)

        # Resource usage (6D compact summary)
        resource_usage = self._get_resource_usage()
        obs_parts.append(resource_usage)

        # Time step
        obs_parts.append(np.array([self.time_step / c.max_steps], dtype=np.float32))

        obs = np.concatenate(obs_parts)
        return obs.astype(np.float32)

    def _get_resource_usage(self) -> np.ndarray:
        """Create a 6-dimensional compact summary of current resource usage."""
        res = self._last_resources if hasattr(self, '_last_resources') else {}
        if not res:
            return np.zeros(6, dtype=np.float32)

        # Frequency utilization (mean allocation across channels)
        freq_util = float(np.mean(res.get("freq_alloc", np.zeros(1))))

        # Time utilization (mean allocation across functions)
        time_util = float(np.mean(res.get("time_alloc", np.zeros(4))))

        # Spatial coverage (mean beam coverage normalized)
        beam_az = res.get("beam_az", np.zeros(1))
        beam_el = res.get("beam_el", np.zeros(1))
        space_cov = float((np.mean(np.abs(beam_az)) / 180.0 + np.mean(np.abs(beam_el)) / 45.0) / 2.0)

        # Power utilization (mean normalized by max power)
        power_db = res.get("power_db", np.array([40.0]))
        power_util = float(np.mean(np.clip(power_db / self.cfg.max_power_db, 0.0, 1.0)))

        # Waveform diversity (entropy of waveform probabilities)
        wf_probs = res.get("waveform_probs", np.ones((4, 1)) / 4)
        wf_entropy = float(np.mean([-np.sum(p * np.log(p + 1e-8)) / np.log(len(p))
                                     for p in wf_probs]))

        # Code diversity (entropy of code probabilities)
        code_probs = res.get("code_probs", np.ones((4, 1)) / 4)
        code_entropy = float(np.mean([-np.sum(p * np.log(p + 1e-8)) / np.log(len(p))
                                       for p in code_probs]))

        return np.array([
            freq_util, time_util, space_cov, power_util, wf_entropy, code_entropy,
        ], dtype=np.float32)

    def _get_info(self) -> dict:
        info = {
            "time_step": self.time_step,
            "function_weights": self._get_function_weights(),
        }
        if hasattr(self, '_last_component_rewards'):
            info["function_rewards"] = self._last_component_rewards
        return info

    def set_curriculum_stage(self, stage: int):
        self.curriculum_stage = stage
        # Apply stage-specific env params from CurriculumScheduler.STAGES
        from training.curriculum import CurriculumScheduler
        if stage < len(CurriculumScheduler.STAGES):
            stage_cfg = CurriculumScheduler.STAGES[stage]
            if "num_opponents" in stage_cfg and stage_cfg["num_opponents"] != self.cfg.num_opponents:
                self.cfg.num_opponents = stage_cfg["num_opponents"]
            if "num_targets" in stage_cfg and stage_cfg["num_targets"] != self.cfg.num_targets:
                self.cfg.num_targets = stage_cfg["num_targets"]
        self._setup_models()

    @staticmethod
    def _softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
        x = x / temp
        x = x - np.max(x)
        e = np.exp(np.clip(x, -20, 20))
        return e / (e.sum() + 1e-10)
