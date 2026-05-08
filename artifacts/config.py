"""Configuration for the HRL-based Integrated EW experiment."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class EnvConfig:
    # Simulation parameters
    num_freq_channels: int = 64
    num_time_slots: int = 16
    num_beams: int = 8
    max_power_db: float = 60.0  # dBm
    num_waveform_types: int = 8
    num_code_types: int = 4

    # Target parameters
    num_targets: int = 4
    max_range_km: float = 200.0
    target_rcs_range: Tuple[float, float] = (-10.0, 30.0)  # dBsm

    # Opponent parameters
    num_opponents: int = 2
    opponent_power_range: Tuple[float, float] = (30.0, 70.0)  # dBm
    num_opponent_intents: int = 4  # search, evade, fire-coordinate, silent-recon

    # Environment constants
    noise_figure_db: float = 5.0
    bandwidth_mhz: float = 10.0
    carrier_freq_ghz: float = 10.0
    antenna_gain_db: float = 40.0
    system_loss_db: float = 6.0

    # Random seed for reproducible simulations
    seed: int = 42

    # Episode
    max_steps: int = 200

    # Curriculum stage (set by trainer)
    curriculum_stage: int = 0


@dataclass
class HRLConfig:
    # Strategic layer (Options-based)
    strategic_lr: float = 3e-4
    strategic_hidden: int = 256
    strategic_options: int = 8  # number of option policies
    option_duration_min: int = 4
    option_duration_max: int = 32

    # Tactical layer (Multi-agent resource allocation)
    tactical_lr: float = 3e-4
    tactical_hidden: int = 128

    # Executive layer (Continuous parameter control)
    executive_lr: float = 3e-4
    executive_hidden: int = 128

    # Shared
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    rollout_length: int = 512
    update_epochs: int = 10
    batch_size: int = 256
    buffer_capacity: int = 1_000_000
    reward_scale: float = 2.0


@dataclass
class CurriculumConfig:
    stages: List[str] = field(default_factory=lambda: [
        "detect_only",       # Stage 1: Detection + basic communication
        "detect_jam",        # Stage 2: Detection + jamming
        "detect_recon_jam",  # Stage 3: Three-function
        "full_integrated",   # Stage 4: All four functions
    ])
    steps_per_stage: int = 50_000
    success_threshold: float = 0.65  # min success rate to advance


@dataclass
class ExperimentConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    hrl: HRLConfig = field(default_factory=HRLConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)

    # Experiment
    seed: int = 42
    total_steps: int = 200_000
    eval_interval: int = 5_000
    num_eval_episodes: int = 20
    log_interval: int = 1_000
    save_interval: int = 20_000

    # Output
    results_dir: str = "artifacts/results"
    model_dir: str = "artifacts/models"

    # Compute
    device: str = "cuda"
    num_workers: int = 4
