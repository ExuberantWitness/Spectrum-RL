"""Target and opponent behavior models."""

import numpy as np
from typing import Tuple, List


class TargetModel:
    """Models a radar target with kinematic state and RCS."""

    def __init__(
        self,
        target_id: int,
        init_position: np.ndarray,  # [x, y, z] in km
        init_velocity: np.ndarray,  # [vx, vy, vz] in m/s
        rcs_dbsm: float,
        is_hostile: bool = True,
    ):
        self.id = target_id
        self.position = np.asarray(init_position, dtype=np.float32)
        self.velocity = np.asarray(init_velocity, dtype=np.float32)
        self.rcs_dbsm = rcs_dbsm
        self.is_hostile = is_hostile
        self.detected = False
        self.jammed = False
        self.angle_deg: float = 0.0
        self.range_km: float = 0.0
        self.doppler_hz: float = 0.0

    def step(self, dt: float = 1.0) -> None:
        self.position += self.velocity * dt / 1000.0  # m/s -> km/s

    def update_relative(self, radar_pos: np.ndarray, fc_ghz: float) -> None:
        rel = self.position - radar_pos
        self.range_km = float(np.linalg.norm(rel))
        self.angle_deg = float(np.degrees(np.arctan2(rel[1], rel[0])))
        radial_vel = np.dot(self.velocity, rel / self.range_km) if self.range_km > 0 else 0
        self.doppler_hz = 2 * radial_vel / (3e8 / (fc_ghz * 1e9))


class OpponentModel:
    """Models an adversary electronic warfare system."""

    INTENT_TYPES = ["search_track", "evade", "fire_coordinate", "silent_recon"]

    def __init__(
        self,
        opponent_id: int,
        position: np.ndarray,
        max_power_db: float = 60.0,
        intent: str = "search_track",
    ):
        self.id = opponent_id
        self.position = np.asarray(position, dtype=np.float32)
        self.max_power_db = max_power_db
        self.intent = intent
        self.intent_idx = self.INTENT_TYPES.index(intent)
        self._intent_counter = 0
        self._intent_duration = np.random.randint(20, 80)
        self._rng = np.random.default_rng(opponent_id)

    def get_action(
        self, radar_pos: np.ndarray, radar_action: np.ndarray
    ) -> dict:
        """Generate opponent action based on current intent."""
        rel = self.position - radar_pos
        range_km = float(np.linalg.norm(rel))

        action = {
            "freq_channels": np.zeros(64, dtype=np.float32),
            "power_db": 0.0,
            "jam_mode": 0,  # 0=none, 1=barrage, 2=spot, 3=sweep, 4=deceptive
        }

        if self.intent == "search_track":
            n_active = min(16, int(64 * (range_km / 200.0)))
            active = self._rng.choice(64, size=n_active, replace=False)
            action["freq_channels"][active] = 1.0
            action["power_db"] = self.max_power_db * 0.7
            action["jam_mode"] = self._rng.choice([2, 3])

        elif self.intent == "evade":
            n_active = 8
            active = self._rng.choice(64, size=n_active, replace=False)
            action["freq_channels"][active] = 1.0
            action["power_db"] = self.max_power_db * 0.4
            action["jam_mode"] = self._rng.choice([1, 4])

        elif self.intent == "fire_coordinate":
            n_active = 32
            active = self._rng.choice(64, size=n_active, replace=False)
            action["freq_channels"][active] = 1.0
            action["power_db"] = self.max_power_db * 0.95
            action["jam_mode"] = self._rng.choice([1, 2])

        elif self.intent == "silent_recon":
            n_active = 4
            active = self._rng.choice(64, size=n_active, replace=False)
            action["freq_channels"][active] = 0.3
            action["power_db"] = self.max_power_db * 0.15
            action["jam_mode"] = 0

        return action

    def update_intent(self) -> None:
        """Possibly switch to a new intent."""
        self._intent_counter += 1
        if self._intent_counter >= self._intent_duration:
            self._intent_counter = 0
            self._intent_duration = self._rng.integers(20, 80)
            probs = np.array([0.35, 0.25, 0.20, 0.20])
            new_idx = self._rng.choice(4, p=probs)
            self.intent_idx = new_idx
            self.intent = self.INTENT_TYPES[new_idx]
