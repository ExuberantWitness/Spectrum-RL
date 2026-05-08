"""Physical channel, propagation, and radar equation models."""

import numpy as np
import torch


class RadarChannel:
    """Models the radar channel: path loss, RCS response, jamming effects."""

    def __init__(self, carrier_freq_ghz: float = 10.0, bandwidth_mhz: float = 10.0):
        self.fc = carrier_freq_ghz  # GHz
        self.bw = bandwidth_mhz * 1e6  # Hz
        self.c = 3e8  # speed of light
        self.wavelength = self.c / (self.fc * 1e9)
        self.k = 1.38e-23  # Boltzmann
        self.t0 = 290.0  # standard noise temp

    def path_loss_db(self, range_km: float) -> float:
        """Two-way free-space path loss in dB."""
        r = range_km * 1e3
        if r < 1.0:
            r = 1.0
        return 20 * np.log10((4 * np.pi * r) / self.wavelength)

    def radar_range_equation(
        self,
        pt_db: float,
        gain_db: float,
        rcs_dbsm: float,
        range_km: float,
        loss_db: float = 6.0,
    ) -> float:
        """Single-pulse SNR via radar range equation."""
        pl_db = self.path_loss_db(range_km)
        ktb = 10 * np.log10(self.k * self.t0 * self.bw)
        snr_db = pt_db + 2 * gain_db + rcs_dbsm - pl_db - ktb - loss_db
        return snr_db

    def detection_probability(self, snr_db: float, pfa: float = 1e-6) -> float:
        """Pd from SNR (Albersheim approximation for Swerling 1)."""
        snr_lin = 10 ** (snr_db / 10.0)
        a = np.log(0.62 / pfa)
        b = np.log(1.0 / (1.0 - 0.5))  # Pd~0.5 reference
        if snr_lin <= 0:
            return pfa
        det = (snr_lin / a) ** 0.12 * b
        pd = 1.0 / (1.0 + np.exp(-det + 0.5))
        return np.clip(pd, pfa, 0.999)

    def jamming_to_signal_ratio(
        self, jammer_power_db: float, jammer_gain_db: float,
        jammer_range_km: float, radar_rcs_dbsm: float,
        target_range_km: float,
    ) -> float:
        """J/S ratio at the radar receiver."""
        j_pl = self.path_loss_db(jammer_range_km)  # one-way
        t_pl = self.path_loss_db(target_range_km)  # two-way
        jsr_db = jammer_power_db + jammer_gain_db - j_pl - (radar_rcs_dbsm - t_pl)
        return jsr_db

    def channel_capacity(self, snr_db: float) -> float:
        """Shannon capacity in bps/Hz."""
        snr_lin = 10 ** (snr_db / 10.0)
        return np.log2(1.0 + snr_lin)


class PropagationModel:
    """Time-varying propagation model with fading and interference."""

    def __init__(
        self,
        num_channels: int = 64,
        coherence_time: int = 20,
        seed: int = 42,
    ):
        self.num_channels = num_channels
        self.coherence_time = coherence_time
        self.rng = np.random.default_rng(seed)
        self._channel_state = self.rng.normal(0, 1, num_channels) + \
            1j * self.rng.normal(0, 1, num_channels)
        self._step_counter = 0

    def step(self) -> np.ndarray:
        """Update and return channel gains (dB)."""
        self._step_counter += 1
        if self._step_counter % self.coherence_time == 0:
            innovation = 0.1 * (
                self.rng.normal(0, 1, self.num_channels)
                + 1j * self.rng.normal(0, 1, self.num_channels)
            )
            self._channel_state = (0.95 * self._channel_state + innovation)
        gains_linear = np.abs(self._channel_state) ** 2
        gains_db = 10 * np.log10(np.clip(gains_linear, 1e-10, None))
        return gains_db

    def apply_interference(
        self,
        channel_gains_db: np.ndarray,
        interference_mask: np.ndarray,
        interference_power_db: float,
    ) -> np.ndarray:
        """Apply external interference on selected channels."""
        interfered = channel_gains_db.copy()
        mask_bool = interference_mask > 0.5
        interfered[mask_bool] = 10 * np.log10(
            np.clip(
                10 ** (channel_gains_db[mask_bool] / 10.0)
                + 10 ** (interference_power_db / 10.0),
                1e-10, None,
            )
        )
        return interfered
