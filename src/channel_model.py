"""
Air-to-Ground (A2G) and UAV-to-UAV (U2U) Communication Model
==============================================================
Paper Reference: Section III.B (Eq. 4–13)
Target Journal: IEEE Internet of Things Journal

Models:
  - Distance & Elevation Angle: theta_{i,m}(t) = (180/pi) * arcsin(h_m(t) / d_{i,m}(t))
  - Elevation-Dependent Line-of-Sight (LoS) Probability:
      P_{i,m}^L(t) = 1 / (1 + a * exp(-b * (theta_{i,m}(t) - a)))
  - Average Path Loss:
      L_{i,m}(t) = L_{FS}(d_{i,m}(t)) + P_{i,m}^L(t) * eta_L + (1 - P_{i,m}^L(t)) * eta_N
  - Uplink Transmission Rate:
      R_{i,m}(t) = B_{i,m} * log2(1 + (P_i * g_{i,m}(t)) / (N_0 * B_{i,m}))
  - Uplink Uploading Delay:
      T_{i,m}^{up}(t) = D_i(t) / R_{i,m}(t)
  - U2U Coordination Link Rate & Delay:
      R_{m,n}^{u2u}(t) = B_{u2u} * log2(1 + (P_m^{u2u} * beta_0 * d_{m,n}^{-alpha_u}) / (N_0 * B_{u2u}))
      T_{m,n}^{u2u}(t) = S_{m,n} / R_{m,n}^{u2u}(t)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import (
        ENV_CONSTANT_A, ENV_CONSTANT_B, PATH_LOSS_EXPONENT,
        LOS_ATTENUATION_DB, NLOS_ATTENUATION_DB, NOISE_PSD,
        CARRIER_FREQUENCY, U2U_BANDWIDTH, U2U_TX_POWER_DBM, U2U_CONTROL_PKT_SIZE, dbm_to_watt
    )
except ImportError:
    ENV_CONSTANT_A = 9.6177
    ENV_CONSTANT_B = 0.1581
    PATH_LOSS_EXPONENT = 2.5
    LOS_ATTENUATION_DB = 0.0
    NLOS_ATTENUATION_DB = -20.0
    NOISE_PSD = 10**((-130 - 30) / 10)
    CARRIER_FREQUENCY = 2e9
    U2U_BANDWIDTH = 5e6
    U2U_TX_POWER_DBM = 24
    U2U_CONTROL_PKT_SIZE = 1e5
    def dbm_to_watt(dbm): return 10 ** ((dbm - 30) / 10)


class ChannelModel:
    """
    Air-to-Ground (A2G) and UAV-to-UAV (U2U) Communication Channel Manager.
    """

    def __init__(self, carrier_freq=CARRIER_FREQUENCY, noise_psd=NOISE_PSD):
        self.fc = carrier_freq
        self.c0 = 3e8  # Speed of light (m/s)
        self.noise_psd = noise_psd
        self.a = ENV_CONSTANT_A
        self.b = ENV_CONSTANT_B
        self.eta_L = LOS_ATTENUATION_DB
        self.eta_N = NLOS_ATTENUATION_DB

    def compute_distance_3d(self, pos1, pos2):
        """Compute 3D Euclidean distance between two nodes."""
        return max(np.linalg.norm(np.array(pos1) - np.array(pos2)), 1.0)

    def compute_elevation_angle(self, iot_pos, uav_pos):
        """Compute elevation angle theta_{i,m}(t) in degrees (Eq. 5)."""
        d_3d = self.compute_distance_3d(iot_pos, uav_pos)
        h_m = uav_pos[2] if len(uav_pos) > 2 else uav_pos[1]
        sin_val = np.clip(h_m / d_3d, 0.0, 1.0)
        return (180.0 / np.pi) * np.arcsin(sin_val)

    def compute_los_probability(self, theta):
        """Compute LoS probability P_{i,m}^L(t) (Eq. 6)."""
        return 1.0 / (1.0 + self.a * np.exp(-self.b * (theta - self.a)))

    def compute_free_space_path_loss(self, distance):
        """Compute Free Space Path Loss L_FS(d) in dB."""
        return 20.0 * np.log10(4.0 * np.pi * self.fc * distance / self.c0)

    def compute_average_path_loss(self, iot_pos, uav_pos):
        """Compute average A2G path loss L_{i,m}(t) in dB (Eq. 7)."""
        d = self.compute_distance_3d(iot_pos, uav_pos)
        theta = self.compute_elevation_angle(iot_pos, uav_pos)
        p_los = self.compute_los_probability(theta)

        l_fs = self.compute_free_space_path_loss(d)
        l_avg_db = l_fs + p_los * self.eta_L + (1.0 - p_los) * self.eta_N
        return l_avg_db

    def compute_channel_gain(self, iot_pos, uav_pos):
        """Compute A2G channel power gain g_{i,m}(t) = 10^(-L/10) (Eq. 8)."""
        l_avg_db = self.compute_average_path_loss(iot_pos, uav_pos)
        return 10.0 ** (-l_avg_db / 10.0)

    def compute_uplink_rate(self, iot_pos, uav_pos, tx_power, bandwidth):
        """Compute achievable A2G uplink transmission rate R_{i,m}(t) (Eq. 9)."""
        g_im = self.compute_channel_gain(iot_pos, uav_pos)
        noise_power = self.noise_psd * bandwidth
        snr = (tx_power * g_im) / max(noise_power, 1e-18)
        rate = bandwidth * np.log2(1.0 + snr)
        return max(rate, 1e3)  # Minimum fallback rate 1 kbps

    def compute_uploading_delay(self, data_size_bits, iot_pos, uav_pos, tx_power, bandwidth):
        """Compute A2G uploading delay T_{i,m}^{up}(t) = D_i / R_{i,m} (Eq. 10)."""
        rate = self.compute_uplink_rate(iot_pos, uav_pos, tx_power, bandwidth)
        return data_size_bits / rate

    def compute_u2u_rate(self, uav_pos1, uav_pos2, bandwidth=U2U_BANDWIDTH, tx_power_dbm=U2U_TX_POWER_DBM):
        """Compute U2U coordination transmission rate R_{m,n}^{u2u}(t) (Eq. 12)."""
        d_u2u = self.compute_distance_3d(uav_pos1, uav_pos2)
        tx_power_w = dbm_to_watt(tx_power_dbm)
        beta_0 = 1e-4  # Reference channel gain
        alpha_u = 2.0  # Aerial path-loss exponent
        channel_gain = beta_0 * (d_u2u ** (-alpha_u))
        noise_power = self.noise_psd * bandwidth
        snr = (tx_power_w * channel_gain) / max(noise_power, 1e-18)
        return bandwidth * np.log2(1.0 + snr)

    def compute_u2u_delay(self, uav_pos1, uav_pos2, pkt_size_bits=U2U_CONTROL_PKT_SIZE):
        """Compute U2U control exchange delay T_{m,n}^{u2u}(t) (Eq. 13)."""
        rate = self.compute_u2u_rate(uav_pos1, uav_pos2)
        return pkt_size_bits / max(rate, 1e3)
