"""
UAV Fleet Energy Expenditure Model
==================================
Paper Reference: Section III.D (Eq. 23–25)
Target Journal: IEEE Internet of Things Journal

Models:
  1. Rotary-Wing Propulsion Power P_m^{fly}(v_m(t)):
       P_m^{fly}(v) = P_1 * (1 + 3*v^2 / v_{tip}^2) + P_2 * sqrt(sqrt(1 + v^4/(4*v_0^4)) - v^2/(2*v_0^2)) + 0.5*d_r*rho*s_r*A_r*v^3
  2. CPU Computation Energy:
       E_{i,m}^{comp}(t) = kappa_m * f_{i,m}(t)^3 * T_{i,m}^{exe}(t)
  3. U2U Communication Energy:
       E_m^{u2u}(t) = sum_n P_m^{u2u} * T_{m,n}^{u2u}(t)
  4. Total UAV Energy Expenditure:
       E_m^{UAV} = sum_t (P_m^{fly}(v_m(t)) * Delta_t + sum_i y_{i,m}(t) * E_{i,m}^{comp}(t) + E_m^{u2u}(t))
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import (
        P1_BLADE, P2_INDUCED, ROTOR_TIP_SPEED, MEAN_ROTOR_VELOCITY,
        DRAG_RATIO, AIR_DENSITY, ROTOR_SOLIDITY, DISC_AREA,
        EFFECTIVE_CAPACITANCE, U2U_TX_POWER_DBM, dbm_to_watt
    )
except ImportError:
    P1_BLADE = 79.86
    P2_INDUCED = 88.63
    ROTOR_TIP_SPEED = 120.0
    MEAN_ROTOR_VELOCITY = 4.03
    DRAG_RATIO = 0.6
    AIR_DENSITY = 1.225
    ROTOR_SOLIDITY = 0.05
    DISC_AREA = 0.503
    EFFECTIVE_CAPACITANCE = 1e-28
    U2U_TX_POWER_DBM = 24
    def dbm_to_watt(dbm): return 10 ** ((dbm - 30) / 10)


class EnergyManager:
    """Manages rotary-wing propulsion power, CPU computation energy, and U2U communication energy."""

    def __init__(self, kappa_m=EFFECTIVE_CAPACITANCE):
        self.kappa = kappa_m
        self.u2u_tx_power = dbm_to_watt(U2U_TX_POWER_DBM)

    def compute_propulsion_power(self, speed_m_s):
        """Compute rotary-wing propulsion power P_m^{fly}(v_m(t)) (Eq. 24)."""
        v = max(speed_m_s, 0.0)
        p_blade = P1_BLADE * (1.0 + 3.0 * (v ** 2) / (ROTOR_TIP_SPEED ** 2))
        
        term1 = (1.0 + (v ** 4) / (4.0 * (MEAN_ROTOR_VELOCITY ** 4))) ** 0.5
        term2 = (v ** 2) / (2.0 * (MEAN_ROTOR_VELOCITY ** 2))
        p_induced = P2_INDUCED * ((term1 - term2) ** 0.5)

        p_parasitic = 0.5 * DRAG_RATIO * AIR_DENSITY * ROTOR_SOLIDITY * DISC_AREA * (v ** 3)
        return p_blade + p_induced + p_parasitic

    def compute_piecewise_propulsion_energy(self, t_fly, t_hover, v_cr=15.0):
        """
        Compute piecewise rotary-wing propulsion energy:
          E_prop = P_m^{fly}(v_cr) * t_fly + P_m^{hov} * t_hover
        where P_m^{hov} = P_m^{fly}(0).
        """
        p_fly = self.compute_propulsion_power(v_cr)
        p_hov = self.compute_propulsion_power(0.0)
        return float(p_fly * max(0.0, t_fly) + p_hov * max(0.0, t_hover))

    def compute_computation_energy(self, allocated_cpu_hz, execution_delay_s):
        """Compute onboard CPU computation energy E_{i,m}^{comp}(t) = kappa * f^3 * T^{exe} (Eq. 24)."""
        return self.kappa * (allocated_cpu_hz ** 3) * execution_delay_s

    def compute_u2u_energy(self, u2u_delay_s):
        """Compute U2U communication energy E_m^{u2u}(t) = P_{u2u} * T^{u2u} (Eq. 23)."""
        return self.u2u_tx_power * u2u_delay_s

    def compute_step_energy(self, speed_m_s, allocated_cpu_hz, execution_delay_s, u2u_delay_s, delta_t=1.0):
        """Compute total UAV energy consumed over one timestep (Eq. 24)."""
        e_fly = self.compute_propulsion_power(speed_m_s) * delta_t
        e_comp = self.compute_computation_energy(allocated_cpu_hz, execution_delay_s)
        e_u2u = self.compute_u2u_energy(u2u_delay_s)
        return e_fly + e_comp + e_u2u
