"""
UAV Operational Fault & Availability Model
===========================================
Paper Reference: Section III.E (Eq. 28–29)
Target Journal: IEEE Internet of Things Journal

Models:
  - Operational Availability State phi_m(t) (Eq. 28):
      phi_m(t) = 1 if UAV m is operational (no hardware/battery fault), else 0.
  - Combined Service Eligibility State e_m(t) (Eq. 29):
      e_m(t) = tau_m(t) * phi_m(t)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import FAULT_PROBABILITY, MIN_BATTERY_RESERVE
except ImportError:
    FAULT_PROBABILITY = 0.001
    MIN_BATTERY_RESERVE = 50.0


class FaultManager:
    """
    Manages operational availability states phi_m(t), failure events,
    failure recovery, and combined eligibility states e_m(t).
    """

    def __init__(self, n_uavs, fault_prob=FAULT_PROBABILITY, random_state=42):
        self.n_uavs = n_uavs
        self.fault_prob = fault_prob
        self.rng = np.random.RandomState(random_state)

        self.phi = np.ones(n_uavs, dtype=int)
        self.eligibility = np.ones(n_uavs, dtype=int)
        self.fault_start_slots = np.zeros(n_uavs, dtype=int)

        self.failures_triggered = 0
        self.failures_recovered = 0
        self.recovery_times = []

    def reset(self, seed=None):
        """Reset fault states and failure tracking counters."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.phi = np.ones(self.n_uavs, dtype=int)
        self.eligibility = np.ones(self.n_uavs, dtype=int)
        self.fault_start_slots = np.zeros(self.n_uavs, dtype=int)

        self.failures_triggered = 0
        self.failures_recovered = 0
        self.recovery_times = []

    def update_fault_and_eligibility(self, current_slot, tau_states, uav_batteries, min_battery_res=MIN_BATTERY_RESERVE):
        """
        Update operational availability phi_m(t) and combined eligibility e_m(t) = tau_m * phi_m (Eq. 28–29).
        """
        for m in range(self.n_uavs):
            # Check battery depletion fault
            if uav_batteries[m] < min_battery_res:
                if self.phi[m] == 1:
                    self.failures_triggered += 1
                    self.fault_start_slots[m] = current_slot
                self.phi[m] = 0
            elif self.phi[m] == 0:
                # Transient operational fault: recover after self-healing interval
                elapsed_fault = current_slot - self.fault_start_slots[m]
                if elapsed_fault >= 5 and self.rng.uniform() < 0.20:
                    self.recover_uav(m, current_slot, tau_states)
            elif self.phi[m] == 1 and self.rng.uniform() < self.fault_prob:
                # Stochastic hardware / operational fault
                self.failures_triggered += 1
                self.fault_start_slots[m] = current_slot
                self.phi[m] = 0

            # Combined eligibility e_m(t) = tau_m(t) * phi_m(t)
            self.eligibility[m] = tau_states[m] * self.phi[m]

        return self.eligibility.copy()

    def recover_uav(self, uav_id, current_slot, tau_states):
        """Recover a failed UAV back to operational state."""
        if 0 <= uav_id < self.n_uavs and self.phi[uav_id] == 0:
            self.phi[uav_id] = 1
            self.eligibility[uav_id] = tau_states[uav_id] * self.phi[uav_id]
            self.failures_recovered += 1
            rec_time = current_slot - self.fault_start_slots[uav_id]
            self.recovery_times.append(rec_time)
            return True
        return False
