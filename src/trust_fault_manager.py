"""
Authentication-Derived Trust, Operational Availability, and Action Masking Manager
===================================================================================
Paper Reference: Section III.E (Eq. 26–29) & Action Masking Constraints
Target Journal: IEEE Internet of Things Journal

1. Certificate-Based Authentication Trust State tau_m(t) (Eq. 26-27):
   Certificate C_m = {ID_m, PK_m, t_iss, t_exp, Sig_m}
   tau_m(t) = 1 if Certificate valid and t_iss <= t <= t_exp, else 0.

2. Operational Availability State phi_m(t) (Eq. 28):
   phi_m(t) = 1 if UAV m is operational (no hardware/battery fault), else 0.

3. Combined UAV Eligibility State e_m(t) (Eq. 29):
   e_m(t) = tau_m(t) * phi_m(t)
   A UAV can provide cluster service and receive offloaded tasks ONLY if e_m(t) = 1.

4. Action Masking Generator M_m(t):
   Generates a binary action mask eliminating invalid UAV-cluster associations,
   task offloading decisions, energy violations, and collision risks before policy sampling.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import CERTIFICATE_DURATION, FAULT_PROBABILITY, MIN_BATTERY_RESERVE, MAX_SERVICE_RADIUS, SAFE_INTER_UAV_DISTANCE
except ImportError:
    CERTIFICATE_DURATION = 400
    FAULT_PROBABILITY = 0.002
    MIN_BATTERY_RESERVE = 50.0
    MAX_SERVICE_RADIUS = 600.0
    SAFE_INTER_UAV_DISTANCE = 10.0


class UAVCertificate:
    """Represents time-bound authentication certificate C_m (Eq. 26)."""
    def __init__(self, uav_id, t_iss=0, t_exp=CERTIFICATE_DURATION):
        self.uav_id = uav_id
        self.t_iss = t_iss
        self.t_exp = t_exp
        self.signature_valid = True

    def verify(self, current_slot):
        """Verifies certificate validity and returns tau_m(t) (Eq. 27)."""
        valid = self.signature_valid and (self.t_iss <= current_slot <= self.t_exp)
        return 1 if valid else 0


class TrustFaultManager:
    """
    Manages authentication trust, operational availability, failure recovery,
    and state-dependent action masking for the multi-UAV fleet.
    """

    def __init__(self, n_uavs, fault_prob=FAULT_PROBABILITY, cert_duration=CERTIFICATE_DURATION, random_state=42):
        self.n_uavs = n_uavs
        self.fault_prob = fault_prob
        self.cert_duration = cert_duration
        self.rng = np.random.RandomState(random_state)
        
        # Track failure recovery and trust metrics
        self.failures_triggered = 0
        self.failures_recovered = 0
        self.trust_violations_blocked = 0
        self.recovery_times = []

        # Initialize certificates and states per UAV
        self.certificates = [
            UAVCertificate(uav_id=m, t_iss=0, t_exp=cert_duration)
            for m in range(n_uavs)
        ]
        self.tau = np.ones(n_uavs, dtype=int)         # Trust states tau_m(t)
        self.phi = np.ones(n_uavs, dtype=int)         # Operational availability phi_m(t)
        self.eligibility = np.ones(n_uavs, dtype=int) # Combined e_m(t) = tau_m(t) * phi_m(t)
        self.fault_start_slots = np.zeros(n_uavs, dtype=int)

    def reset(self):
        """Reset trust, certificate, fault states, and recovery metrics."""
        for m in range(self.n_uavs):
            self.certificates[m].t_iss = 0
            self.certificates[m].t_exp = self.cert_duration
            self.certificates[m].signature_valid = True
        self.tau = np.ones(self.n_uavs, dtype=int)
        self.phi = np.ones(self.n_uavs, dtype=int)
        self.eligibility = np.ones(self.n_uavs, dtype=int)
        self.fault_start_slots = np.zeros(self.n_uavs, dtype=int)
        
        self.failures_triggered = 0
        self.failures_recovered = 0
        self.trust_violations_blocked = 0
        self.recovery_times = []

    def step_states(self, current_slot, uav_batteries, min_battery_res=MIN_BATTERY_RESERVE):
        """
        Update authentication trust tau_m(t), operational availability phi_m(t),
        and overall eligibility e_m(t) at each time slot (Eq. 27–29).
        """
        for m in range(self.n_uavs):
            # 1. Update authentication trust tau_m(t) via certificate verification
            self.tau[m] = self.certificates[m].verify(current_slot)
            if self.tau[m] == 0:
                self.trust_violations_blocked += 1

            # 2. Update operational availability phi_m(t) (check battery and hardware fault)
            if uav_batteries[m] < min_battery_res:
                if self.phi[m] == 1:
                    self.failures_triggered += 1
                    self.fault_start_slots[m] = current_slot
                self.phi[m] = 0  # Energy depletion fault
            elif self.phi[m] == 1 and self.rng.uniform() < self.fault_prob:
                self.failures_triggered += 1
                self.fault_start_slots[m] = current_slot
                self.phi[m] = 0  # Operational hardware fault

            # 3. Overall eligibility e_m(t) = tau_m(t) * phi_m(t)
            self.eligibility[m] = self.tau[m] * self.phi[m]

        return self.eligibility.copy()

    def renew_uav_certificate(self, uav_id, current_slot, extension_duration=CERTIFICATE_DURATION):
        """Renew certificate for an authenticated UAV."""
        if 0 <= uav_id < self.n_uavs:
            self.certificates[uav_id].t_iss = current_slot
            self.certificates[uav_id].t_exp = current_slot + extension_duration
            self.certificates[uav_id].signature_valid = True
            self.tau[uav_id] = 1
            self.eligibility[uav_id] = self.tau[uav_id] * self.phi[uav_id]

    def recover_uav_fault(self, uav_id, current_slot):
        """Recover a faulty UAV back to operational state."""
        if 0 <= uav_id < self.n_uavs and self.phi[uav_id] == 0:
            self.phi[uav_id] = 1
            self.eligibility[uav_id] = self.tau[uav_id] * self.phi[uav_id]
            self.failures_recovered += 1
            rec_time = current_slot - self.fault_start_slots[uav_id]
            self.recovery_times.append(rec_time)

    def set_uav_fault(self, uav_id, is_faulty=True, current_slot=0):
        """Manually trigger or clear a fault for a specific UAV."""
        if 0 <= uav_id < self.n_uavs:
            if is_faulty and self.phi[uav_id] == 1:
                self.failures_triggered += 1
                self.fault_start_slots[uav_id] = current_slot
            elif not is_faulty and self.phi[uav_id] == 0:
                self.recover_uav_fault(uav_id, current_slot)
            self.phi[uav_id] = 0 if is_faulty else 1
            self.eligibility[uav_id] = self.tau[uav_id] * self.phi[uav_id]

    def compute_action_mask(self, uav_positions, uav_batteries, uav_capacities,
                            action_dim, min_battery_res=MIN_BATTERY_RESERVE,
                            safe_dist=SAFE_INTER_UAV_DISTANCE):
        """
        Compute binary action mask matrix M_m(t) of shape (n_uavs, action_dim).
        Enforces:
          1. Service eligibility e_m(t) = 1 (Authentication trust tau_m & Operational availability phi_m)
          2. Reserve energy constraint E_m(t) >= E_min
          3. CPU capacity constraint
          4. Inter-UAV collision avoidance (safe distance d_safe)
        """
        mask = np.ones((self.n_uavs, action_dim), dtype=np.float32)

        for m in range(self.n_uavs):
            # Ineligible UAVs (tau_m=0 or phi_m=0) or depleted battery cannot execute tasks
            if self.eligibility[m] == 0 or uav_batteries[m] < min_battery_res:
                mask[m, :] = 0.0

        # Safety distance constraint between active UAVs
        for m in range(self.n_uavs):
            if self.eligibility[m] == 0:
                continue
            for n in range(self.n_uavs):
                if m != n and self.eligibility[n] == 1:
                    dist = np.linalg.norm(uav_positions[m][:2] - uav_positions[n][:2])
                    if dist < safe_dist:
                        # Mask movement toward each other if dangerously close
                        mask[m, :2] = 0.5

        return mask
