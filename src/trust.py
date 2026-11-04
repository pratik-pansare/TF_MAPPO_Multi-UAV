"""
Certificate-Based Authentication Trust Model
=============================================
Paper Reference: Section III.E (Eq. 26–27)
Target Journal: IEEE Internet of Things Journal

Models:
  - Certificate C_m = {ID_m, PK_m, t_{iss}, t_{exp}, Sig_m} (Eq. 26)
  - Authentication-Derived Service Trust State tau_m(t) (Eq. 27):
      tau_m(t) = 1 if Certificate valid and t_{iss} <= t <= t_{exp}, else 0.
"""

import numpy as np
import sys
import os


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import CERTIFICATE_DURATION
except ImportError:
    CERTIFICATE_DURATION = 1000


class UAVCertificate:
    """Represents time-bound authentication credential C_m (Eq. 26)."""

    def __init__(self, uav_id, t_iss=0, t_exp=CERTIFICATE_DURATION):
        self.uav_id = uav_id
        self.t_iss = t_iss
        self.t_exp = t_exp
        self.signature_valid = True

    def verify(self, current_slot):
        """Verifies credential signature and time validity to return tau_m(t) (Eq. 27)."""
        valid = self.signature_valid and (self.t_iss <= current_slot <= self.t_exp)
        return 1 if valid else 0

    def renew(self, current_slot, extension_duration=CERTIFICATE_DURATION):
        """Renew certificate issue and expiration timestamp."""
        self.t_iss = current_slot
        self.t_exp = current_slot + extension_duration
        self.signature_valid = True


class TrustManager:
    """Manages authentication trust states tau_m(t) for all UAVs in the fleet."""

    def __init__(self, n_uavs, cert_duration=CERTIFICATE_DURATION):
        self.n_uavs = n_uavs
        self.cert_duration = cert_duration
        self.certificates = [
            UAVCertificate(uav_id=m, t_iss=0, t_exp=cert_duration)
            for m in range(n_uavs)
        ]
        self.tau = np.ones(n_uavs, dtype=int)
        self.trust_violations_blocked = 0

    def reset(self):
        """Reset authentication certificates for all UAVs."""
        for m in range(self.n_uavs):
            self.certificates[m].renew(current_slot=0, extension_duration=self.cert_duration)
        self.tau = np.ones(self.n_uavs, dtype=int)
        self.trust_violations_blocked = 0

    def update_trust_states(self, current_slot):
        """Update authentication trust tau_m(t) at each time slot (Eq. 27)."""
        for m in range(self.n_uavs):
            self.tau[m] = self.certificates[m].verify(current_slot)
            if self.tau[m] == 0:
                self.trust_violations_blocked += 1
        return self.tau.copy()

    def revoke_certificate(self, uav_id):
        """Revoke signature validity for a specific UAV."""
        if 0 <= uav_id < self.n_uavs:
            self.certificates[uav_id].signature_valid = False
            self.tau[uav_id] = 0
