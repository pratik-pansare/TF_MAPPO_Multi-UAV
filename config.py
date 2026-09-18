"""
Configuration — Trust- and Fault-Aware Action-Masked MAPPO
===========================================================
Paper: Trust- and Fault-Aware Action-Masked MAPPO for Dynamic Multi-UAV
       Enabled IoT Mobile Edge Computing with Adaptive Task Offloading
       and Trajectory Optimization
Target Journal: IEEE Internet of Things Journal
"""

import numpy as np

# ============================================================
# 1. AREA & DEPLOYMENT MODEL
# ============================================================
AREA_RADIUS   = 500         # Service area radius (m)
NUM_USERS     = 100         # Number of static ground IoT devices (N)
MAX_NUM_USERS = 180
NUM_UAVS      = 5           # Number of rotary-wing UAV-MEC servers (M)
MAX_UAVS      = 8
MIN_UAVS      = 3
NUM_CLUSTERS  = 5           # Number of pre-deployment IoT service clusters (C)
NUM_JAMMERS   = 0           # Jammer module disabled for this paper

# ============================================================
# 2. HETEROGENEOUS UAV PROFILES
# ============================================================
UAV_PROFILES = [
    {'cpu_freq': 3.0e9, 'bandwidth': 15e6, 'tx_power_dbm': 27, 'altitude': 110, 'max_displacement': 55},
    {'cpu_freq': 2.0e9, 'bandwidth': 10e6, 'tx_power_dbm': 25, 'altitude': 100, 'max_displacement': 50},
    {'cpu_freq': 1.2e9, 'bandwidth': 8e6,  'tx_power_dbm': 22, 'altitude': 85,  'max_displacement': 60},
    {'cpu_freq': 2.5e9, 'bandwidth': 12e6, 'tx_power_dbm': 26, 'altitude': 105, 'max_displacement': 52},
    {'cpu_freq': 1.8e9, 'bandwidth': 9e6,  'tx_power_dbm': 24, 'altitude': 95,  'max_displacement': 58},
    {'cpu_freq': 2.8e9, 'bandwidth': 14e6, 'tx_power_dbm': 26, 'altitude': 108, 'max_displacement': 53},
    {'cpu_freq': 1.5e9, 'bandwidth': 8.5e6,'tx_power_dbm': 23, 'altitude': 90,  'max_displacement': 62},
    {'cpu_freq': 3.2e9, 'bandwidth': 16e6, 'tx_power_dbm': 28, 'altitude': 115, 'max_displacement': 50},
]
NUM_UAV_CAPABILITY_DIMS = 5

# Rotary-Wing Propulsion Model Parameters (Paper Section III.D)
P1_BLADE            = 79.86  # Blade profile power constant
P2_INDUCED          = 88.63  # Induced power constant
ROTOR_TIP_SPEED     = 120.0  # Rotor tip speed (m/s)
MEAN_ROTOR_VELOCITY = 4.03   # Mean rotor induced velocity (m/s)
DRAG_RATIO          = 0.6
AIR_DENSITY         = 1.225  # Air density (kg/m^3)
ROTOR_SOLIDITY      = 0.05
DISC_AREA           = 0.503

# ============================================================
# 3. AIR-TO-GROUND (A2G) COMMUNICATION MODEL
# ============================================================
RBS_BANDWIDTH          = 10e6    # Bandwidth (Hz)
IOT_TX_POWER_DBM       = 20      # IoT transmission power (dBm)
ENV_CONSTANT_A         = 9.6177  # Environmental constant a
ENV_CONSTANT_B         = 0.1581  # Environmental constant b
PATH_LOSS_EXPONENT     = 2.5
LOS_ATTENUATION_DB     = 0.0     # Excessive LoS path loss (dB)
NLOS_ATTENUATION_DB    = -20.0   # Excessive NLoS path loss (dB)
NOISE_PSD_DBM          = -130    # Noise PSD (dBm/Hz)
CARRIER_FREQUENCY      = 2e9     # Carrier frequency (2 GHz)
RICIAN_K_FACTOR        = 10
CHANNEL_POWER_GAIN_REF = 1e-4

# UAV-to-UAV (U2U) Coordination Link Constants
U2U_BANDWIDTH          = 5e6     # 5 MHz U2U control link bandwidth
U2U_TX_POWER_DBM       = 24      # 24 dBm U2U transmit power
U2U_CONTROL_PKT_SIZE   = 1e5     # Size of control info exchanged (bits)

# ============================================================
# 4. TASK & COMPUTATION MODEL
# ============================================================
IOT_CPU_FREQUENCY      = 0.5e9   # IoT local CPU frequency (Hz)
EFFECTIVE_CAPACITANCE  = 1e-28   # Switched capacitance coefficient kappa_m

TASK_DATA_SIZE_RANGE   = (0.5e6, 5e6)  # Task data size range (bits)
TASK_DATA_SIZE         = 2.75e6
TASK_CPU_CYCLES_RANGE  = (0.1e9, 1e9)  # Required CPU cycles per task
TASK_DEADLINE_RANGE    = (0.2, 2.0)    # Task deadline range (seconds)
TASK_GENERATION_PROB   = 0.7           # Task generation probability per slot

# ============================================================
# 5. TRUST & FAULT MODEL
# ============================================================
CERTIFICATE_DURATION   = 400     # Certificate duration (slots)
FAULT_PROBABILITY      = 0.002   # Per-slot UAV operational fault probability
MIN_BATTERY_RESERVE    = 50.0    # Reserve energy threshold (Joules)
MAX_SERVICE_RADIUS     = 600.0   # Maximum UAV service coverage radius (m)

# ============================================================
# 6. TIME & MOBILITY MODEL
# ============================================================
TOTAL_TIME             = 500
NUM_TIME_SLOTS         = 500
SLOT_DURATION          = TOTAL_TIME / NUM_TIME_SLOTS  # Delta_t = 1.0 s

# ============================================================
# 7. ACTION MASKING & CONSTRAINTS
# ============================================================
SAFE_INTER_UAV_DISTANCE = 10.0   # Minimum safe distance d_safe (m)
MIN_TASK_COMPLETION_RATIO = 0.85 # Constraint C11 (eta_min)

# ============================================================
# 8. MAPPO TRAINING HYPERPARAMETERS
# ============================================================
MAPPO_LR_ACTOR         = 3e-4
MAPPO_LR_CRITIC        = 1e-4
MAPPO_GAMMA            = 0.99
MAPPO_GAE_LAMBDA       = 0.95
MAPPO_CLIP_EPS         = 0.2
MAPPO_ENTROPY_COEF     = 0.01
MAPPO_VALUE_COEF       = 0.5
MAPPO_MAX_GRAD_NORM    = 0.5
MAPPO_ROLLOUT_LENGTH   = 500
MAPPO_NUM_MINI_BATCHES = 5
MAPPO_PPO_EPOCHS       = 4
MAPPO_HIDDEN_DIM       = 256
MAPPO_CRITIC_HIDDEN_DIM = 512

TOTAL_TIMESTEPS        = 5_000_000
EVAL_INTERVAL          = 50_000
CHECKPOINT_INTERVAL    = 500_000
LOG_INTERVAL           = 20_000
NUM_SEEDS              = 5       # 5 random seeds for 95% CI reporting

# ============================================================
# 9. OBJECTIVE WEIGHTS (Problem P1)
# ============================================================
ALPHA_LATENCY          = 0.5     # Latency weight
BETA_ENERGY            = 0.5     # Energy weight
OMEGA_E                = 0.5     # Energy weight in Objective P1
OMEGA_T                = 0.5     # Latency weight in Objective P1
LAMBDA_W               = 0.5     # Workload coefficient in IoT clustering

# Model Feature Flags
USE_DIFFUSION          = False   # Disabled: Gaussian Action-Masked Actor used
USE_COOP_CACHE         = False   # Disabled: Caching removed
USE_LEARNED_TRAJ       = True    # Enabled: Adaptive Trajectory Optimization
ABLATION_MODE          = "full"
CHECKPOINT_DIR         = "results/checkpoints"
LOG_DIR                = "results/logs"

# ============================================================
# HELPER FUNCTIONS & DERIVED CONSTANTS
# ============================================================
def dbm_to_watt(dbm):
    return 10 ** ((dbm - 30) / 10)

def db_to_linear(db):
    return 10 ** (db / 10)

IOT_TX_POWER     = dbm_to_watt(IOT_TX_POWER_DBM)
NOISE_PSD        = dbm_to_watt(NOISE_PSD_DBM)
LOS_ATTENUATION  = db_to_linear(LOS_ATTENUATION_DB)
NLOS_ATTENUATION = db_to_linear(NLOS_ATTENUATION_DB)
UAV_ALT_MIN      = 50.0
UAV_ALT_MAX      = 150.0
