# src/__init__.py
# ----------------------------------------------------------------
# Lazy imports — torch-dependent modules loaded on demand.
# Importing src directly is optional; each module can be imported
# individually without going through this package init.
# ----------------------------------------------------------------
try:
    from src.channel_model import ChannelModel, NOMACommunication
    from src.caching import CooperativeCacheManager, FrequencyWeightCaching
    from src.buffer import RolloutBuffer
    from src.metrics import MetricsTracker, compute_cost
    from src.env import MultiUAVMECEnv, DomainRandomiser
    from src.actor import GaussianOnlyActor
    from src.mappo_agent import MAPPOAgent, CentralisedCritic
except ImportError:
    pass  # torch not installed; modules still importable individually
