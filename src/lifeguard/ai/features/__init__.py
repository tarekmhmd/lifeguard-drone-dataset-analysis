"""Per-track features and the rule-based submersion detector (Blueprint M6)."""

from lifeguard.ai.features.config import FeatureConfig
from lifeguard.ai.features.extractor import FeatureExtractor
from lifeguard.ai.features.submersion import PoolTrackView, SubmersionMonitor, SubmersionTimer

__all__ = ["FeatureConfig", "FeatureExtractor", "PoolTrackView", "SubmersionMonitor", "SubmersionTimer"]
