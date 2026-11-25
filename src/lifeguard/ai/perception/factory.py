"""Engine factory selecting the configured PerceptionEngine.

Engines are swapped by configuration only (mock -> pretrained -> fine-tuned, checklist I1: "no code changes
in system/ for the swap"). With ``model_registry`` set, every file loaded here is integrity-checked first.
"""
from __future__ import annotations

from pathlib import Path

from lifeguard.ai.config import AIConfig
from lifeguard.ai.perception.mock_engine import MockPerceptionEngine
from lifeguard.common.models import ModelRegistry
from lifeguard.contracts.enums import DetClass
from lifeguard.contracts.interfaces import PerceptionEngine, PoolGeometry


def _model_file(registry: ModelRegistry | None, path: Path | None, name: str | None) -> Path:
    """The file to load: by registry name, or a path that must be registered when a registry is in use."""
    if name is not None:
        if registry is None:
            raise ValueError(f"model {name!r} referenced by name but no model_registry is configured")
        return registry.verify(name)
    if path is None:
        raise ValueError("no model file configured")
    if registry is not None:
        registry.verify_file(path)
    return path


def create_engine(
    ai_config: AIConfig,
    cam_id: str,
    session_id: str,
    geometry: PoolGeometry | None = None,
) -> PerceptionEngine:
    """Build the configured engine.

    ``engine: mock`` replays a fixture (no geometry, decision C-09). ``engine: yolo`` builds the real
    pipeline and requires a ``PoolGeometry``; there is deliberately no silent stub fallback.

    Raises:
        ValueError: If the configuration block is missing, or a real engine has no geometry.
        ConfigError: If the model registry file is invalid.
        ModelLoadError: If a model file cannot be loaded; ``ModelIntegrityError`` (a subclass) if it is
            unregistered, missing or fails its checksum.
    """
    registry = ModelRegistry.load(ai_config.model_registry) if ai_config.model_registry else None

    if ai_config.engine == "mock":
        if ai_config.mock is None:
            raise ValueError("mock configuration is required")
        return MockPerceptionEngine(
            _model_file(registry, ai_config.mock.fixture_path, None),
            cam_id,
            session_id,
            ai_config.mock.on_exhausted,
        )

    if ai_config.yolo is None:
        raise ValueError("yolo configuration is required")
    if geometry is None:
        raise ValueError("engine 'yolo' requires a PoolGeometry")
    # Imported here so the mock path never needs the real-pipeline modules.
    from lifeguard.ai.detection.ultralytics_detector import UltralyticsDetector
    from lifeguard.ai.perception.real_engine import RealPerceptionEngine
    from lifeguard.ai.temporal.base import BehaviorClassifier
    from lifeguard.ai.temporal.rules import RuleBehaviorBaseline
    from lifeguard.ai.temporal.softmax import SoftmaxBehaviorModel

    behavior: BehaviorClassifier | None = None
    if ai_config.behavior == "rules":
        behavior = RuleBehaviorBaseline()
    elif ai_config.behavior == "softmax":
        behavior = SoftmaxBehaviorModel.load(
            _model_file(registry, ai_config.behavior_model, ai_config.behavior_model_name)
        )

    yolo = ai_config.yolo
    detector = UltralyticsDetector(
        _model_file(registry, yolo.weights, yolo.model),
        conf=yolo.conf,
        iou=yolo.iou,
        imgsz=yolo.imgsz,
        device=yolo.device,
        class_map={class_id: DetClass.PERSON for class_id in yolo.class_ids},
        distress_classes=yolo.distress_class_ids,
    )
    return RealPerceptionEngine(
        detector,
        geometry,
        cam_id,
        session_id,
        tracker_config=ai_config.tracker,
        feature_config=ai_config.features,
        behavior_classifier=behavior,
    )
