"""Phase 3 training and evaluation machinery.

Importing this package pulls in configuration, feature extraction, dataset
materialisation, and the baselines. It deliberately does not import
:mod:`voxshield.training._cnn`, so the two scikit-learn baselines remain
trainable in an environment with no PyTorch installed; the CNN module is reached
only when ``logmel_cnn`` is actually requested.
"""

from voxshield.training.calibration import (
    CALIBRATION_METHODS,
    Calibrator,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    build_calibrator,
    calibrator_from_dict,
    fit_calibrator,
)
from voxshield.training.config import (
    DEFAULT_ML_CONFIG_DIR,
    MODEL_FAMILIES,
    FeatureSpec,
    ModelSpec,
    TrainingConfig,
    TrainingError,
    TrainSpec,
    load_training_config,
    parse_training_config,
)
from voxshield.training.datasets import (
    SUBGROUP_AXES,
    FeatureMatrix,
    class_counts,
    load_feature_matrix,
    read_segment,
    require_clean_gates,
    resolve_rows,
)
from voxshield.training.features import (
    FeatureExtractor,
    cepstral_mean_normalise,
    dct_matrix,
    delta_features,
    mfcc_from_log_mel,
    pad_or_truncate,
)
from voxshield.training.models import (
    BaselineModel,
    FitSummary,
    LogisticBaseline,
    LogMelCNN,
    XGBoostBaseline,
    build_baseline,
)
from voxshield.training.registry import (
    ModelRegistry,
    ModelRegistryEntry,
    ModelRegistryError,
    load_registry,
)

__all__ = [
    "CALIBRATION_METHODS",
    "DEFAULT_ML_CONFIG_DIR",
    "MODEL_FAMILIES",
    "SUBGROUP_AXES",
    "BaselineModel",
    "Calibrator",
    "FeatureExtractor",
    "FeatureMatrix",
    "FeatureSpec",
    "FitSummary",
    "IdentityCalibrator",
    "IsotonicCalibrator",
    "LogMelCNN",
    "LogisticBaseline",
    "ModelRegistry",
    "ModelRegistryEntry",
    "ModelRegistryError",
    "ModelSpec",
    "PlattCalibrator",
    "TrainSpec",
    "TrainingConfig",
    "TrainingError",
    "XGBoostBaseline",
    "build_baseline",
    "build_calibrator",
    "calibrator_from_dict",
    "cepstral_mean_normalise",
    "class_counts",
    "dct_matrix",
    "delta_features",
    "fit_calibrator",
    "load_feature_matrix",
    "load_registry",
    "load_training_config",
    "mfcc_from_log_mel",
    "pad_or_truncate",
    "parse_training_config",
    "read_segment",
    "require_clean_gates",
    "resolve_rows",
]
