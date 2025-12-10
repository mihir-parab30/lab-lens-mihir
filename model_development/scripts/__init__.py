"""
Gemini 2.0 Flash Exp Training Module for Medical Text Summarization
Complete MLOps implementation with experiment tracking, validation, bias detection, and more
"""

from .gemini_model import (
    GeminiSummarizer,
    load_gemini_model
)
from .train_gemini import GeminiTrainer
from .gemini_inference import GeminiInference

# Optional imports for advanced features (suppress warnings)
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=UserWarning)
    
    try:
        from .mlflow_tracking import MLflowTracker, get_best_run
        MLFLOW_AVAILABLE = True
    except ImportError:
        MLFLOW_AVAILABLE = False
        MLflowTracker = None
        get_best_run = None

    try:
        from .model_validation import ModelValidator
        VALIDATION_AVAILABLE = True
    except ImportError:
        VALIDATION_AVAILABLE = False
        ModelValidator = None

    try:
        from .hyperparameter_tuning import HyperparameterTuner
        TUNING_AVAILABLE = True
    except ImportError:
        TUNING_AVAILABLE = False
        HyperparameterTuner = None

    try:
        from .model_bias_detection import ModelBiasDetector
        BIAS_DETECTION_AVAILABLE = True
    except ImportError:
        BIAS_DETECTION_AVAILABLE = False
        ModelBiasDetector = None

    try:
        from .sensitivity_analysis import SensitivityAnalyzer
        SENSITIVITY_AVAILABLE = True
    except ImportError:
        SENSITIVITY_AVAILABLE = False
        SensitivityAnalyzer = None

    try:
        from .model_registry import ModelRegistry
        REGISTRY_AVAILABLE = True
    except ImportError:
        REGISTRY_AVAILABLE = False
        ModelRegistry = None

    try:
        from .train_with_tracking import CompleteModelTrainer
        COMPLETE_TRAINER_AVAILABLE = True
    except ImportError:
        COMPLETE_TRAINER_AVAILABLE = False
        CompleteModelTrainer = None

__all__ = [
    'GeminiSummarizer',
    'GeminiTrainer',
    'GeminiInference',
    'load_gemini_model',
]

# Add optional exports if available
if MLFLOW_AVAILABLE:
    __all__.extend(['MLflowTracker', 'get_best_run'])
if VALIDATION_AVAILABLE:
    __all__.append('ModelValidator')
if TUNING_AVAILABLE:
    __all__.append('HyperparameterTuner')
if BIAS_DETECTION_AVAILABLE:
    __all__.append('ModelBiasDetector')
if SENSITIVITY_AVAILABLE:
    __all__.append('SensitivityAnalyzer')
if REGISTRY_AVAILABLE:
    __all__.append('ModelRegistry')
if COMPLETE_TRAINER_AVAILABLE:
    __all__.append('CompleteModelTrainer')


