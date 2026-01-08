"""
Y2M - YOLO to Mobile Conversion Pipeline

A modular CLI utility that converts YOLO .pt models to optimized 
TensorFlow Lite format for mobile deployment.
"""

__version__ = "1.0.0"
__author__ = "Y2M Pipeline"

from .converter import YOLOConverter
from .optimizer import ModelOptimizer
from .utils import validate_model_path, create_metadata

__all__ = [
    "YOLOConverter",
    "ModelOptimizer", 
    "validate_model_path",
    "create_metadata",
]
