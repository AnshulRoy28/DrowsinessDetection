"""
Y2M Utilities Module

Handles file validation, metadata generation, and cleanup operations.
"""

import os
import json
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from datetime import datetime

# Error codes for specific failure cases
ERR_FILE_NOT_FOUND = "ERR_FILE_NOT_FOUND"
ERR_INVALID_EXTENSION = "ERR_INVALID_EXTENSION"
ERR_FILE_EMPTY = "ERR_FILE_EMPTY"
ERR_READ_PERMISSION = "ERR_READ_PERMISSION"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def validate_model_path(model_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate the input model path.
    
    Args:
        model_path: Path to the .pt model file
        
    Returns:
        Tuple of (is_valid, error_code or None)
    """
    path = Path(model_path)
    
    # Check if file exists
    if not path.exists():
        logger.error(f"File not found: {model_path}")
        return False, ERR_FILE_NOT_FOUND
    
    # Check file extension
    if path.suffix.lower() != '.pt':
        logger.error(f"Invalid file extension: {path.suffix}. Expected .pt")
        return False, ERR_INVALID_EXTENSION
    
    # Check file size > 0
    if path.stat().st_size == 0:
        logger.error(f"File is empty: {model_path}")
        return False, ERR_FILE_EMPTY
    
    # Check read permissions
    if not os.access(path, os.R_OK):
        logger.error(f"Cannot read file (permission denied): {model_path}")
        return False, ERR_READ_PERMISSION
    
    logger.info(f"[OK] Model validated: {path.name} ({path.stat().st_size / 1024 / 1024:.2f} MB)")
    return True, None


def create_output_directory(output_path: str) -> Path:
    """
    Create the output directory structure.
    
    Args:
        output_path: Path to the output directory
        
    Returns:
        Path object for the created directory
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[OK] Output directory ready: {output_dir}")
    return output_dir


def create_metadata(
    model_name: str,
    output_dir: Path,
    class_names: Optional[list] = None,
    input_size: Tuple[int, int] = (640, 640),
    quantized: bool = False,
    extra_info: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Generate metadata.json with model information.
    
    Args:
        model_name: Original model filename
        output_dir: Directory to save metadata
        class_names: List of class names from the model
        input_size: Input dimensions (height, width)
        quantized: Whether Int8 quantization was applied
        extra_info: Additional metadata to include
        
    Returns:
        Path to the created metadata.json file
    """
    metadata = {
        "model_name": model_name,
        "conversion_date": datetime.now().isoformat(),
        "input_size": list(input_size),
        "input_format": "NHWC",  # TFLite uses channels-last
        "quantized": quantized,
        "output_files": {
            "float32": f"{Path(model_name).stem}_float32.tflite",
        },
        "y2m_version": "1.0.0"
    }
    
    if quantized:
        metadata["output_files"]["int8"] = f"{Path(model_name).stem}_int8.tflite"
    
    if class_names:
        metadata["class_names"] = class_names
        metadata["num_classes"] = len(class_names)
    
    if extra_info:
        metadata.update(extra_info)
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"[OK] Metadata saved: {metadata_path}")
    return metadata_path


def cleanup_intermediate_files(directory: Path, extensions: list = ['.onnx']) -> int:
    """
    Remove intermediate files to save disk space.
    
    Args:
        directory: Directory to clean
        extensions: List of file extensions to remove
        
    Returns:
        Number of files deleted
    """
    deleted_count = 0
    for ext in extensions:
        for file in directory.glob(f"*{ext}"):
            try:
                file.unlink()
                logger.info(f"[OK] Cleaned up: {file.name}")
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Could not delete {file.name}: {e}")
    
    return deleted_count


def get_model_info(model) -> Dict[str, Any]:
    """
    Extract model information from YOLO model.
    
    Args:
        model: Loaded YOLO model object
        
    Returns:
        Dictionary with model information
    """
    info = {}
    
    try:
        # Get class names if available
        if hasattr(model, 'names'):
            info['class_names'] = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
        
        # Get model task type
        if hasattr(model, 'task'):
            info['task'] = model.task
            
    except Exception as e:
        logger.warning(f"Could not extract all model info: {e}")
    
    return info
