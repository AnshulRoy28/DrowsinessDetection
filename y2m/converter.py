"""
Y2M Converter Module

Core conversion engine handling PT to ONNX to TFLite translation.
Uses Ultralytics built-in export for reliability.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class YOLOConverter:
    """
    Handles the conversion of YOLO models from PyTorch to TFLite format.
    
    Conversion path: .pt -> .onnx -> SavedModel -> .tflite
    """
    
    def __init__(self, model_path: str, output_dir: str):
        """
        Initialize the converter.
        
        Args:
            model_path: Path to the input .pt model file
            output_dir: Directory for output files
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.model = None
        self.model_info = {}
        
    def load_model(self) -> bool:
        """
        Load the YOLO model from the .pt file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            from ultralytics import YOLO
            
            logger.info(f"Loading model: {self.model_path.name}")
            self.model = YOLO(str(self.model_path))
            
            # Extract model info
            if hasattr(self.model, 'names'):
                self.model_info['class_names'] = (
                    list(self.model.names.values()) 
                    if isinstance(self.model.names, dict) 
                    else list(self.model.names)
                )
            if hasattr(self.model, 'task'):
                self.model_info['task'] = self.model.task
                
            logger.info(f"[OK] Model loaded successfully")
            return True
            
        except ImportError:
            logger.error("Ultralytics not installed. Run: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def export_to_onnx(
        self, 
        opset_version: int = 12,
        input_size: Tuple[int, int] = (640, 640),
        simplify: bool = True
    ) -> Optional[Path]:
        """
        Export the model to ONNX format.
        
        Args:
            opset_version: ONNX opset version (12 recommended for mobile)
            input_size: Fixed input dimensions (height, width)
            simplify: Whether to simplify the ONNX graph
            
        Returns:
            Path to the exported ONNX file, or None if failed
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return None
            
        try:
            logger.info(f"Converting to ONNX (opset={opset_version}, size={input_size})...")
            
            # Use Ultralytics built-in export
            onnx_path = self.model.export(
                format='onnx',
                opset=opset_version,
                imgsz=input_size,
                simplify=simplify,
                dynamic=False  # Fixed dimensions for mobile compatibility
            )
            
            logger.info(f"[OK] ONNX export complete: {Path(onnx_path).name}")
            return Path(onnx_path)
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return None
    
    def export_to_tflite(
        self,
        input_size: Tuple[int, int] = (640, 640)
    ) -> Optional[Path]:
        """
        Export the model directly to TFLite format.
        Uses Ultralytics internal conversion via SavedModel.
        
        Args:
            input_size: Fixed input dimensions (height, width)
            
        Returns:
            Path to the exported TFLite file, or None if failed
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return None
            
        try:
            logger.info(f"Converting to TFLite (size={input_size})...")
            logger.info("This may take a few minutes...")
            
            # Use Ultralytics built-in TFLite export
            # This handles PT -> ONNX -> TF SavedModel -> TFLite internally
            tflite_path = self.model.export(
                format='tflite',
                imgsz=input_size,
                dynamic=False  # Fixed dimensions for mobile
            )
            
            # Move to output directory with proper naming
            source_path = Path(tflite_path)
            dest_name = f"{self.model_path.stem}_float32.tflite"
            dest_path = self.output_dir / dest_name
            
            # Copy file to output location
            import shutil
            self.output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest_path)
            
            logger.info(f"[OK] TFLite export complete: {dest_name}")
            logger.info(f"  Size: {dest_path.stat().st_size / 1024 / 1024:.2f} MB")
            
            return dest_path
            
        except ImportError as e:
            if 'tensorflow' in str(e).lower():
                logger.error("TensorFlow not installed. Run: pip install tensorflow-cpu")
            else:
                logger.error(f"Missing dependency: {e}")
            return None
        except Exception as e:
            logger.error(f"TFLite export failed: {e}")
            return None
    
    def get_class_names(self) -> list:
        """Get the class names from the loaded model."""
        return self.model_info.get('class_names', [])
    
    def get_model_info(self) -> dict:
        """Get all extracted model information."""
        return self.model_info
