"""
TFLite Inference Test for Drowsiness Detection Model

This script loads the converted TFLite model and runs inference on webcam.
Uses rolling average of past 5 inferences for stable state determination.

Usage:
    python test_tflite_inference.py
"""

import cv2
import numpy as np
from pathlib import Path
from collections import deque

# Try TensorFlow Lite runtime
try:
    import tensorflow as tf
    interpreter_class = tf.lite.Interpreter
    print("Using TensorFlow Lite")
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        interpreter_class = tflite.Interpreter
        print("Using TFLite Runtime")
    except ImportError:
        print("ERROR: Neither tensorflow nor tflite-runtime installed.")
        print("Install with: pip install tensorflow")
        exit(1)


class TFLiteDetector:
    """TFLite model wrapper for drowsiness detection."""
    
    def __init__(self, model_path: str):
        self.interpreter = interpreter_class(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Get input shape
        self.input_shape = self.input_details[0]['shape']
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        
        # Class names
        self.class_names = ['drowsy', 'notdrowsy']
        
        print(f"Model loaded: {model_path}")
        print(f"Input shape: {self.input_shape}")
        print(f"Classes: {self.class_names}")
    
    def preprocess(self, frame):
        """Preprocess frame for inference."""
        # Resize to model input size
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize to 0-1 and add batch dimension
        normalized = rgb.astype(np.float32) / 255.0
        batched = np.expand_dims(normalized, axis=0)
        return batched
    
    def detect(self, frame):
        """Run detection on a frame. Returns list of (class_name, confidence, bbox)."""
        # Preprocess
        input_data = self.preprocess(frame)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        detections = []
        
        # Parse YOLO output format (typically [batch, num_detections, 4+num_classes] or similar)
        # The exact format depends on how the model was exported
        if len(output_data.shape) == 3:
            # Shape: [1, num_boxes, 4 + num_classes] - standard YOLO format
            predictions = output_data[0]  # Remove batch dim
            
            # YOLO output: [x_center, y_center, width, height, class1_conf, class2_conf, ...]
            for pred in predictions:
                if len(pred) >= 6:  # x, y, w, h, class1, class2
                    x, y, w, h = pred[:4]
                    class_scores = pred[4:]
                    
                    # Get best class
                    class_id = np.argmax(class_scores)
                    confidence = class_scores[class_id]
                    
                    if confidence > 0.25:  # Confidence threshold
                        class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                        detections.append((class_name, float(confidence), (x, y, w, h)))
        
        elif len(output_data.shape) == 2:
            # Possible classification output [batch, num_classes]
            scores = output_data[0]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.25:
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                detections.append((class_name, float(confidence), None))
        
        return detections


def test_webcam(detector):
    """Run real-time drowsiness detection on webcam with rolling average."""
    print("\n" + "=" * 50)
    print("   WEBCAM INFERENCE TEST (TFLite)")
    print("=" * 50)
    print("Press 'q' to quit\n")

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return False

    inference_count = 0
    detection_history = deque(maxlen=5)
    last_reported_state = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read from webcam")
            break

        # Run inference
        detections = detector.detect(frame)
        inference_count += 1

        # Draw detections on frame
        display_frame = frame.copy()
        current_detection = None
        
        for det in detections:
            class_name, confidence, bbox = det
            if current_detection is None or confidence > current_detection[1]:
                current_detection = (class_name, confidence)
            
            # Draw label on frame
            label = f"{class_name}: {confidence:.2f}"
            color = (0, 0, 255) if class_name == 'drowsy' else (0, 255, 0)
            cv2.putText(display_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Add to history
        if current_detection:
            detection_history.append(current_detection)
        
        # Calculate rolling average state (every 5 frames)
        if len(detection_history) >= 1 and inference_count % 5 == 0:
            drowsy_count = sum(1 for d in detection_history if d[0] == 'drowsy')
            notdrowsy_count = sum(1 for d in detection_history if d[0] == 'notdrowsy')
            avg_conf = sum(d[1] for d in detection_history) / len(detection_history)
            
            if drowsy_count > notdrowsy_count:
                current_state = "DROWSY"
                state_color = (0, 0, 255)  # Red
            elif notdrowsy_count > drowsy_count:
                current_state = "ALERT"
                state_color = (0, 255, 0)  # Green
            else:
                current_state = "UNCERTAIN"
                state_color = (0, 255, 255)  # Yellow
            
            # Draw state on frame
            cv2.putText(display_frame, f"State: {current_state}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2)
            
            # Report if state changed or every 30 frames
            if current_state != last_reported_state or inference_count % 30 == 0:
                print(f"[Frame {inference_count}] State: {current_state} (avg conf: {avg_conf:.2f}, history: {drowsy_count}D/{notdrowsy_count}A)")
                last_reported_state = current_state
        
        elif len(detection_history) == 0 and inference_count % 30 == 0:
            print(f"[Frame {inference_count}] No detections in recent frames")
            cv2.putText(display_frame, "No Detection", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)

        # Display
        cv2.imshow("Drowsiness Detection (TFLite) - Press 'q' to quit", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[SUCCESS] Ran {inference_count} inferences successfully!")
    return True


def main():
    print("\n" + "=" * 50)
    print("   DROWSINESS DETECTION - TFLite INFERENCE TEST")
    print("=" * 50)

    # Find TFLite model
    script_dir = Path(__file__).parent
    model_path = script_dir / "converted_models" / "best_float32.tflite"
    
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return 1

    print(f"\nLoading model: {model_path}")

    # Create detector
    detector = TFLiteDetector(str(model_path))
    
    # Run test
    success = test_webcam(detector)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
