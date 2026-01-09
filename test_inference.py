"""
Test Inference Script for Drowsiness Detection Model

This script loads the YOLO model (best.pt) and runs inference on:
1. The webcam (live detection)
2. A sample image (if provided)

Usage:
    python test_inference.py              # Run with webcam
    python test_inference.py --image path/to/image.jpg  # Run on image
"""

import argparse
import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Install with:")
    print("  pip install ultralytics")
    exit(1)


def test_webcam(model):
    """Run real-time drowsiness detection on webcam with rolling average."""
    print("\n" + "=" * 50)
    print("   WEBCAM INFERENCE TEST")
    print("=" * 50)
    print("Press 'q' to quit\n")

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return False

    inference_count = 0
    
    # Rolling average tracking (last 5 inferences)
    from collections import deque
    detection_history = deque(maxlen=5)  # Store tuples of (class_name, confidence)
    last_reported_state = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read from webcam")
            break

        # Run inference
        results = model(frame, verbose=False)
        inference_count += 1

        # Process results
        annotated_frame = results[0].plot()
        
        # Get detection info - handle different result formats
        current_detection = None
        
        if results[0].boxes is not None:
            boxes = results[0].boxes
            if hasattr(boxes, 'data') and len(boxes.data) > 0:
                # Get the highest confidence detection
                for i in range(len(boxes.data)):
                    cls = int(boxes.cls[i].item()) if hasattr(boxes.cls[i], 'item') else int(boxes.cls[i])
                    conf = float(boxes.conf[i].item()) if hasattr(boxes.conf[i], 'item') else float(boxes.conf[i])
                    class_name = model.names[cls]
                    
                    if current_detection is None or conf > current_detection[1]:
                        current_detection = (class_name, conf)
        
        # Add to history
        if current_detection:
            detection_history.append(current_detection)
        
        # Calculate rolling average state (every 5 frames or when we have enough data)
        if len(detection_history) >= 1 and inference_count % 5 == 0:
            # Count drowsy vs not drowsy in history
            drowsy_count = sum(1 for d in detection_history if d[0] == 'drowsy')
            notdrowsy_count = sum(1 for d in detection_history if d[0] == 'notdrowsy')
            avg_conf = sum(d[1] for d in detection_history) / len(detection_history)
            
            # Determine current state based on majority
            if drowsy_count > notdrowsy_count:
                current_state = "DROWSY"
            elif notdrowsy_count > drowsy_count:
                current_state = "ALERT"
            else:
                current_state = "UNCERTAIN"
            
            # Report if state changed or every 30 frames
            if current_state != last_reported_state or inference_count % 30 == 0:
                print(f"[Frame {inference_count}] State: {current_state} (avg conf: {avg_conf:.2f}, history: {drowsy_count}D/{notdrowsy_count}A)")
                last_reported_state = current_state
        
        elif len(detection_history) == 0 and inference_count % 30 == 0:
            print(f"[Frame {inference_count}] No detections in recent frames")

        # Display
        cv2.imshow("Drowsiness Detection - Press 'q' to quit", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[SUCCESS] Ran {inference_count} inferences successfully!")
    return True


def test_image(model, image_path):
    """Run inference on a single image."""
    print("\n" + "=" * 50)
    print("   IMAGE INFERENCE TEST")
    print("=" * 50)

    if not Path(image_path).exists():
        print(f"ERROR: Image not found: {image_path}")
        return False

    print(f"Loading image: {image_path}")
    
    # Run inference
    results = model(image_path)
    
    # Process results
    boxes = results[0].boxes
    print(f"\nDetections found: {len(boxes)}")
    
    for i, box in enumerate(boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls]
        xyxy = box.xyxy[0].tolist()
        print(f"  [{i+1}] Class: '{class_name}', Confidence: {conf:.2f}, Box: {xyxy}")

    # Show annotated image
    annotated = results[0].plot()
    cv2.imshow("Detection Result - Press any key to close", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n[SUCCESS] Inference completed!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test Drowsiness Detection Model Inference")
    parser.add_argument("--image", "-i", type=str, help="Path to an image to test (uses webcam if not provided)")
    parser.add_argument("--model", "-m", type=str, default="best.pt", help="Path to the model file (default: best.pt)")
    args = parser.parse_args()

    # Find model
    model_path = Path(args.model)
    if not model_path.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent
        model_path = script_dir / args.model
        if not model_path.exists():
            print(f"ERROR: Model not found: {args.model}")
            return 1

    print("\n" + "=" * 50)
    print("   DROWSINESS DETECTION - INFERENCE TEST")
    print("=" * 50)
    print(f"\nLoading model: {model_path}")

    # Load model
    model = YOLO(str(model_path))
    print(f"Model loaded successfully!")
    print(f"Classes: {model.names}")

    # Run test
    if args.image:
        success = test_image(model, args.image)
    else:
        success = test_webcam(model)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
