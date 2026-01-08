# Y2M - YOLO to Mobile Pipeline

Complete pipeline for converting YOLO models to TFLite and deploying on Android.

## Project Components

### 1. Conversion Pipeline (`y2m/`)
Python CLI to convert YOLO `.pt` models to TFLite format.

```bash
# Install dependencies (use WSL/Linux)
pip install -r requirements.txt

# Convert model
python -m y2m.cli --weights best.pt --output ./converted_models
```

### 2. Android App (`android_app/`)
Real-time drowsiness detection app using the converted TFLite model.

**See [android_app/README.md](android_app/README.md) for build instructions.**

---

## Output Files
```
converted_models/
├── best_float32.tflite   # Model for Android
└── metadata.json         # Class names
```

## Requirements
- Python 3.9+ with TensorFlow (for conversion)
- Android Studio 2022+ (for APK build)
