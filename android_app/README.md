# Drowsy Detector - Android App

Real-time drowsiness detection using TensorFlow Lite and CameraX.

## Quick Start (Android Studio)

### 1. Clone & Open
```bash
git clone <repository-url>
```
Open the `android_app/` folder in **Android Studio 2022.3+**

### 2. Wait for Gradle Sync
- Android Studio will download dependencies automatically
- Takes ~5 minutes on first run

### 3. Build APK
- Go to **Build → Build Bundle(s) / APK(s) → Build APK(s)**
- APK location: `app/build/outputs/apk/debug/app-debug.apk`

### 4. Install on Device
- Connect Android device via USB (enable USB debugging)
- Click **Run → Run 'app'**

Or via command line:
```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

---

## Model Info

| Property | Value |
|----------|-------|
| Input Size | 640×640 RGB |
| Classes | `drowsy`, `notdrowsy` |
| Model Size | 5.77 MB (float32) |
| Min Android | 7.0 (API 24) |

---

## Permissions Required
- **Camera** - For real-time face detection

---

## Project Structure
```
android_app/
├── app/src/main/
│   ├── java/.../MainActivity.kt      # Camera + UI
│   ├── java/.../DrowsinessClassifier.kt  # TFLite inference
│   ├── assets/best_float32.tflite    # ML model
│   └── res/layout/activity_main.xml  # Layout
└── build.gradle.kts                  # Dependencies
```
