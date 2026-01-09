package com.y2m.drowsydetector

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.core.content.ContextCompat
import com.y2m.drowsydetector.ui.DrowsinessScreen
import com.y2m.drowsydetector.ui.theme.DrowsyDetectorTheme
import kotlinx.coroutines.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private lateinit var detector: DrowsinessDetector
    private lateinit var alertManager: AlertManager
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var previewView: PreviewView

    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private var isProcessing = false
    private var frameCount = 0
    
    // Compose state
    private var detectionResult by mutableStateOf(
        DetectionResult(DrowsinessState.UNKNOWN, 0f, 0f)
    )

    // Permission launcher
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            startCamera()
        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        Log.d(TAG, "onCreate - initializing...")

        // Initialize components
        try {
            detector = DrowsinessDetector(this)
            Log.d(TAG, "Detector initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize detector", e)
            Toast.makeText(this, "Failed to load model: ${e.message}", Toast.LENGTH_LONG).show()
            return
        }
        
        alertManager = AlertManager(this)
        cameraExecutor = Executors.newSingleThreadExecutor()
        previewView = PreviewView(this)

        // Set Compose content
        setContent {
            DrowsyDetectorTheme {
                DrowsinessScreen(
                    detectionResult = detectionResult,
                    previewView = previewView
                )
            }
        }

        // Check camera permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        Log.d(TAG, "Starting camera...")
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // Preview use case
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            // Image analysis - request RGBA for easy processing
            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        processImage(imageProxy)
                    }
                }

            // Use front camera
            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
                Log.d(TAG, "Camera bound successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Camera binding failed", e)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImage(imageProxy: ImageProxy) {
        if (isProcessing) {
            imageProxy.close()
            return
        }

        isProcessing = true
        frameCount++

        scope.launch(Dispatchers.Default) {
            try {
                val bitmap = imageProxyToBitmap(imageProxy)
                
                if (bitmap == null) {
                    Log.e(TAG, "Failed to convert image to bitmap")
                    isProcessing = false
                    imageProxy.close()
                    return@launch
                }

                // Run detection with smoothing
                val result = detector.processFrame(bitmap)
                
                if (frameCount % 30 == 0) {
                    Log.d(TAG, "Frame $frameCount: ${result.state} (${result.confidence})")
                }

                // Update UI state
                withContext(Dispatchers.Main) {
                    detectionResult = result
                    
                    // Manage alerts
                    when (result.state) {
                        DrowsinessState.DROWSY -> {
                            if (!alertManager.isAlarmPlaying()) {
                                alertManager.startAlarm()
                            }
                        }
                        DrowsinessState.AWAKE -> {
                            if (alertManager.isAlarmPlaying()) {
                                alertManager.stopAlarm()
                            }
                        }
                        else -> { /* No action for unknown state */ }
                    }
                }

            } catch (e: Exception) {
                Log.e(TAG, "Processing error", e)
            } finally {
                imageProxy.close()
                isProcessing = false
            }
        }
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            val buffer = imageProxy.planes[0].buffer
            val pixelStride = imageProxy.planes[0].pixelStride
            val rowStride = imageProxy.planes[0].rowStride
            val rowPadding = rowStride - pixelStride * imageProxy.width
            
            val bitmapWidth = imageProxy.width + rowPadding / pixelStride
            val bitmap = Bitmap.createBitmap(bitmapWidth, imageProxy.height, Bitmap.Config.ARGB_8888)
            
            buffer.rewind()
            bitmap.copyPixelsFromBuffer(buffer)
            
            // Crop to actual size
            val croppedBitmap = Bitmap.createBitmap(bitmap, 0, 0, imageProxy.width, imageProxy.height)
            
            // Rotate and mirror for front camera
            val matrix = android.graphics.Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                postScale(-1f, 1f) // Mirror
            }
            
            Bitmap.createBitmap(croppedBitmap, 0, 0, croppedBitmap.width, croppedBitmap.height, matrix, true)
        } catch (e: Exception) {
            Log.e(TAG, "Error converting image", e)
            null
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
        cameraExecutor.shutdown()
        detector.close()
        alertManager.release()
    }

    companion object {
        private const val TAG = "DrowsyDetector"
    }
}
