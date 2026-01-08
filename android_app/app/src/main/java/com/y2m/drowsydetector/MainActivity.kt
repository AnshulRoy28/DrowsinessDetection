package com.y2m.drowsydetector

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.y2m.drowsydetector.databinding.ActivityMainBinding
import kotlinx.coroutines.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var classifier: DrowsinessClassifier
    private lateinit var cameraExecutor: ExecutorService

    private val scope = CoroutineScope(Dispatchers.Main + SupervisorJob())
    private var isProcessing = false

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
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Initialize classifier
        classifier = DrowsinessClassifier(this)
        cameraExecutor = Executors.newSingleThreadExecutor()

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
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // Preview use case
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(binding.cameraPreview.surfaceProvider)
                }

            // Image analysis use case
            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(640, 640))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        processImage(imageProxy)
                    }
                }

            // Use front camera for drowsiness detection
            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
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

        scope.launch(Dispatchers.Default) {
            try {
                // Convert ImageProxy to Bitmap
                val bitmap = imageProxyToBitmap(imageProxy)
                
                // Run classification
                val (label, confidence) = classifier.classify(bitmap)
                val isDrowsy = label == "Drowsy"

                // Update UI on main thread
                withContext(Dispatchers.Main) {
                    updateUI(label, confidence, isDrowsy)
                }

            } catch (e: Exception) {
                Log.e(TAG, "Classification error", e)
            } finally {
                imageProxy.close()
                isProcessing = false
            }
        }
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val buffer = imageProxy.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)

        // Convert YUV to bitmap (simplified - using the Y plane as grayscale)
        val bitmap = Bitmap.createBitmap(
            imageProxy.width,
            imageProxy.height,
            Bitmap.Config.ARGB_8888
        )

        // For proper conversion, you'd use YuvToRgbConverter
        // This is a simplified version
        val yBuffer = imageProxy.planes[0].buffer
        val ySize = yBuffer.remaining()
        val yBytes = ByteArray(ySize)
        yBuffer.get(yBytes)

        val pixels = IntArray(imageProxy.width * imageProxy.height)
        for (i in pixels.indices) {
            val y = yBytes[i].toInt() and 0xFF
            pixels[i] = (0xFF shl 24) or (y shl 16) or (y shl 8) or y
        }
        bitmap.setPixels(pixels, 0, imageProxy.width, 0, 0, imageProxy.width, imageProxy.height)

        // Rotate if needed
        val matrix = Matrix().apply {
            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
        }
        
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun updateUI(label: String, confidence: Float, isDrowsy: Boolean) {
        // Update status text
        binding.statusText.text = label
        binding.confidenceText.text = "Confidence: ${(confidence * 100).toInt()}%"

        // Update background color based on drowsiness
        val color = if (isDrowsy) {
            getColor(android.R.color.holo_red_light)
        } else {
            getColor(android.R.color.holo_green_light)
        }
        binding.statusCard.setCardBackgroundColor(color)

        // Show warning if drowsy
        if (isDrowsy && confidence > 0.7f) {
            binding.warningText.text = "⚠️ DROWSINESS DETECTED!"
            binding.warningText.visibility = android.view.View.VISIBLE
        } else {
            binding.warningText.visibility = android.view.View.GONE
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
        cameraExecutor.shutdown()
        classifier.close()
    }

    companion object {
        private const val TAG = "DrowsyDetector"
    }
}
