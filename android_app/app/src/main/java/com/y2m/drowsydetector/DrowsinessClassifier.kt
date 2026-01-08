package com.y2m.drowsydetector

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * TFLite classifier for drowsiness detection.
 * 
 * Model expects 640x640 RGB input and outputs 2 classes:
 * - drowsy (index 0)
 * - notdrowsy (index 1)
 */
class DrowsinessClassifier(context: Context) {

    private val interpreter: Interpreter
    private val inputSize = 640
    private val numClasses = 2
    private val labels = listOf("Drowsy", "Not Drowsy")

    // Input buffer: 1 x 640 x 640 x 3 (NHWC format) x 4 bytes (float32)
    private val inputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(1 * inputSize * inputSize * 3 * 4)
        .order(ByteOrder.nativeOrder())

    // Output buffer: 1 x 2 (2 classes)
    private val outputBuffer: Array<FloatArray> = Array(1) { FloatArray(numClasses) }

    init {
        // Load model from assets
        val modelBuffer = FileUtil.loadMappedFile(context, "best_float32.tflite")
        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }
        interpreter = Interpreter(modelBuffer, options)
    }

    /**
     * Classify a bitmap image.
     * 
     * @param bitmap Input image (will be resized to 640x640)
     * @return Pair of (label, confidence)
     */
    fun classify(bitmap: Bitmap): Pair<String, Float> {
        // Resize to model input size
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        
        // Preprocess: convert to float buffer
        preprocessBitmap(resizedBitmap)
        
        // Run inference
        interpreter.run(inputBuffer, outputBuffer)
        
        // Get prediction
        val scores = outputBuffer[0]
        val maxIndex = scores.indices.maxByOrNull { scores[it] } ?: 0
        val confidence = scores[maxIndex]
        
        return Pair(labels[maxIndex], confidence)
    }

    /**
     * Convert bitmap to normalized float buffer.
     * Normalizes pixel values to [0, 1] range.
     */
    private fun preprocessBitmap(bitmap: Bitmap) {
        inputBuffer.rewind()
        
        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)
        
        for (pixel in pixels) {
            // Extract RGB and normalize to [0, 1]
            val r = ((pixel shr 16) and 0xFF) / 255.0f
            val g = ((pixel shr 8) and 0xFF) / 255.0f
            val b = (pixel and 0xFF) / 255.0f
            
            inputBuffer.putFloat(r)
            inputBuffer.putFloat(g)
            inputBuffer.putFloat(b)
        }
    }

    /**
     * Check if user is drowsy based on classification.
     */
    fun isDrowsy(bitmap: Bitmap): Boolean {
        val (label, _) = classify(bitmap)
        return label == "Drowsy"
    }

    /**
     * Release interpreter resources.
     */
    fun close() {
        interpreter.close()
    }
}
