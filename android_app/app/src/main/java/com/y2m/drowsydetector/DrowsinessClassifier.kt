package com.y2m.drowsydetector

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * TFLite classifier for drowsiness detection.
 * 
 * Handles YOLO-style output format with detection boxes.
 * Model expects 640x640 RGB input.
 * Classes: drowsy (0), notdrowsy (1)
 */
class DrowsinessClassifier(context: Context) {

    private val interpreter: Interpreter
    private val inputSize = 640
    private val labels = listOf("Drowsy", "Not Drowsy")
    
    // Confidence threshold for detections
    private val confidenceThreshold = 0.25f

    // Input buffer: 1 x 640 x 640 x 3 (NHWC format) x 4 bytes (float32)
    private val inputBuffer: ByteBuffer = ByteBuffer
        .allocateDirect(1 * inputSize * inputSize * 3 * 4)
        .order(ByteOrder.nativeOrder())

    init {
        // Load model from assets
        val modelBuffer = FileUtil.loadMappedFile(context, "best_float32.tflite")
        val options = Interpreter.Options().apply {
            setNumThreads(4)
        }
        interpreter = Interpreter(modelBuffer, options)
        
        // Log model info for debugging
        val inputShape = interpreter.getInputTensor(0).shape()
        val outputShape = interpreter.getOutputTensor(0).shape()
        Log.d(TAG, "Model input shape: ${inputShape.contentToString()}")
        Log.d(TAG, "Model output shape: ${outputShape.contentToString()}")
    }

    /**
     * Classify a bitmap image.
     * 
     * @param bitmap Input image (will be resized to 640x640)
     * @return Pair of (label, confidence)
     */
    fun classify(bitmap: Bitmap): Pair<String, Float> {
        try {
            // Resize to model input size
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
            
            // Preprocess: convert to float buffer
            preprocessBitmap(resizedBitmap)
            
            // Get output shape to determine format
            val outputShape = interpreter.getOutputTensor(0).shape()
            Log.d(TAG, "Running inference, output shape: ${outputShape.contentToString()}")
            
            return when {
                // YOLO format: [1, num_boxes, 6] where 6 = x,y,w,h,class1_conf,class2_conf
                outputShape.size == 3 && outputShape[2] >= 6 -> {
                    runYoloInference(outputShape)
                }
                // Classification format: [1, 2]
                outputShape.size == 2 && outputShape[1] == 2 -> {
                    runClassificationInference()
                }
                // YOLO transposed format: [1, 6, num_boxes]
                outputShape.size == 3 && outputShape[1] >= 6 -> {
                    runYoloTransposedInference(outputShape)
                }
                else -> {
                    Log.w(TAG, "Unknown output format, trying generic approach")
                    runGenericInference(outputShape)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Classification failed", e)
            return Pair("Not Drowsy", 0.5f) // Default to not drowsy on error
        }
    }

    private fun runYoloInference(outputShape: IntArray): Pair<String, Float> {
        val numBoxes = outputShape[1]
        val boxSize = outputShape[2]
        val output = Array(1) { Array(numBoxes) { FloatArray(boxSize) } }
        
        interpreter.run(inputBuffer, output)
        
        var bestLabel = "Not Drowsy"
        var bestConfidence = 0f
        var detectionCount = 0
        
        for (i in 0 until numBoxes) {
            val box = output[0][i]
            // box format: [x, y, w, h, class0_conf, class1_conf, ...]
            if (boxSize >= 6) {
                val drowsyConf = box[4]
                val alertConf = box[5]
                val maxConf = maxOf(drowsyConf, alertConf)
                
                if (maxConf > confidenceThreshold && maxConf > bestConfidence) {
                    bestConfidence = maxConf
                    bestLabel = if (drowsyConf > alertConf) "Drowsy" else "Not Drowsy"
                    detectionCount++
                }
            }
        }
        
        Log.d(TAG, "YOLO: Found $detectionCount detections, best: $bestLabel ($bestConfidence)")
        return Pair(bestLabel, if (bestConfidence > 0) bestConfidence else 0.5f)
    }

    private fun runYoloTransposedInference(outputShape: IntArray): Pair<String, Float> {
        val numChannels = outputShape[1]
        val numBoxes = outputShape[2]
        val output = Array(1) { Array(numChannels) { FloatArray(numBoxes) } }
        
        interpreter.run(inputBuffer, output)
        
        var bestLabel = "Not Drowsy"
        var bestConfidence = 0f
        
        for (i in 0 until numBoxes) {
            // Transposed format: channels are first dimension
            val drowsyConf = if (numChannels > 4) output[0][4][i] else 0f
            val alertConf = if (numChannels > 5) output[0][5][i] else 0f
            val maxConf = maxOf(drowsyConf, alertConf)
            
            if (maxConf > confidenceThreshold && maxConf > bestConfidence) {
                bestConfidence = maxConf
                bestLabel = if (drowsyConf > alertConf) "Drowsy" else "Not Drowsy"
            }
        }
        
        Log.d(TAG, "YOLO Transposed: best: $bestLabel ($bestConfidence)")
        return Pair(bestLabel, if (bestConfidence > 0) bestConfidence else 0.5f)
    }

    private fun runClassificationInference(): Pair<String, Float> {
        val output = Array(1) { FloatArray(2) }
        interpreter.run(inputBuffer, output)
        
        val scores = output[0]
        val maxIndex = if (scores[0] > scores[1]) 0 else 1
        val confidence = scores[maxIndex]
        
        Log.d(TAG, "Classification: ${labels[maxIndex]} ($confidence)")
        return Pair(labels[maxIndex], confidence)
    }

    private fun runGenericInference(outputShape: IntArray): Pair<String, Float> {
        // Flatten and try to find max scores
        val totalSize = outputShape.reduce { acc, i -> acc * i }
        val flatOutput = FloatArray(totalSize)
        val outputBuffer = ByteBuffer.allocateDirect(totalSize * 4).order(ByteOrder.nativeOrder())
        
        interpreter.run(inputBuffer, outputBuffer)
        
        outputBuffer.rewind()
        for (i in 0 until totalSize) {
            flatOutput[i] = outputBuffer.float
        }
        
        // Find peaks that might be class scores
        var drowsyScore = 0f
        var alertScore = 0f
        
        for (i in flatOutput.indices step (outputShape.lastOrNull() ?: 6)) {
            if (i + 5 < flatOutput.size) {
                if (flatOutput[i + 4] > drowsyScore) drowsyScore = flatOutput[i + 4]
                if (flatOutput[i + 5] > alertScore) alertScore = flatOutput[i + 5]
            }
        }
        
        val label = if (drowsyScore > alertScore) "Drowsy" else "Not Drowsy"
        val confidence = maxOf(drowsyScore, alertScore, 0.5f)
        
        Log.d(TAG, "Generic: $label ($confidence)")
        return Pair(label, confidence)
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

    companion object {
        private const val TAG = "DrowsinessClassifier"
    }
}
