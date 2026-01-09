package com.y2m.drowsydetector

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Drowsiness detection state
 */
enum class DrowsinessState {
    AWAKE,
    DROWSY,
    UNKNOWN
}

/**
 * Detection result with state and confidence
 */
data class DetectionResult(
    val state: DrowsinessState,
    val confidence: Float,
    val rawScore: Float
)

/**
 * High-level drowsiness detector with consecutive frame smoothing.
 * 
 * Wraps the TFLite classifier and implements the detection logic:
 * - Drowsy threshold: score >= 0.5
 * - Requires 3 consecutive drowsy frames to trigger alert
 * - Immediately clears alert when awake detected
 */
class DrowsinessDetector(context: Context) {
    
    private val classifier = DrowsinessClassifier(context)
    
    // State flow for reactive UI updates
    private val _state = MutableStateFlow(DetectionResult(DrowsinessState.UNKNOWN, 0f, 0f))
    val state: StateFlow<DetectionResult> = _state.asStateFlow()
    
    // Consecutive frame counter for smoothing
    private var consecutiveDrowsyFrames = 0
    private val DROWSY_THRESHOLD = 0.5f
    private val REQUIRED_CONSECUTIVE_FRAMES = 3
    
    // Current confirmed state (after smoothing)
    private var confirmedState = DrowsinessState.AWAKE
    
    /**
     * Process a camera frame and update detection state.
     * 
     * @param bitmap Camera frame to analyze
     * @return Detection result with state and confidence
     */
    fun processFrame(bitmap: Bitmap): DetectionResult {
        try {
            // Run classification
            val (label, confidence) = classifier.classify(bitmap)
            
            // Determine raw state from single frame
            val isDrowsy = label == "Drowsy" && confidence >= DROWSY_THRESHOLD
            val rawScore = if (label == "Drowsy") confidence else 1f - confidence
            
            // Apply consecutive frame smoothing
            if (isDrowsy) {
                consecutiveDrowsyFrames++
                Log.d(TAG, "Drowsy frame detected ($consecutiveDrowsyFrames/$REQUIRED_CONSECUTIVE_FRAMES)")
                
                // Only trigger drowsy state after consecutive frames
                if (consecutiveDrowsyFrames >= REQUIRED_CONSECUTIVE_FRAMES) {
                    confirmedState = DrowsinessState.DROWSY
                }
            } else {
                // Immediately clear drowsy state when awake detected
                consecutiveDrowsyFrames = 0
                confirmedState = DrowsinessState.AWAKE
            }
            
            val result = DetectionResult(
                state = confirmedState,
                confidence = confidence,
                rawScore = rawScore
            )
            
            _state.value = result
            return result
            
        } catch (e: Exception) {
            Log.e(TAG, "Detection error", e)
            return DetectionResult(DrowsinessState.UNKNOWN, 0f, 0f)
        }
    }
    
    /**
     * Reset detection state
     */
    fun reset() {
        consecutiveDrowsyFrames = 0
        confirmedState = DrowsinessState.AWAKE
        _state.value = DetectionResult(DrowsinessState.AWAKE, 0f, 0f)
    }
    
    /**
     * Release resources
     */
    fun close() {
        classifier.close()
    }
    
    companion object {
        private const val TAG = "DrowsinessDetector"
    }
}
