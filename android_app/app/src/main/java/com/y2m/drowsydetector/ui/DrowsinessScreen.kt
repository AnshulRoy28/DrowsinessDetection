package com.y2m.drowsydetector.ui

import androidx.camera.view.PreviewView
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.y2m.drowsydetector.DetectionResult
import com.y2m.drowsydetector.DrowsinessState
import com.y2m.drowsydetector.ui.theme.DangerRed
import com.y2m.drowsydetector.ui.theme.DangerRedBright
import com.y2m.drowsydetector.ui.theme.DarkBackground
import com.y2m.drowsydetector.ui.theme.SafeGreen

/**
 * Main drowsiness detection screen with:
 * - Full-screen background that changes color based on state
 * - Circular camera preview
 * - Status text indicator
 * - Flashing animation when drowsy
 */
@Composable
fun DrowsinessScreen(
    detectionResult: DetectionResult,
    previewView: PreviewView,
    modifier: Modifier = Modifier
) {
    val isDrowsy = detectionResult.state == DrowsinessState.DROWSY
    
    // Animate background color
    val backgroundColor by animateColorAsState(
        targetValue = when (detectionResult.state) {
            DrowsinessState.DROWSY -> DangerRed
            DrowsinessState.AWAKE -> DarkBackground
            DrowsinessState.UNKNOWN -> DarkBackground
        },
        animationSpec = tween(durationMillis = 300),
        label = "backgroundColor"
    )
    
    // Flashing effect for drowsy state
    val infiniteTransition = rememberInfiniteTransition(label = "flash")
    val flashAlpha by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = if (isDrowsy) 0.5f else 0f,
        animationSpec = infiniteRepeatable(
            animation = tween(500, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "flashAlpha"
    )
    
    Box(
        modifier = modifier
            .fillMaxSize()
            .background(backgroundColor)
    ) {
        // Flashing overlay when drowsy
        if (isDrowsy) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .background(DangerRedBright.copy(alpha = flashAlpha))
            )
        }
        
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(modifier = Modifier.weight(0.1f))
            
            // Camera Preview (circular)
            Box(
                modifier = Modifier
                    .size(300.dp)
                    .clip(CircleShape)
                    .background(Color.Black)
            ) {
                AndroidView(
                    factory = { previewView },
                    modifier = Modifier.fillMaxSize()
                )
            }
            
            Spacer(modifier = Modifier.weight(0.1f))
            
            // Status Indicator
            StatusIndicator(
                state = detectionResult.state,
                confidence = detectionResult.confidence
            )
            
            Spacer(modifier = Modifier.height(32.dp))
        }
    }
}

@Composable
private fun StatusIndicator(
    state: DrowsinessState,
    confidence: Float
) {
    val statusText = when (state) {
        DrowsinessState.DROWSY -> "DROWSY DETECTED!"
        DrowsinessState.AWAKE -> "ACTIVE / AWAKE"
        DrowsinessState.UNKNOWN -> "Initializing..."
    }
    
    val statusColor = when (state) {
        DrowsinessState.DROWSY -> Color.White
        DrowsinessState.AWAKE -> SafeGreen
        DrowsinessState.UNKNOWN -> Color.Gray
    }
    
    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = statusText,
            color = statusColor,
            fontSize = 32.sp,
            fontWeight = FontWeight.Bold,
            textAlign = TextAlign.Center
        )
        
        Spacer(modifier = Modifier.height(8.dp))
        
        Text(
            text = "Confidence: ${(confidence * 100).toInt()}%",
            color = Color.White.copy(alpha = 0.7f),
            fontSize = 16.sp,
            textAlign = TextAlign.Center
        )
    }
}
