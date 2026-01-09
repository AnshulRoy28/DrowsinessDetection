package com.y2m.drowsydetector.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

// Alert Colors
val SafeGreen = Color(0xFF4CAF50)
val SafeGreenDark = Color(0xFF388E3C)
val DangerRed = Color(0xFFD32F2F)
val DangerRedBright = Color(0xFFFF1744)
val DarkBackground = Color(0xFF121212)
val DarkSurface = Color(0xFF1E1E1E)

private val DarkColorScheme = darkColorScheme(
    primary = SafeGreen,
    secondary = SafeGreenDark,
    tertiary = DangerRed,
    background = DarkBackground,
    surface = DarkSurface,
    onPrimary = Color.White,
    onSecondary = Color.White,
    onTertiary = Color.White,
    onBackground = Color.White,
    onSurface = Color.White,
)

private val LightColorScheme = lightColorScheme(
    primary = SafeGreen,
    secondary = SafeGreenDark,
    tertiary = DangerRed,
    background = DarkBackground,
    surface = DarkSurface,
    onPrimary = Color.White,
    onSecondary = Color.White,
    onTertiary = Color.White,
    onBackground = Color.White,
    onSurface = Color.White,
)

@Composable
fun DrowsyDetectorTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    // Always use dark theme for driving at night
    val colorScheme = DarkColorScheme

    MaterialTheme(
        colorScheme = colorScheme,
        content = content
    )
}
