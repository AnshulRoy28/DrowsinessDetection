package com.y2m.drowsydetector

import android.content.Context
import android.media.AudioAttributes
import android.media.MediaPlayer
import android.media.RingtoneManager
import android.util.Log

/**
 * Manages audio alerts for drowsiness detection.
 * 
 * Plays a looping alarm when drowsiness is detected,
 * and stops when the user wakes up.
 */
class AlertManager(private val context: Context) {
    
    private var mediaPlayer: MediaPlayer? = null
    private var isPlaying = false
    
    /**
     * Start playing the alarm sound in a loop.
     */
    fun startAlarm() {
        if (isPlaying) return
        
        try {
            // Use system alarm sound
            val alarmUri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_ALARM)
                ?: RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION)
                ?: RingtoneManager.getDefaultUri(RingtoneManager.TYPE_RINGTONE)
            
            mediaPlayer = MediaPlayer().apply {
                setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_ALARM)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                        .build()
                )
                setDataSource(context, alarmUri)
                isLooping = true
                prepare()
                start()
            }
            
            isPlaying = true
            Log.d(TAG, "Alarm started")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to start alarm", e)
        }
    }
    
    /**
     * Stop the alarm sound.
     */
    fun stopAlarm() {
        if (!isPlaying) return
        
        try {
            mediaPlayer?.apply {
                if (isPlaying) {
                    stop()
                }
                release()
            }
            mediaPlayer = null
            isPlaying = false
            Log.d(TAG, "Alarm stopped")
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to stop alarm", e)
        }
    }
    
    /**
     * Check if alarm is currently playing.
     */
    fun isAlarmPlaying(): Boolean = isPlaying
    
    /**
     * Release all resources.
     */
    fun release() {
        stopAlarm()
    }
    
    companion object {
        private const val TAG = "AlertManager"
    }
}
