package com.y2m.drowsydetector

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.TimeUnit

/**
 * MongoDB Atlas Data API logger for drowsiness events.
 * 
 * Logs drowsiness detection events to MongoDB Atlas via the Data API.
 * This approach is secure and doesn't require embedding connection strings.
 */
class MongoLogger {

    private val client = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .writeTimeout(10, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .build()

    // MongoDB Atlas Data API endpoint
    // You'll need to enable Data API in MongoDB Atlas and get an API key
    private val dataApiUrl = "https://data.mongodb-api.com/app/data-ewqsl/endpoint/data/v1/action/insertOne"
    private val apiKey = "YOUR_MONGODB_DATA_API_KEY" // Replace with actual API key from MongoDB Atlas

    private val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.US).apply {
        timeZone = TimeZone.getTimeZone("UTC")
    }

    /**
     * Log a drowsiness event to MongoDB.
     * 
     * @param isDrowsy Whether the user is drowsy
     * @param confidence Confidence score (0-1)
     * @param state State string ("Drowsy" or "Alert")
     */
    suspend fun logEvent(isDrowsy: Boolean, confidence: Float, state: String) {
        withContext(Dispatchers.IO) {
            try {
                val timestamp = dateFormat.format(Date())
                
                // Build the document to insert
                val document = JSONObject().apply {
                    put("timestamp", timestamp)
                    put("isDrowsy", isDrowsy)
                    put("confidence", confidence.toDouble())
                    put("state", state)
                    put("deviceId", android.os.Build.MODEL)
                    put("appVersion", "1.0.0")
                }

                // Build the request body for MongoDB Data API
                val requestBody = JSONObject().apply {
                    put("collection", "drowsiness_logs")
                    put("database", "drowsiguard")
                    put("dataSource", "Cluster0")
                    put("document", document)
                }

                val request = Request.Builder()
                    .url(dataApiUrl)
                    .addHeader("Content-Type", "application/json")
                    .addHeader("api-key", apiKey)
                    .post(requestBody.toString().toRequestBody("application/json".toMediaType()))
                    .build()

                val response = client.newCall(request).execute()
                
                if (response.isSuccessful) {
                    Log.d(TAG, "Successfully logged event to MongoDB: $state")
                } else {
                    Log.e(TAG, "Failed to log event: ${response.code} - ${response.message}")
                }
                response.close()
                
            } catch (e: Exception) {
                Log.e(TAG, "Error logging to MongoDB", e)
            }
        }
    }

    /**
     * Alternative: Log to your backend server instead of direct MongoDB.
     * This is more secure for production use.
     */
    suspend fun logToBackend(isDrowsy: Boolean, confidence: Float, state: String, backendUrl: String) {
        withContext(Dispatchers.IO) {
            try {
                val timestamp = dateFormat.format(Date())
                
                val payload = JSONObject().apply {
                    put("timestamp", timestamp)
                    put("isDrowsy", isDrowsy)
                    put("confidence", confidence.toDouble())
                    put("state", state)
                    put("deviceId", android.os.Build.MODEL)
                }

                val request = Request.Builder()
                    .url("$backendUrl/api/drowsiness/log")
                    .addHeader("Content-Type", "application/json")
                    .post(payload.toString().toRequestBody("application/json".toMediaType()))
                    .build()

                val response = client.newCall(request).execute()
                
                if (response.isSuccessful) {
                    Log.d(TAG, "Successfully logged to backend: $state")
                } else {
                    Log.e(TAG, "Backend log failed: ${response.code}")
                }
                response.close()
                
            } catch (e: Exception) {
                Log.e(TAG, "Error logging to backend", e)
            }
        }
    }

    companion object {
        private const val TAG = "MongoLogger"
    }
}
