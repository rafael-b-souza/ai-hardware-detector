package com.aidetector.server

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.net.wifi.WifiManager
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import java.net.Inet4Address
import java.net.NetworkInterface

/** Foreground service que mantém o servidor de inferência em execução contínua. */
class InferenceService : Service() {

    companion object {
        private const val TAG = "InferenceService"
        private const val CHANNEL_ID = "inference_server"
        private const val NOTIFICATION_ID = 1001
        
        const val ACTION_START = "com.aidetector.START_SERVER"
        const val ACTION_STOP = "com.aidetector.STOP_SERVER"
        const val EXTRA_PORT = "port"
        const val EXTRA_API_KEY = "api_key"
    }

    private var server: InferenceServer? = null
    private var wakeLock: PowerManager.WakeLock? = null
    private var wifiLock: WifiManager.WifiLock? = null
    private val binder = LocalBinder()

    inner class LocalBinder : Binder() {
        fun getServer(): InferenceServer? = server
        fun getServerAddress(): String = "http://${getLocalIpAddress()}:${server?.listeningPort ?: 8080}"
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> {
                val port = intent.getIntExtra(EXTRA_PORT, 8080)
                val apiKey = intent.getStringExtra(EXTRA_API_KEY)
                startInferenceServer(port, apiKey)
            }
            ACTION_STOP -> {
                stopInferenceServer()
                stopSelf()
            }
        }
        return START_STICKY
    }

    private fun startInferenceServer(port: Int, apiKey: String?) {
        if (server != null) {
            Log.w(TAG, "Servidor já está rodando")
            return
        }

        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "AIDetector::InferenceWakeLock"
        ).apply { acquire(24 * 60 * 60 * 1000L) }

        val wm = applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
        wifiLock = wm.createWifiLock(
            WifiManager.WIFI_MODE_FULL_HIGH_PERF,
            "AIDetector::InferenceWifiLock"
        ).apply { acquire() }

        server = InferenceServer(applicationContext, port).also {
            it.startServer(apiKey = apiKey, daemon = false)
        }

        val address = "http://${getLocalIpAddress()}:$port"
        Log.i(TAG, "Servidor iniciado em $address")

        val notification = buildNotification(address, apiKey != null)
        startForeground(NOTIFICATION_ID, notification)
    }

    private fun stopInferenceServer() {
        server?.stopServer()
        server = null

        wakeLock?.let {
            if (it.isHeld) it.release()
        }
        wakeLock = null

        wifiLock?.let {
            if (it.isHeld) it.release()
        }
        wifiLock = null

        Log.i(TAG, "Servidor parado")
    }

    override fun onDestroy() {
        stopInferenceServer()
        super.onDestroy()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "AI Inference Server",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Servidor de inferência de IA rodando em background"
                setShowBadge(false)
            }
            val nm = getSystemService(NotificationManager::class.java)
            nm.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(address: String, authEnabled: Boolean): Notification {
        val stopIntent = Intent(this, InferenceService::class.java).apply {
            action = ACTION_STOP
        }
        val stopPendingIntent = PendingIntent.getService(
            this, 0, stopIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("AI Server ativo")
            .setContentText("$address ${if (authEnabled) "🔒" else "🔓"}")
            .setSmallIcon(android.R.drawable.ic_menu_manage)
            .setOngoing(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .addAction(
                android.R.drawable.ic_media_pause,
                "Parar",
                stopPendingIntent
            )
            .setStyle(NotificationCompat.BigTextStyle().bigText(
                "$address\n" +
                "Auth: ${if (authEnabled) "Habilitada" else "Desabilitada"}\n" +
                "Hardware: ${Build.SOC_MODEL}\n" +
                "Toque em Parar para encerrar"
            ))
            .build()
    }

    private fun getLocalIpAddress(): String {
        try {
            NetworkInterface.getNetworkInterfaces()?.toList()?.forEach { intf ->
                if (intf.name.startsWith("wlan") || intf.name.startsWith("eth")) {
                    intf.inetAddresses?.toList()?.forEach { addr ->
                        if (!addr.isLoopbackAddress && addr is Inet4Address) {
                            return addr.hostAddress ?: "0.0.0.0"
                        }
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Erro obtendo IP: ${e.message}")
        }
        return "0.0.0.0"
    }
}
