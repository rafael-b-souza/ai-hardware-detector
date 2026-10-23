package com.aidetector.ui

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.IBinder
import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.aidetector.server.InferenceService
import java.net.Inet4Address
import java.net.NetworkInterface

@Composable
fun ServerControlPanel() {
    val context = LocalContext.current
    val clipboard = LocalClipboardManager.current

    var isRunning by remember { mutableStateOf(false) }
    var port by remember { mutableStateOf("8080") }
    var apiKey by remember { mutableStateOf("") }
    var useAuth by remember { mutableStateOf(false) }
    var serverAddress by remember { mutableStateOf("") }
    var showApiKeyField by remember { mutableStateOf(false) }

    val localIp = remember { getLocalIp() }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF1A1A2E))
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                Box(
                    modifier = Modifier
                        .size(12.dp)
                        .clip(CircleShape)
                        .background(if (isRunning) Color(0xFF00C853) else Color(0xFF5F5E5A))
                )
                Spacer(Modifier.width(10.dp))
                Text(
                    if (isRunning) "Servidor ativo" else "Servidor parado",
                    fontWeight = FontWeight.Bold,
                    fontSize = 18.sp
                )
            }

            Spacer(Modifier.height(16.dp))

            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Porta:", fontSize = 14.sp, color = Color.Gray,
                    modifier = Modifier.width(60.dp))
                OutlinedTextField(
                    value = port,
                    onValueChange = { if (it.length <= 5) port = it.filter { c -> c.isDigit() } },
                    modifier = Modifier.width(100.dp).height(48.dp),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    singleLine = true,
                    enabled = !isRunning,
                    textStyle = LocalTextStyle.current.copy(
                        fontFamily = FontFamily.Monospace,
                        fontSize = 14.sp
                    )
                )
                Spacer(Modifier.width(16.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Switch(
                        checked = useAuth,
                        onCheckedChange = {
                            useAuth = it
                            showApiKeyField = it
                        },
                        enabled = !isRunning,
                        colors = SwitchDefaults.colors(
                            checkedThumbColor = Color(0xFF00E5FF),
                            checkedTrackColor = Color(0xFF00E5FF).copy(0.3f)
                        )
                    )
                    Spacer(Modifier.width(6.dp))
                    Text("Auth", fontSize = 13.sp, color = Color.Gray)
                }
            }

            if (showApiKeyField) {
                Spacer(Modifier.height(8.dp))
                OutlinedTextField(
                    value = apiKey,
                    onValueChange = { apiKey = it },
                    modifier = Modifier.fillMaxWidth().height(48.dp),
                    placeholder = { Text("API Key (deixe vazio para gerar)", fontSize = 13.sp) },
                    singleLine = true,
                    enabled = !isRunning,
                    textStyle = LocalTextStyle.current.copy(
                        fontFamily = FontFamily.Monospace,
                        fontSize = 13.sp
                    )
                )
            }

            Spacer(Modifier.height(16.dp))

            Button(
                onClick = {
                    if (isRunning) {
                        stopServer(context)
                        isRunning = false
                        serverAddress = ""
                    } else {
                        val portNum = port.toIntOrNull() ?: 8080
                        val key = if (useAuth) {
                            apiKey.ifEmpty {
                                java.util.UUID.randomUUID().toString().take(32).also { apiKey = it }
                            }
                        } else null

                        startServer(context, portNum, key)
                        isRunning = true
                        serverAddress = "http://$localIp:$portNum"
                    }
                },
                modifier = Modifier.fillMaxWidth().height(52.dp),
                shape = RoundedCornerShape(12.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (isRunning) Color(0xFFFF5252) else Color(0xFF00E5FF)
                )
            ) {
                Icon(
                    if (isRunning) Icons.Default.Stop else Icons.Default.PlayArrow,
                    contentDescription = null,
                    tint = if (isRunning) Color.White else Color.Black
                )
                Spacer(Modifier.width(8.dp))
                Text(
                    if (isRunning) "PARAR SERVIDOR" else "INICIAR SERVIDOR",
                    color = if (isRunning) Color.White else Color.Black,
                    fontWeight = FontWeight.Bold
                )
            }

            if (isRunning && serverAddress.isNotEmpty()) {
                Spacer(Modifier.height(16.dp))

                Surface(
                    onClick = {
                        clipboard.setText(AnnotatedString(serverAddress))
                        Toast.makeText(context, "Endereço copiado!", Toast.LENGTH_SHORT).show()
                    },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(8.dp),
                    color = Color(0xFF0F6E56).copy(alpha = 0.15f)
                ) {
                    Row(
                        modifier = Modifier.padding(12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(Icons.Default.ContentCopy, "Copiar",
                            tint = Color(0xFF0F6E56), modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(8.dp))
                        Text(
                            serverAddress,
                            fontFamily = FontFamily.Monospace,
                            fontSize = 14.sp,
                            color = Color(0xFF76FF03),
                            fontWeight = FontWeight.Bold
                        )
                    }
                }

                if (useAuth && apiKey.isNotEmpty()) {
                    Spacer(Modifier.height(8.dp))
                    Surface(
                        onClick = {
                            clipboard.setText(AnnotatedString(apiKey))
                            Toast.makeText(context, "API Key copiada!", Toast.LENGTH_SHORT).show()
                        },
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(8.dp),
                        color = Color(0xFFFFD600).copy(alpha = 0.1f)
                    ) {
                        Row(
                            modifier = Modifier.padding(12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(Icons.Default.Key, "Key",
                                tint = Color(0xFFFFD600), modifier = Modifier.size(18.dp))
                            Spacer(Modifier.width(8.dp))
                            Text(
                                "Key: ${apiKey.take(8)}...${apiKey.takeLast(4)}",
                                fontFamily = FontFamily.Monospace,
                                fontSize = 13.sp,
                                color = Color(0xFFFFD600)
                            )
                        }
                    }
                }

                Spacer(Modifier.height(16.dp))
                Text("Endpoints:", fontSize = 13.sp, color = Color.Gray)
                Spacer(Modifier.height(4.dp))

                val endpoints = listOf(
                    "GET" to "/api/v1/status",
                    "GET" to "/api/v1/hardware",
                    "POST" to "/api/v1/detect",
                    "POST" to "/api/v1/classify",
                    "POST" to "/api/v1/llm/generate",
                    "POST" to "/api/v1/llm/chat",
                    "POST" to "/api/v1/inference/raw"
                )

                endpoints.forEach { (method, path) ->
                    val fullUrl = "$serverAddress$path"
                    Surface(
                        onClick = {
                            clipboard.setText(AnnotatedString(fullUrl))
                            Toast.makeText(context, "Copiado: $fullUrl", Toast.LENGTH_SHORT).show()
                        },
                        modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
                        shape = RoundedCornerShape(4.dp),
                        color = Color.Transparent
                    ) {
                        Row(
                            modifier = Modifier.padding(vertical = 4.dp, horizontal = 4.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                method,
                                fontSize = 10.sp,
                                fontWeight = FontWeight.Bold,
                                fontFamily = FontFamily.Monospace,
                                color = if (method == "GET") Color(0xFF0F6E56) else Color(0xFF534AB7),
                                modifier = Modifier.width(36.dp)
                            )
                            Text(
                                path,
                                fontSize = 12.sp,
                                fontFamily = FontFamily.Monospace,
                                color = Color(0xFF00E5FF)
                            )
                        }
                    }
                }

                Spacer(Modifier.height(12.dp))
                val curlExample = buildString {
                    append("curl ")
                    if (useAuth) append("-H \"X-API-Key: $apiKey\" ")
                    append("$serverAddress/api/v1/hardware")
                }
                Surface(
                    onClick = {
                        clipboard.setText(AnnotatedString(curlExample))
                        Toast.makeText(context, "curl copiado!", Toast.LENGTH_SHORT).show()
                    },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(8.dp),
                    color = Color(0xFF2A2A3E)
                ) {
                    Text(
                        curlExample,
                        modifier = Modifier.padding(12.dp),
                        fontSize = 11.sp,
                        fontFamily = FontFamily.Monospace,
                        color = Color(0xFFAABBCC)
                    )
                }
            }
        }
    }
}

private fun startServer(context: Context, port: Int, apiKey: String?) {
    val intent = Intent(context, InferenceService::class.java).apply {
        action = InferenceService.ACTION_START
        putExtra(InferenceService.EXTRA_PORT, port)
        apiKey?.let { putExtra(InferenceService.EXTRA_API_KEY, it) }
    }
    context.startForegroundService(intent)
}

private fun stopServer(context: Context) {
    val intent = Intent(context, InferenceService::class.java).apply {
        action = InferenceService.ACTION_STOP
    }
    context.startService(intent)
}

private fun getLocalIp(): String {
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
    } catch (_: Exception) {}
    return "0.0.0.0"
}
