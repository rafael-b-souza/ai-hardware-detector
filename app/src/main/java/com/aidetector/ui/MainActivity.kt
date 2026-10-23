package com.aidetector.ui

import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.aidetector.benchmark.BenchmarkResult
import com.aidetector.benchmark.BenchmarkRunner
import com.aidetector.detection.AcceleratorInfo
import com.aidetector.detection.AcceleratorType
import com.aidetector.detection.HardwareDetector
import com.aidetector.detection.ModelType
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            AIDetectorApp()
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AIDetectorApp() {
    val context = LocalContext.current
    val detector = remember { HardwareDetector(context) }
    val benchmarkRunner = remember { BenchmarkRunner(context) }
    val scope = rememberCoroutineScope()

    var accelerators by remember { mutableStateOf<List<AcceleratorInfo>>(emptyList()) }
    var benchmarkResults by remember { mutableStateOf<List<BenchmarkResult>>(emptyList()) }
    var isScanning by remember { mutableStateOf(false) }
    var isBenchmarking by remember { mutableStateOf(false) }
    var selectedModelType by remember { mutableStateOf(ModelType.OBJECT_DETECTION) }

    MaterialTheme(
        colorScheme = darkColorScheme(
            primary = Color(0xFF00E5FF),
            secondary = Color(0xFF76FF03),
            surface = Color(0xFF1A1A2E),
            background = Color(0xFF0F0F1A),
            onSurface = Color.White,
            error = Color(0xFFFF5252)
        )
    ) {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = {
                        Column {
                            Text("AI Hardware Detector", fontWeight = FontWeight.Bold)
                            Text("Detecta & otimiza aceleradores de IA",
                                fontSize = 12.sp, color = Color.Gray)
                        }
                    },
                    colors = TopAppBarDefaults.topAppBarColors(
                        containerColor = Color(0xFF0F0F1A)
                    )
                )
            }
        ) { padding ->
            LazyColumn(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(padding)
                    .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                item {
                    Button(
                        onClick = {
                            isScanning = true
                            accelerators = detector.detectAllAccelerators()
                            isScanning = false
                        },
                        modifier = Modifier.fillMaxWidth().height(56.dp),
                        shape = RoundedCornerShape(12.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = Color(0xFF00E5FF)
                        )
                    ) {
                        if (isScanning) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(24.dp),
                                color = Color.Black
                            )
                        } else {
                            Icon(Icons.Default.Search, "Scan", tint = Color.Black)
                            Spacer(Modifier.width(8.dp))
                            Text("DETECTAR HARDWARE", color = Color.Black,
                                fontWeight = FontWeight.Bold)
                        }
                    }
                }

                item {
                    Spacer(Modifier.height(8.dp))
                    Text("Servidor de Inferência",
                        fontSize = 20.sp, fontWeight = FontWeight.Bold,
                        color = Color(0xFFFF9100))
                }

                item {
                    ServerControlPanel()
                }

                if (accelerators.isNotEmpty()) {
                    item {
                        Text("Aceleradores Disponíveis",
                            fontSize = 20.sp, fontWeight = FontWeight.Bold,
                            color = Color(0xFF00E5FF))
                    }

                    items(accelerators) { accel ->
                        AcceleratorCard(accel)
                    }

                    item {
                        Spacer(Modifier.height(8.dp))
                        Text("Melhor Acelerador por Tipo de Modelo",
                            fontSize = 20.sp, fontWeight = FontWeight.Bold,
                            color = Color(0xFF76FF03))
                    }

                    item {
                        ModelRecommendations(detector)
                    }

                    item {
                        Spacer(Modifier.height(8.dp))
                        Text("Benchmark",
                            fontSize = 20.sp, fontWeight = FontWeight.Bold,
                            color = Color(0xFFFFD600))
                    }

                    item {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            ModelType.values().take(4).forEach { type ->
                                FilterChip(
                                    selected = selectedModelType == type,
                                    onClick = { selectedModelType = type },
                                    label = { Text(type.name.replace("_", " "), fontSize = 10.sp) }
                                )
                            }
                        }
                    }

                    item {
                        Button(
                            onClick = {
                                scope.launch {
                                    isBenchmarking = true
                                    try {
                                        benchmarkResults = benchmarkRunner.benchmarkModel(
                                            modelPath = "model.tflite",
                                            modelType = selectedModelType
                                        )
                                    } catch (e: Exception) {
                                        Toast.makeText(context,
                                            "Coloque um modelo .tflite em assets/",
                                            Toast.LENGTH_LONG).show()
                                    }
                                    isBenchmarking = false
                                }
                            },
                            modifier = Modifier.fillMaxWidth(),
                            shape = RoundedCornerShape(12.dp),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = Color(0xFFFFD600)
                            ),
                            enabled = !isBenchmarking
                        ) {
                            if (isBenchmarking) {
                                CircularProgressIndicator(Modifier.size(24.dp))
                                Spacer(Modifier.width(8.dp))
                                Text("Rodando benchmark...", color = Color.Black)
                            } else {
                                Text("RODAR BENCHMARK", color = Color.Black,
                                    fontWeight = FontWeight.Bold)
                            }
                        }
                    }

                    items(benchmarkResults) { result ->
                        BenchmarkCard(result)
                    }
                }
            }
        }
    }
}

@Composable
fun AcceleratorCard(accel: AcceleratorInfo) {
    val accentColor = when (accel.type) {
        AcceleratorType.NPU -> Color(0xFF00E5FF)
        AcceleratorType.GPU -> Color(0xFF76FF03)
        AcceleratorType.DSP -> Color(0xFFFFD600)
        AcceleratorType.CPU -> Color(0xFFFF9100)
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF1A1A2E))
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth()
            ) {
                Box(
                    modifier = Modifier
                        .size(48.dp)
                        .clip(RoundedCornerShape(12.dp))
                        .background(accentColor.copy(alpha = 0.2f)),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        accel.type.name,
                        color = accentColor,
                        fontWeight = FontWeight.Bold,
                        fontSize = 14.sp
                    )
                }
                Spacer(Modifier.width(12.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(accel.name, fontWeight = FontWeight.Bold, fontSize = 16.sp)
                    Text(
                        "Prioridade: ${accel.priority} | ${accel.estimatedTOPS} TOPS",
                        fontSize = 12.sp, color = Color.Gray
                    )
                }
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(8.dp))
                        .background(if (accel.available) Color(0xFF00C853).copy(0.2f)
                                    else Color(0xFFFF5252).copy(0.2f))
                        .padding(horizontal = 12.dp, vertical = 4.dp)
                ) {
                    Text(
                        if (accel.available) "ATIVO" else "OFF",
                        color = if (accel.available) Color(0xFF00C853) else Color(0xFFFF5252),
                        fontSize = 12.sp, fontWeight = FontWeight.Bold
                    )
                }
            }

            if (accel.supportedOps.isNotEmpty()) {
                Spacer(Modifier.height(8.dp))
                Text(
                    "Ops: ${accel.supportedOps.take(5).joinToString(", ")}${
                        if (accel.supportedOps.size > 5) " +${accel.supportedOps.size - 5}" else ""
                    }",
                    fontSize = 11.sp, color = Color.Gray
                )
            }

            Spacer(Modifier.height(4.dp))
            Text(accel.chipInfo, fontSize = 10.sp, color = Color.DarkGray)
        }
    }
}

@Composable
fun ModelRecommendations(detector: HardwareDetector) {
    val recommendations = ModelType.values().map { type ->
        type to detector.getBestAccelerator(type)
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = Color(0xFF1A1A2E))
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            recommendations.forEach { (model, accel) ->
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(
                        model.name.replace("_", " "),
                        fontSize = 13.sp
                    )
                    Text(
                        accel.name,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold,
                        color = when (accel) {
                            AcceleratorType.NPU -> Color(0xFF00E5FF)
                            AcceleratorType.GPU -> Color(0xFF76FF03)
                            AcceleratorType.DSP -> Color(0xFFFFD600)
                            AcceleratorType.CPU -> Color(0xFFFF9100)
                        }
                    )
                }
                if (model != ModelType.values().last()) {
                    Divider(color = Color(0xFF2A2A3E))
                }
            }
        }
    }
}

@Composable
fun BenchmarkCard(result: BenchmarkResult) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (result.success) Color(0xFF1A1A2E)
                            else Color(0xFF2E1A1A)
        )
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(result.delegateName, fontWeight = FontWeight.Bold)
                Text(
                    if (result.success) "%.1f FPS".format(result.fps) else "FALHOU",
                    fontWeight = FontWeight.Bold,
                    color = if (result.success) Color(0xFF76FF03) else Color(0xFFFF5252)
                )
            }
            if (result.success) {
                Spacer(Modifier.height(4.dp))
                Text(
                    "Média: %.2fms | Min: %.2fms | Max: %.2fms | Mem: %.1fMB".format(
                        result.avgInferenceMs, result.minInferenceMs,
                        result.maxInferenceMs, result.memoryUsageMb
                    ),
                    fontSize = 11.sp, color = Color.Gray
                )
            } else {
                Text("Erro: ${result.error}", fontSize = 11.sp, color = Color(0xFFFF5252))
            }
        }
    }
}
