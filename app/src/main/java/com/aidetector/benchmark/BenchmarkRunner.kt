package com.aidetector.benchmark

import android.content.Context
import android.os.SystemClock
import android.util.Log
import com.aidetector.delegates.DelegateFactory
import com.aidetector.detection.AcceleratorType
import com.aidetector.detection.HardwareDetector
import com.aidetector.detection.ModelType
import kotlinx.coroutines.*
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Mede a latência e consumo de memória dos aceleradores disponíveis para um modelo. */
data class BenchmarkResult(
    val accelerator: AcceleratorType,
    val delegateName: String,
    val modelName: String,
    val warmupTimeMs: Float,
    val avgInferenceMs: Float,
    val minInferenceMs: Float,
    val maxInferenceMs: Float,
    val fps: Float,
    val memoryUsageMb: Float,
    val success: Boolean,
    val error: String? = null
)

class BenchmarkRunner(private val context: Context) {

    private val delegateFactory = DelegateFactory(context)
    private val detector = HardwareDetector(context)

    companion object {
        private const val TAG = "BenchmarkRunner"
        private const val WARMUP_RUNS = 5
        private const val BENCHMARK_RUNS = 20
    }

    suspend fun benchmarkModel(
        modelPath: String,
        modelType: ModelType,
        inputShape: IntArray = intArrayOf(1, 224, 224, 3),
        inputType: Int = 1
    ): List<BenchmarkResult> = withContext(Dispatchers.Default) {
        val available = detector.detectAllAccelerators()
        val results = mutableListOf<BenchmarkResult>()

        Log.i(TAG, "Benchmark: $modelPath")
        Log.i(TAG, "Aceleradores disponíveis: ${available.map { it.type }}")

        for (accel in available) {
            val result = benchmarkSingleAccelerator(
                modelPath = modelPath,
                modelType = modelType,
                accelerator = accel.type,
                inputShape = inputShape,
                inputType = inputType
            )
            results.add(result)
            
            Log.i(TAG, "${accel.type}: ${result.avgInferenceMs}ms (${result.fps} FPS) - ${
                if (result.success) "OK" else "FAIL: ${result.error}"
            }")
        }

        results.sortedBy { if (it.success) it.avgInferenceMs else Float.MAX_VALUE }
    }

    private suspend fun benchmarkSingleAccelerator(
        modelPath: String,
        modelType: ModelType,
        accelerator: AcceleratorType,
        inputShape: IntArray,
        inputType: Int
    ): BenchmarkResult = withContext(Dispatchers.Default) {
        try {
            val result = delegateFactory.createInterpreter(
                modelPath = modelPath,
                modelType = modelType,
                forceAccelerator = accelerator
            )

            val interpreter = result.interpreter
            val inputBuffer = createInputBuffer(inputShape, inputType)
            val outputShape = interpreter.getOutputTensor(0).shape()
            val outputBuffer = ByteBuffer.allocateDirect(
                outputShape.fold(1) { acc, i -> acc * i } * 4
            ).order(ByteOrder.nativeOrder())

            val warmupStart = SystemClock.elapsedRealtimeNanos()
            repeat(WARMUP_RUNS) {
                inputBuffer.rewind()
                outputBuffer.rewind()
                interpreter.run(inputBuffer, outputBuffer)
            }
            val warmupMs = (SystemClock.elapsedRealtimeNanos() - warmupStart) / 1_000_000f

            val times = mutableListOf<Float>()
            val runtime = Runtime.getRuntime()
            val memBefore = runtime.totalMemory() - runtime.freeMemory()

            repeat(BENCHMARK_RUNS) {
                inputBuffer.rewind()
                outputBuffer.rewind()

                val start = SystemClock.elapsedRealtimeNanos()
                interpreter.run(inputBuffer, outputBuffer)
                val elapsed = (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000f
                times.add(elapsed)
            }

            val memAfter = runtime.totalMemory() - runtime.freeMemory()
            val memUsageMb = (memAfter - memBefore) / (1024f * 1024f)

            interpreter.close()

            val avgMs = times.average().toFloat()

            BenchmarkResult(
                accelerator = accelerator,
                delegateName = result.delegateName,
                modelName = modelPath.substringAfterLast("/"),
                warmupTimeMs = warmupMs / WARMUP_RUNS,
                avgInferenceMs = avgMs,
                minInferenceMs = times.min(),
                maxInferenceMs = times.max(),
                fps = 1000f / avgMs,
                memoryUsageMb = memUsageMb.coerceAtLeast(0f),
                success = true
            )
        } catch (e: Exception) {
            Log.e(TAG, "Benchmark falhou para $accelerator: ${e.message}")
            BenchmarkResult(
                accelerator = accelerator,
                delegateName = accelerator.name,
                modelName = modelPath,
                warmupTimeMs = 0f,
                avgInferenceMs = Float.MAX_VALUE,
                minInferenceMs = Float.MAX_VALUE,
                maxInferenceMs = Float.MAX_VALUE,
                fps = 0f,
                memoryUsageMb = 0f,
                success = false,
                error = e.message
            )
        } finally {
            delegateFactory.releaseAll()
        }
    }

    suspend fun findBestAccelerator(
        modelPath: String,
        modelType: ModelType,
        inputShape: IntArray = intArrayOf(1, 224, 224, 3)
    ): AcceleratorType {
        val results = benchmarkModel(modelPath, modelType, inputShape)
        return results
            .filter { it.success }
            .minByOrNull { it.avgInferenceMs }
            ?.accelerator
            ?: AcceleratorType.CPU
    }

    private fun createInputBuffer(shape: IntArray, inputType: Int): ByteBuffer {
        val totalElements = shape.fold(1) { acc, i -> acc * i }
        val bytesPerElement = if (inputType == 1) 4 else 1

        return ByteBuffer.allocateDirect(totalElements * bytesPerElement).apply {
            order(ByteOrder.nativeOrder())
            val random = java.util.Random(42)
            if (inputType == 1) {
                val floatBuffer = asFloatBuffer()
                repeat(totalElements) {
                    floatBuffer.put(random.nextFloat())
                }
            } else {
                repeat(totalElements) {
                    put(random.nextInt(256).toByte())
                }
            }
            rewind()
        }
    }
}
