package com.aidetector.delegates

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import com.aidetector.detection.AcceleratorType
import com.aidetector.detection.HardwareDetector
import com.aidetector.detection.ModelType
import java.io.FileInputStream
import java.io.File
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/** Inicializa interpreters TFLite vinculados ao melhor acelerador disponível no dispositivo. */
class DelegateFactory(private val context: Context) {

    private val detector = HardwareDetector(context)
    private val activeDelegates = mutableListOf<AutoCloseable>()

    companion object {
        private const val TAG = "DelegateFactory"
    }

    data class InterpreterResult(
        val interpreter: Interpreter,
        val acceleratorUsed: AcceleratorType,
        val delegateName: String
    )

    fun createInterpreter(
        modelPath: String,
        modelType: ModelType,
        forceAccelerator: AcceleratorType? = null,
        numThreads: Int = 4
    ): InterpreterResult {
        val modelBuffer = loadModelFile(modelPath)
        val targetAccelerator = forceAccelerator ?: detector.getBestAccelerator(modelType)

        Log.i(TAG, "Criando interpreter para $modelType -> Alvo: $targetAccelerator")

        val fallbackChain = buildFallbackChain(targetAccelerator)
        
        for (accel in fallbackChain) {
            try {
                val result = createWithAccelerator(modelBuffer, accel, numThreads)
                Log.i(TAG, "Usando delegate: ${result.delegateName}")
                return result
            } catch (e: Exception) {
                Log.w(TAG, "Falha com $accel: ${e.message}, tentando proximo...")
            }
        }

        Log.w(TAG, "Aceleradores falharam, usando CPU")
        return createCPUInterpreter(modelBuffer, numThreads)
    }

    private fun createWithAccelerator(
        modelBuffer: MappedByteBuffer,
        accelerator: AcceleratorType,
        numThreads: Int
    ): InterpreterResult {
        return when (accelerator) {
            AcceleratorType.NPU -> createNNAPIInterpreter(modelBuffer, numThreads)
            AcceleratorType.GPU -> createGPUInterpreter(modelBuffer, numThreads)
            AcceleratorType.DSP -> createDSPInterpreter(modelBuffer, numThreads)
            AcceleratorType.CPU -> createCPUInterpreter(modelBuffer, numThreads)
        }
    }

    private fun createNNAPIInterpreter(
        modelBuffer: MappedByteBuffer,
        numThreads: Int
    ): InterpreterResult {
        val nnapiOptions = NnApiDelegate.Options().apply {
            setAllowFp16(true)
            setUseNnapiCpu(false)
            setExecutionPreference(
                NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED
            )
            setCacheDir(context.cacheDir.absolutePath)
            setModelToken("aidetector_${modelBuffer.hashCode()}")
        }

        val delegate = NnApiDelegate(nnapiOptions)
        activeDelegates.add(delegate)

        val options = Interpreter.Options().apply {
            addDelegate(delegate)
            setNumThreads(numThreads)
            setUseXNNPACK(true)
        }

        return InterpreterResult(
            interpreter = Interpreter(modelBuffer, options),
            acceleratorUsed = AcceleratorType.NPU,
            delegateName = "NNAPI NPU Delegate (FP16)"
        )
    }

    private fun createGPUInterpreter(
        modelBuffer: MappedByteBuffer,
        numThreads: Int
    ): InterpreterResult {
        val compatList = CompatibilityList()
        
        if (!compatList.isDelegateSupportedOnThisDevice) {
            throw UnsupportedOperationException("GPU delegate não suportado")
        }

        val gpuOptions = GpuDelegate.Options().apply {
            setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
            setPrecisionLossAllowed(true)
        }

        val delegate = GpuDelegate(gpuOptions)
        activeDelegates.add(delegate)

        val options = Interpreter.Options().apply {
            addDelegate(delegate)
            setNumThreads(numThreads)
        }

        return InterpreterResult(
            interpreter = Interpreter(modelBuffer, options),
            acceleratorUsed = AcceleratorType.GPU,
            delegateName = "GPU Delegate (OpenCL, FP16)"
        )
    }

    private fun createDSPInterpreter(
        modelBuffer: MappedByteBuffer,
        numThreads: Int
    ): InterpreterResult {
        val nnapiOptions = NnApiDelegate.Options().apply {
            setAllowFp16(false)
            setUseNnapiCpu(false)
            setAcceleratorName("google-edgetpu")
            setExecutionPreference(
                NnApiDelegate.Options.EXECUTION_PREFERENCE_FAST_SINGLE_ANSWER
            )
        }

        val delegate = NnApiDelegate(nnapiOptions)
        activeDelegates.add(delegate)

        val options = Interpreter.Options().apply {
            addDelegate(delegate)
            setNumThreads(numThreads)
        }

        return InterpreterResult(
            interpreter = Interpreter(modelBuffer, options),
            acceleratorUsed = AcceleratorType.DSP,
            delegateName = "Hexagon DSP Delegate (INT8)"
        )
    }

    private fun createCPUInterpreter(
        modelBuffer: MappedByteBuffer,
        numThreads: Int
    ): InterpreterResult {
        val options = Interpreter.Options().apply {
            setNumThreads(numThreads)
            setUseXNNPACK(true)
            setAllowFp16PrecisionForFp32(true)
        }

        return InterpreterResult(
            interpreter = Interpreter(modelBuffer, options),
            acceleratorUsed = AcceleratorType.CPU,
            delegateName = "CPU XNNPACK ($numThreads threads)"
        )
    }

    private fun buildFallbackChain(target: AcceleratorType): List<AcceleratorType> {
        return when (target) {
            AcceleratorType.NPU -> listOf(AcceleratorType.NPU, AcceleratorType.GPU, AcceleratorType.CPU)
            AcceleratorType.GPU -> listOf(AcceleratorType.GPU, AcceleratorType.NPU, AcceleratorType.CPU)
            AcceleratorType.DSP -> listOf(AcceleratorType.DSP, AcceleratorType.NPU, AcceleratorType.CPU)
            AcceleratorType.CPU -> listOf(AcceleratorType.CPU)
        }
    }

    private fun loadModelFile(modelPath: String): MappedByteBuffer {
        return if (File(modelPath).exists()) {
            FileInputStream(File(modelPath)).use { fis ->
                fis.channel.map(FileChannel.MapMode.READ_ONLY, 0, File(modelPath).length())
            }
        } else {
            val assetFd = context.assets.openFd(modelPath)
            FileInputStream(assetFd.fileDescriptor).use { fis ->
                fis.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    assetFd.startOffset,
                    assetFd.declaredLength
                )
            }
        }
    }

    fun releaseAll() {
        activeDelegates.forEach { delegate ->
            try {
                delegate.close()
            } catch (e: Exception) {
                Log.w(TAG, "Erro ao fechar delegate: ${e.message}")
            }
        }
        activeDelegates.clear()
    }
}
