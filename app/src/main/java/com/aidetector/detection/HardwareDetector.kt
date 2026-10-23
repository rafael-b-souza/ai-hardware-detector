package com.aidetector.detection

import android.content.Context
import android.os.Build
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File

/** Detecta aceleradores de hardware disponíveis para inferência (NPU, GPU, DSP e CPU). */
data class AcceleratorInfo(
    val type: AcceleratorType,
    val name: String,
    val available: Boolean,
    val estimatedTOPS: Float,
    val supportedOps: List<String>,
    val chipInfo: String,
    val priority: Int
)

enum class AcceleratorType {
    NPU,    // Neural Processing Unit (NNAPI)
    GPU,    // GPU Delegate (OpenCL/Vulkan)
    DSP,    // Hexagon DSP (Qualcomm)
    CPU     // CPU com XNNPACK
}

enum class ModelType {
    IMAGE_CLASSIFICATION,
    OBJECT_DETECTION,
    VIDEO_DETECTION,
    LLM_INFERENCE,
    POSE_ESTIMATION,
    SEGMENTATION,
    OCR,
    AUDIO_CLASSIFICATION
}

class HardwareDetector(private val context: Context) {

    companion object {
        private const val TAG = "HardwareDetector"
        
        // Chipsets conhecidos e suas capacidades
        private val CHIPSET_NPU_INFO = mapOf(
            // Qualcomm Snapdragon
            "SM-S908" to ChipInfo("Snapdragon 8 Gen 1", 27f, true),   // S22 Ultra
            "SM-S906" to ChipInfo("Snapdragon 8 Gen 1", 27f, true),   // S22+
            "SM-S901" to ChipInfo("Snapdragon 8 Gen 1", 27f, true),   // S22
            "SM-G998" to ChipInfo("Snapdragon 888", 26f, true),       // S21 Ultra
            "SM-G996" to ChipInfo("Snapdragon 888", 26f, true),       // S21+
            "SM-G991" to ChipInfo("Snapdragon 888", 26f, true),       // S21
            "SM-N950" to ChipInfo("Snapdragon 835", 0f, false),       // Note 8 (sem NPU dedicada)
            // Exynos variants
            "Exynos 2200" to ChipInfo("Exynos 2200 (Xclipse GPU)", 22f, true),
            "Exynos 2100" to ChipInfo("Exynos 2100", 26f, true),
        )
    }

    data class ChipInfo(
        val name: String,
        val npuTops: Float,
        val hasNpu: Boolean
    )

    fun detectAllAccelerators(): List<AcceleratorInfo> {
        val accelerators = mutableListOf<AcceleratorInfo>()
        accelerators.add(detectNPU())
        accelerators.add(detectGPU())
        accelerators.add(detectDSP())
        accelerators.add(detectCPU())
        
        return accelerators
            .filter { it.available }
            .sortedBy { it.priority }
    }

    private fun detectNPU(): AcceleratorInfo {
        val model = Build.MODEL
        val soc = Build.SOC_MODEL
        val board = Build.BOARD
        val hardware = Build.HARDWARE
        
        val nnapiAvailable = Build.VERSION.SDK_INT >= 27
        val chipInfo = identifyChipset(model, soc)
        val npuFunctional = if (nnapiAvailable && chipInfo?.hasNpu == true) {
            testNNAPIDelegate()
        } else false
        
        return AcceleratorInfo(
            type = AcceleratorType.NPU,
            name = chipInfo?.name?.let { "NPU ($it)" } ?: "NNAPI NPU",
            available = npuFunctional,
            estimatedTOPS = chipInfo?.npuTops ?: 0f,
            supportedOps = if (npuFunctional) listOf(
                "CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED",
                "SOFTMAX", "RESHAPE", "ADD", "MUL", "RELU",
                "AVERAGE_POOL_2D", "MAX_POOL_2D", "CONCATENATION",
                "BATCH_NORM", "QUANTIZED_OPS"
            ) else emptyList(),
            chipInfo = "SoC: ${soc.ifEmpty { board }} | Model: $model | HW: $hardware",
            priority = 1
        )
    }

    private fun detectGPU(): AcceleratorInfo {
        val gpuCompatList = CompatibilityList()
        val gpuAvailable = gpuCompatList.isDelegateSupportedOnThisDevice
        
        val gpuName = when {
            Build.SOC_MODEL.contains("8 Gen 1", true) -> "Adreno 730"
            Build.SOC_MODEL.contains("888", true) -> "Adreno 660"
            Build.HARDWARE.contains("qcom", true) && 
                Build.MODEL.startsWith("SM-N950") -> "Adreno 540"
            else -> "GPU (${Build.HARDWARE})"
        }

        return AcceleratorInfo(
            type = AcceleratorType.GPU,
            name = "GPU ($gpuName)",
            available = gpuAvailable,
            estimatedTOPS = when {
                gpuName.contains("730") -> 3.7f
                gpuName.contains("660") -> 3.4f
                gpuName.contains("540") -> 1.5f
                else -> 1.0f
            },
            supportedOps = if (gpuAvailable) listOf(
                "CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED",
                "ADD", "MUL", "RELU", "SOFTMAX", "RESHAPE",
                "TRANSPOSE_CONV", "PAD", "FLOAT32_OPS"
            ) else emptyList(),
            chipInfo = "OpenCL: ${gpuCompatList.isDelegateSupportedOnThisDevice}",
            priority = 2
        )
    }

    private fun detectDSP(): AcceleratorInfo {
        val isQualcomm = Build.HARDWARE.contains("qcom", true) ||
                         Build.SOC_MANUFACTURER.contains("Qualcomm", true)
        
        val hexagonAvailable = isQualcomm && checkHexagonLibs()
        
        return AcceleratorInfo(
            type = AcceleratorType.DSP,
            name = "Hexagon DSP",
            available = hexagonAvailable,
            estimatedTOPS = when {
                Build.SOC_MODEL.contains("8 Gen 1", true) -> 15f
                Build.SOC_MODEL.contains("888", true) -> 15f
                Build.MODEL.startsWith("SM-N950") -> 0.5f
                else -> 1f
            },
            supportedOps = if (hexagonAvailable) listOf(
                "QUANTIZED_CONV_2D", "QUANTIZED_DEPTHWISE_CONV",
                "QUANTIZED_FULLY_CONNECTED", "QUANTIZED_ADD",
                "INT8_OPS", "UINT8_OPS"
            ) else emptyList(),
            chipInfo = "Qualcomm: $isQualcomm",
            priority = 3
        )
    }

    private fun detectCPU(): AcceleratorInfo {
        val cpuCores = Runtime.getRuntime().availableProcessors()
        val cpuArch = Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown"
        
        return AcceleratorInfo(
            type = AcceleratorType.CPU,
            name = "CPU ($cpuArch, ${cpuCores} cores)",
            available = true,
            estimatedTOPS = cpuCores * 0.05f,
            supportedOps = listOf("ALL_OPS"),
            chipInfo = "Cores: $cpuCores | Arch: $cpuArch | ABI: ${Build.SUPPORTED_ABIS.joinToString()}",
            priority = 10
        )
    }

    fun getBestAccelerator(modelType: ModelType): AcceleratorType {
        val available = detectAllAccelerators()
        val hasNPU = available.any { it.type == AcceleratorType.NPU }
        val hasGPU = available.any { it.type == AcceleratorType.GPU }
        val hasDSP = available.any { it.type == AcceleratorType.DSP }

        return when (modelType) {
            ModelType.IMAGE_CLASSIFICATION,
            ModelType.OBJECT_DETECTION,
            ModelType.VIDEO_DETECTION,
            ModelType.POSE_ESTIMATION,
            ModelType.SEGMENTATION -> when {
                hasNPU -> AcceleratorType.NPU
                hasGPU -> AcceleratorType.GPU
                else -> AcceleratorType.CPU
            }

            ModelType.LLM_INFERENCE -> when {
                hasGPU -> AcceleratorType.GPU
                hasNPU -> AcceleratorType.NPU
                else -> AcceleratorType.CPU
            }

            ModelType.OCR -> when {
                hasNPU -> AcceleratorType.NPU
                hasDSP -> AcceleratorType.DSP
                hasGPU -> AcceleratorType.GPU
                else -> AcceleratorType.CPU
            }

            ModelType.AUDIO_CLASSIFICATION -> when {
                hasDSP -> AcceleratorType.DSP
                hasNPU -> AcceleratorType.NPU
                else -> AcceleratorType.CPU
            }
        }
    }

    private fun identifyChipset(model: String, soc: String): ChipInfo? {
        val prefix = model.take(7)
        CHIPSET_NPU_INFO[prefix]?.let { return it }
        
        CHIPSET_NPU_INFO.entries.find { (_, info) ->
            soc.contains(info.name, true)
        }?.value?.let { return it }
        
        return null
    }

    private fun testNNAPIDelegate(): Boolean {
        return try {
            val delegate = NnApiDelegate(
                NnApiDelegate.Options().apply {
                    setAllowFp16(true)
                    setUseNnapiCpu(false)  // Forçar acelerador de hardware
                }
            )
            delegate.close()
            true
        } catch (e: Exception) {
            Log.w(TAG, "NNAPI delegate não disponível: ${e.message}")
            false
        }
    }

    private fun checkHexagonLibs(): Boolean {
        return try {
            val libPaths = listOf(
                "/vendor/lib64/libhexagon_nn_skel.so",
                "/vendor/lib64/libQnnHtp.so",
                "/vendor/lib/rfsa/adsp/libhexagon_nn_skel.so"
            )
            libPaths.any { File(it).exists() }
        } catch (e: SecurityException) {
            // Se não temos permissão, assumir disponível em Qualcomm
            Build.HARDWARE.contains("qcom", true)
        }
    }
}
