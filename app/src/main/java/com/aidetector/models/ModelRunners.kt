package com.aidetector.models

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.media.Image
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.aidetector.delegates.DelegateFactory
import com.aidetector.detection.AcceleratorType
import com.aidetector.detection.ModelType
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

data class Detection(
    val label: String,
    val confidence: Float,
    val boundingBox: RectF
)

class VideoObjectDetector(context: Context) {
    
    private val factory = DelegateFactory(context)
    private var interpreterResult: DelegateFactory.InterpreterResult? = null
    private val labels = mutableListOf<String>()
    
    companion object {
        private const val TAG = "VideoObjectDetector"
        private const val INPUT_SIZE = 300
        private const val CONFIDENCE_THRESHOLD = 0.5f
        private const val MAX_DETECTIONS = 10
    }

    fun initialize(
        modelPath: String = "ssd_mobilenet_v2.tflite",
        labelsPath: String = "coco_labels.txt",
        forceAccelerator: AcceleratorType? = null
    ) {
        interpreterResult = factory.createInterpreter(
            modelPath = modelPath,
            modelType = ModelType.VIDEO_DETECTION,
            forceAccelerator = forceAccelerator
        )
        Log.i(TAG, "Detector inicializado: ${interpreterResult?.delegateName}")
    }

    fun detectFrame(bitmap: Bitmap): List<Detection> {
        val interpreter = interpreterResult?.interpreter
            ?: throw IllegalStateException("Detector não inicializado")

        val inputBuffer = preprocessBitmap(bitmap)

        val outputLocations = Array(1) { Array(MAX_DETECTIONS) { FloatArray(4) } }
        val outputClasses = Array(1) { FloatArray(MAX_DETECTIONS) }
        val outputScores = Array(1) { FloatArray(MAX_DETECTIONS) }
        val numDetections = FloatArray(1)

        val outputs = mapOf(
            0 to outputLocations,
            1 to outputClasses,
            2 to outputScores,
            3 to numDetections
        )

        interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)

        val detections = mutableListOf<Detection>()
        val count = numDetections[0].toInt().coerceAtMost(MAX_DETECTIONS)

        for (i in 0 until count) {
            val score = outputScores[0][i]
            if (score >= CONFIDENCE_THRESHOLD) {
                val box = outputLocations[0][i]
                detections.add(Detection(
                    label = getLabelForIndex(outputClasses[0][i].toInt()),
                    confidence = score,
                    boundingBox = RectF(box[1], box[0], box[3], box[2])
                ))
            }
        }

        return detections.sortedByDescending { it.confidence }
    }

    fun detectCameraFrame(image: Image): List<Detection> {
        val bitmap = yuvToBitmap(image)
        return detectFrame(bitmap)
    }

    fun getAcceleratorUsed(): String = interpreterResult?.delegateName ?: "Não inicializado"

    fun release() {
        interpreterResult?.interpreter?.close()
        factory.releaseAll()
    }

    private fun preprocessBitmap(bitmap: Bitmap): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val buffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3)
            .order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        for (pixel in pixels) {
            buffer.put(((pixel shr 16) and 0xFF).toByte())
            buffer.put(((pixel shr 8) and 0xFF).toByte())
            buffer.put((pixel and 0xFF).toByte())
        }
        buffer.rewind()
        return buffer
    }

    private fun getLabelForIndex(index: Int): String {
        return if (index < labels.size) labels[index] else "Objeto $index"
    }

    private fun yuvToBitmap(image: Image): Bitmap {
        val width = image.width
        val height = image.height
        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    }
}

data class LLMResponse(
    val text: String,
    val tokensGenerated: Int,
    val tokensPerSecond: Float,
    val acceleratorUsed: String
)

class LocalLLMRunner(private val context: Context) {

    private val factory = DelegateFactory(context)
    private var interpreterResult: DelegateFactory.InterpreterResult? = null

    companion object {
        private const val TAG = "LocalLLMRunner"
        private const val MAX_SEQ_LENGTH = 512
        private const val VOCAB_SIZE = 32000
    }

    fun initialize(modelPath: String = "gemma_2b_int8.tflite") {
        interpreterResult = factory.createInterpreter(
            modelPath = modelPath,
            modelType = ModelType.LLM_INFERENCE,
        )
        Log.i(TAG, "LLM inicializado: ${interpreterResult?.delegateName}")
    }

    fun generate(
        prompt: String,
        maxTokens: Int = 128,
        temperature: Float = 0.7f
    ): LLMResponse {
        val interpreter = interpreterResult?.interpreter
            ?: throw IllegalStateException("LLM não inicializado")

        val startTime = System.nanoTime()
        val tokens = tokenize(prompt)
        val generatedTokens = mutableListOf<Int>()

        var inputIds = tokens.toIntArray()
        
        for (step in 0 until maxTokens) {
            val inputBuffer = createTokenBuffer(inputIds)
            val outputLogits = Array(1) { FloatArray(VOCAB_SIZE) }

            interpreter.run(inputBuffer, outputLogits)

            val nextToken = sampleWithTemperature(outputLogits[0], temperature)
            if (nextToken == 2) break
            
            generatedTokens.add(nextToken)
            inputIds = intArrayOf(*inputIds, nextToken)
            
            if (inputIds.size > MAX_SEQ_LENGTH) {
                inputIds = inputIds.takeLast(MAX_SEQ_LENGTH).toIntArray()
            }
        }

        val elapsedMs = (System.nanoTime() - startTime) / 1_000_000f
        val tokPerSec = generatedTokens.size / (elapsedMs / 1000f)

        return LLMResponse(
            text = detokenize(generatedTokens),
            tokensGenerated = generatedTokens.size,
            tokensPerSecond = tokPerSec,
            acceleratorUsed = interpreterResult?.delegateName ?: "unknown"
        )
    }

    fun release() {
        interpreterResult?.interpreter?.close()
        factory.releaseAll()
    }

    private fun tokenize(text: String): List<Int> {
        return text.toCharArray().map { it.code % VOCAB_SIZE }
    }

    private fun detokenize(tokens: List<Int>): String {
        return tokens.map { it.toChar() }.joinToString("")
    }

    private fun createTokenBuffer(ids: IntArray): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(ids.size * 4)
            .order(ByteOrder.nativeOrder())
        ids.forEach { buffer.putInt(it) }
        buffer.rewind()
        return buffer
    }

    private fun sampleWithTemperature(logits: FloatArray, temperature: Float): Int {
        if (temperature <= 0f) return logits.indices.maxByOrNull { logits[it] } ?: 0

        val scaled = logits.map { it / temperature }
        val maxLogit = scaled.max()
        val exps = scaled.map { Math.exp((it - maxLogit).toDouble()) }
        val sumExps = exps.sum()
        val probs = exps.map { (it / sumExps).toFloat() }

        val random = Math.random().toFloat()
        var cumulative = 0f
        for (i in probs.indices) {
            cumulative += probs[i]
            if (random <= cumulative) return i
        }
        return probs.size - 1
    }
}

data class ClassificationResult(
    val label: String,
    val confidence: Float
)

/** Remoção de fundo usando RMBG-1.4 em formato TFLite (512x512 float32). */
class BackgroundRemover(private val context: Context) {

    private val factory = DelegateFactory(context)
    private var interpreterResult: DelegateFactory.InterpreterResult? = null

    companion object {
        private const val TAG = "BackgroundRemover"
        const val INPUT_SIZE = 512
    }

    fun initialize(modelPath: String = "rmbg2.tflite", forceAccelerator: AcceleratorType? = null) {
        interpreterResult = factory.createInterpreter(
            modelPath = modelPath,
            modelType = ModelType.SEGMENTATION,
            forceAccelerator = forceAccelerator
        )
        Log.i(TAG, "BackgroundRemover initialized: ${interpreterResult?.delegateName}")
    }

    fun removeBackground(bitmap: Bitmap): Bitmap {
        val interpreter = interpreterResult?.interpreter
            ?: throw IllegalStateException("BackgroundRemover not initialized")

        val inputBuffer = preprocessBitmap(bitmap)

        val outTensor = interpreter.getOutputTensor(0)
        val outShape = outTensor.shape()
        Log.d(TAG, "Output tensor: shape=${outShape.toList()} dtype=${outTensor.dataType()}")

        val mask: Array<Array<Array<FloatArray>>>
        if (outShape.size == 4 && outShape[1] == 1) {
            // NCHW: [1, 1, H, W]
            mask = Array(1) { Array(1) { Array(outShape[2]) { FloatArray(outShape[3]) } } }
            interpreter.run(inputBuffer, mask)
        } else {
            // NHWC: [1, H, W, 1]
            val tmp = Array(1) { Array(outShape[1]) { Array(outShape[2]) { FloatArray(outShape[3]) } } }
            interpreter.run(inputBuffer, tmp)
            val h = outShape[1]; val w = outShape[2]
            mask = Array(1) { Array(1) { Array(h) { FloatArray(w) } } }
            for (y in 0 until h) for (x in 0 until w) mask[0][0][y][x] = tmp[0][y][x][0]
        }

        // Sigmoid + min-max normalization
        val h = mask[0][0].size; val w = mask[0][0][0].size
        var ma = Float.MIN_VALUE; var mi = Float.MAX_VALUE
        for (y in 0 until h) for (x in 0 until w) {
            val v = 1f / (1f + Math.exp(-mask[0][0][y][x].toDouble()).toFloat())
            mask[0][0][y][x] = v
            if (v > ma) ma = v
            if (v < mi) mi = v
        }
        val range = if (ma - mi > 0f) ma - mi else 1f
        Log.d(TAG, "Mask sigmoid: min=$mi max=$ma range=$range")
        for (y in 0 until h) for (x in 0 until w)
            mask[0][0][y][x] = (mask[0][0][y][x] - mi) / range

        return applyMask(bitmap, mask)
    }

    private fun preprocessBitmap(bitmap: Bitmap): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        // rmbg2.tflite espera float32 [1, 512, 512, 3] NHWC, ImageNet normalized
        val buffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
            .order(ByteOrder.nativeOrder())
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        // RMBG-1.4 normalization: (pixel/255 - 0.5) / 1.0
        for (pixel in pixels) {
            for (c in 0..2) {
                val shift = 16 - c * 8
                val v = ((pixel shr shift) and 0xFF) / 255.0f
                buffer.putFloat(v - 0.5f)
            }
        }
        buffer.rewind()
        return buffer
    }

    private fun applyMask(original: Bitmap, mask: Array<Array<Array<FloatArray>>>): Bitmap {
        val w = original.width; val h = original.height
        val maskH = mask[0][0].size; val maskW = mask[0][0][0].size
        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val origPixels = IntArray(w * h)
        original.getPixels(origPixels, 0, w, 0, 0, w, h)
        for (i in origPixels.indices) {
            val py = (i / w) * maskH / h
            val px = (i % w) * maskW / w
            val alpha = (mask[0][0][py][px].coerceIn(0f, 1f) * 255).toInt()
            origPixels[i] = (origPixels[i] and 0x00FFFFFF) or (alpha shl 24)
        }
        result.setPixels(origPixels, 0, w, 0, 0, w, h)
        return result
    }

    fun getAcceleratorUsed(): String = interpreterResult?.delegateName ?: "N/A"

    fun release() {
        interpreterResult?.interpreter?.close()
        factory.releaseAll()
    }
}

/** Remoção de fundo usando modelos ONNX via ONNX Runtime com aceleração NNAPI/CPU. */
class OrtBackgroundRemover(private val context: Context) {

    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null
    private var inputName: String = "input"
    private var inputHeight: Int = 1024
    private var inputWidth: Int = 1024
    private var acceleratorName: String = "CPU"

    companion object {
        private const val TAG = "OrtBackgroundRemover"
    }

    fun initialize(modelPath: String) {
        val file = File(context.filesDir, modelPath)
        if (!file.exists()) throw IllegalStateException("Model not found: ${file.absolutePath}")

        env = OrtEnvironment.getEnvironment()
        val opts = OrtSession.SessionOptions()

        // Try NNAPI first, fall back to CPU
        try {
            opts.addNnapi()
            acceleratorName = "NNAPI"
            Log.i(TAG, "NNAPI EP enabled")
        } catch (e: Exception) {
            Log.w(TAG, "NNAPI unavailable: ${e.message}")
            acceleratorName = "CPU"
        }

        session = env!!.createSession(file.absolutePath, opts)

        // Discover input name and shape from model
        val inputInfo = session!!.inputInfo
        inputName = inputInfo.keys.first()
        val shape = (inputInfo[inputName]!!.info as? ai.onnxruntime.TensorInfo)?.shape
        if (shape != null && shape.size == 4) {
            // NCHW: [1, 3, H, W]
            inputHeight = shape[2].toInt()
            inputWidth  = shape[3].toInt()
        }
        Log.i(TAG, "ORT session ready: input='$inputName' shape=[1,3,$inputHeight,$inputWidth] accel=$acceleratorName")

        // Warmup — NNAPI compiles the graph on first run
        runWarmup()
    }

    private fun runWarmup() {
        try {
            val buf = FloatBuffer.allocate(1 * 3 * inputHeight * inputWidth)
            val tensor = OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong()))
            val result = session!!.run(mapOf(inputName to tensor))
            result.close()
            tensor.close()
            Log.i(TAG, "Warmup complete")
        } catch (e: Exception) {
            Log.w(TAG, "Warmup failed (non-fatal): ${e.message}")
        }
    }

    fun removeBackground(bitmap: Bitmap): Bitmap {
        val sess = session ?: throw IllegalStateException("OrtBackgroundRemover not initialized")

        val inputBuffer = preprocessBitmap(bitmap)
        val tensor = OnnxTensor.createTensor(
            env, inputBuffer,
            longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong())
        )

        val results = sess.run(mapOf(inputName to tensor))
        val outTensor = results.get(0) as OnnxTensor
        val raw = outTensor.floatBuffer

        // Determine output shape
        val outShape = outTensor.info.shape
        val h: Int; val w: Int
        if (outShape.size == 4 && outShape[1] == 1L) {
            // NCHW [1, 1, H, W]
            h = outShape[2].toInt(); w = outShape[3].toInt()
        } else {
            // NHWC [1, H, W, 1]
            h = outShape[1].toInt(); w = outShape[2].toInt()
        }

        // Sigmoid + min-max normalization
        val mask = FloatArray(h * w)
        var mi = Float.MAX_VALUE; var ma = -Float.MAX_VALUE
        for (i in mask.indices) {
            val v = (1f / (1f + Math.exp(-raw.get().toDouble()))).toFloat()
            mask[i] = v
            if (v < mi) mi = v
            if (v > ma) ma = v
        }
        val range = if (ma - mi > 0f) ma - mi else 1f
        Log.d(TAG, "Mask sigmoid: min=$mi max=$ma range=$range accel=$acceleratorName")
        for (i in mask.indices) mask[i] = (mask[i] - mi) / range

        results.close()
        tensor.close()

        return applyMask(bitmap, mask, h, w)
    }

    private fun preprocessBitmap(bitmap: Bitmap): FloatBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, inputWidth, inputHeight, true)
        val pixels = IntArray(inputWidth * inputHeight)
        resized.getPixels(pixels, 0, inputWidth, 0, 0, inputWidth, inputHeight)
        // NCHW: R plane, G plane, B plane — (pixel/255 - 0.5) / 0.5
        val buf = FloatBuffer.allocate(1 * 3 * inputHeight * inputWidth)
        for (c in 0..2) {
            val shift = 16 - c * 8
            for (pixel in pixels) {
                val v = ((pixel shr shift) and 0xFF) / 255.0f
                buf.put((v - 0.5f) / 0.5f)
            }
        }
        buf.rewind()
        return buf
    }

    private fun applyMask(original: Bitmap, mask: FloatArray, maskH: Int, maskW: Int): Bitmap {
        val w = original.width; val h = original.height
        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val origPixels = IntArray(w * h)
        original.getPixels(origPixels, 0, w, 0, 0, w, h)
        for (i in origPixels.indices) {
            val py = (i / w) * maskH / h
            val px = (i % w) * maskW / w
            val alpha = (mask[py * maskW + px].coerceIn(0f, 1f) * 255).toInt()
            origPixels[i] = (origPixels[i] and 0x00FFFFFF) or (alpha shl 24)
        }
        result.setPixels(origPixels, 0, w, 0, 0, w, h)
        return result
    }

    fun getAcceleratorUsed(): String = acceleratorName

    fun release() {
        session?.close()
        env?.close()
        session = null
        env = null
    }
}

class ImageClassifier(context: Context) {

    private val factory = DelegateFactory(context)
    private var interpreterResult: DelegateFactory.InterpreterResult? = null

    fun initialize(modelPath: String = "efficientnet_lite0.tflite") {
        interpreterResult = factory.createInterpreter(
            modelPath = modelPath,
            modelType = ModelType.IMAGE_CLASSIFICATION
        )
    }

    fun classify(bitmap: Bitmap, topK: Int = 5): List<ClassificationResult> {
        val interpreter = interpreterResult?.interpreter
            ?: throw IllegalStateException("Classificador não inicializado!")

        val inputBuffer = preprocessBitmap(bitmap, 224)
        val output = Array(1) { FloatArray(1000) }  // ImageNet classes

        interpreter.run(inputBuffer, output)

        return output[0]
            .mapIndexed { index, conf -> ClassificationResult("class_$index", conf) }
            .sortedByDescending { it.confidence }
            .take(topK)
    }

    fun getAcceleratorUsed(): String = interpreterResult?.delegateName ?: "N/A"

    fun release() {
        interpreterResult?.interpreter?.close()
        factory.releaseAll()
    }

    private fun preprocessBitmap(bitmap: Bitmap, size: Int): ByteBuffer {
        val resized = Bitmap.createScaledBitmap(bitmap, size, size, true)
        val buffer = ByteBuffer.allocateDirect(1 * size * size * 3)
            .order(ByteOrder.nativeOrder())
        val pixels = IntArray(size * size)
        resized.getPixels(pixels, 0, size, 0, 0, size, size)
        for (pixel in pixels) {
            buffer.put(((pixel shr 16) and 0xFF).toByte())
            buffer.put(((pixel shr 8) and 0xFF).toByte())
            buffer.put((pixel and 0xFF).toByte())
        }
        buffer.rewind()
        return buffer
    }
}
