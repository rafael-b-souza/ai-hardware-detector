package com.aidetector.server

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import android.util.Log
import com.aidetector.benchmark.BenchmarkRunner
import com.aidetector.delegates.DelegateFactory
import com.aidetector.detection.AcceleratorType
import com.aidetector.detection.HardwareDetector
import com.aidetector.detection.ModelType
import com.aidetector.models.BackgroundRemover
import com.aidetector.models.OrtBackgroundRemover
import com.aidetector.models.ImageClassifier
import com.aidetector.models.LocalLLMRunner
import com.aidetector.llm.LlamaRunner
import com.aidetector.models.VideoObjectDetector
import fi.iki.elonen.NanoHTTPD
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream

/** Servidor HTTP embarcado que expõe endpoints de inferência e diagnóstico de hardware. */
class InferenceServer(
    private val context: Context,
    port: Int = 8080
) : NanoHTTPD(port) {

    companion object {
        private const val TAG = "InferenceServer"
        private const val MAX_UPLOAD_SIZE = 50 * 1024 * 1024
        private const val API_VERSION = "1.0.0"
    }

    private val detector = HardwareDetector(context)
    private val delegateFactory = DelegateFactory(context)
    private val benchmarkRunner = BenchmarkRunner(context)

    private var objectDetector: VideoObjectDetector? = null
    private var imageClassifier: ImageClassifier? = null
    private var llmRunner: LocalLLMRunner? = null
    private var llamaRunner: LlamaRunner? = null
    private var llamaLoadedModel: String? = null
    private var backgroundRemover: BackgroundRemover? = null
    private var backgroundRemoverAccelerator: AcceleratorType? = null
    private var ortBackgroundRemover: OrtBackgroundRemover? = null
    private var ortBackgroundRemoverModel: String? = null

    private val loadedModels = mutableMapOf<String, LoadedModelInfo>()

    private var totalRequests = 0L
    private var totalInferenceTimeMs = 0L
    private val requestQueue = java.util.concurrent.LinkedBlockingQueue<Long>()
    private var serverStartTime = 0L

    private var apiKey: String? = null

    data class LoadedModelInfo(
        val name: String,
        val path: String,
        val type: ModelType,
        val accelerator: AcceleratorType,
        val loadedAt: Long = System.currentTimeMillis()
    )

    fun startServer(
        apiKey: String? = null,
        daemon: Boolean = true
    ) {
        this.apiKey = apiKey
        this.serverStartTime = System.currentTimeMillis()
        start(SOCKET_READ_TIMEOUT, daemon)
        Log.i(TAG, "Servidor de inferência iniciado na porta $listeningPort")
        Log.i(TAG, "API Key: ${if (apiKey != null) "Habilitada" else "Desabilitada"}")
    }

    override fun serve(session: IHTTPSession): Response {
        val uri = session.uri.trimEnd('/')
        val method = session.method

        val corsHeaders = mutableMapOf(
            "Access-Control-Allow-Origin" to "*",
            "Access-Control-Allow-Methods" to "GET, POST, DELETE, OPTIONS",
            "Access-Control-Allow-Headers" to "Content-Type, Authorization, X-API-Key",
            "Access-Control-Max-Age" to "86400"
        )

        if (method == Method.OPTIONS) {
            return newFixedLengthResponse(Response.Status.OK, "text/plain", "")
                .applyCors(corsHeaders)
        }

        if (apiKey != null && !checkAuth(session)) {
            return jsonError(Response.Status.UNAUTHORIZED, "API key inválida ou ausente")
                .applyCors(corsHeaders)
        }

        totalRequests++

        val response = try {
            when {
                method == Method.GET && uri == "/api/v1/status" -> handleStatus()
                method == Method.GET && uri == "/api/v1/hardware" -> handleHardware()
                method == Method.GET && uri == "/api/v1/models" -> handleListModels()
                method == Method.GET && uri == "/api/v1/queue" -> handleQueue()

                method == Method.POST && uri == "/api/v1/detect" -> handleDetect(session)
                method == Method.POST && uri == "/api/v1/classify" -> handleClassify(session)
                method == Method.POST && uri == "/api/v1/llm/generate" -> handleLLMGenerate(session)
                method == Method.POST && uri == "/api/v1/llm/chat" -> handleLLMChat(session)
                method == Method.POST && uri == "/api/v1/inference/raw" -> handleRawInference(session)

                method == Method.POST && uri == "/api/v1/models/load" -> handleLoadModel(session)
                method == Method.DELETE && uri.startsWith("/api/v1/models/") -> {
                    val modelName = uri.removePrefix("/api/v1/models/")
                    handleUnloadModel(modelName)
                }

                method == Method.POST && uri == "/api/v1/benchmark" -> handleBenchmark(session)

                method == Method.POST && uri == "/api/v1/rembg" -> handleRembg(session)
                method == Method.POST && uri == "/api/v1/models/upload" -> handleUploadModel(session)

                method == Method.GET && (uri == "/" || uri == "/api" || uri == "/api/v1") -> handleDocs()

                else -> jsonError(Response.Status.NOT_FOUND, "Endpoint não encontrado: $method $uri")
            }
        } catch (e: OutOfMemoryError) {
            Log.e(TAG, "OOM processando $uri", e)
            System.gc()
            jsonError(Response.Status.INTERNAL_ERROR, "OutOfMemory: modelo muito grande para inferência neste dispositivo")
        } catch (e: Throwable) {
            Log.e(TAG, "Erro processando $uri: ${e.message}", e)
            jsonError(Response.Status.INTERNAL_ERROR, "Erro interno: ${e.message}")
        }

        return response.applyCors(corsHeaders)
    }

    private fun handleStatus(): Response {
        val accelerators = detector.detectAllAccelerators()
        val uptimeMs = System.currentTimeMillis() - serverStartTime
        val avgLatency = if (totalRequests > 0) totalInferenceTimeMs / totalRequests else 0

        val json = JSONObject().apply {
            put("status", "running")
            put("version", API_VERSION)
            put("device", android.os.Build.MODEL)
            put("soc", android.os.Build.SOC_MODEL)
            put("android_version", android.os.Build.VERSION.SDK_INT)
            put("uptime_seconds", uptimeMs / 1000)
            put("total_requests", totalRequests)
            put("avg_latency_ms", avgLatency)
            put("loaded_models", loadedModels.size)
            put("accelerators_available", accelerators.size)
            put("best_accelerator", accelerators.firstOrNull()?.type?.name ?: "CPU")
            put("port", listeningPort)
            put("auth_enabled", apiKey != null)
        }

        return jsonResponse(json)
    }

    private fun handleHardware(): Response {
        val accelerators = detector.detectAllAccelerators()

        val json = JSONObject().apply {
            put("device", JSONObject().apply {
                put("model", android.os.Build.MODEL)
                put("manufacturer", android.os.Build.MANUFACTURER)
                put("soc", android.os.Build.SOC_MODEL)
                put("board", android.os.Build.BOARD)
                put("hardware", android.os.Build.HARDWARE)
                put("abi", JSONArray(android.os.Build.SUPPORTED_ABIS.toList()))
                val activityManager = context.getSystemService(android.app.ActivityManager::class.java)
                val memInfo = android.app.ActivityManager.MemoryInfo()
                activityManager.getMemoryInfo(memInfo)
                put("ram_mb", memInfo.totalMem / (1024 * 1024))
                put("available_ram_mb", memInfo.availMem / (1024 * 1024))
                put("cpu_cores", Runtime.getRuntime().availableProcessors())
            })
            put("accelerators", JSONArray().apply {
                accelerators.forEach { accel ->
                    put(JSONObject().apply {
                        put("type", accel.type.name)
                        put("name", accel.name)
                        put("available", accel.available)
                        put("estimated_tops", accel.estimatedTOPS)
                        put("priority", accel.priority)
                        put("supported_ops", JSONArray(accel.supportedOps))
                        put("chip_info", accel.chipInfo)
                    })
                }
            })
            put("recommendations", JSONObject().apply {
                ModelType.values().forEach { type ->
                    put(type.name.lowercase(), detector.getBestAccelerator(type).name)
                }
            })
        }

        return jsonResponse(json)
    }

    private fun handleListModels(): Response {
        val filesInDir = context.filesDir.listFiles()
            ?.filter { it.extension.lowercase() in listOf("gguf", "tflite", "onnx") }
            ?.map { it.name } ?: emptyList()

        val assetsFiles = try {
            context.assets.list("")?.filter {
                it.endsWith(".tflite") || it.endsWith(".gguf") || it.endsWith(".onnx")
            } ?: emptyList()
        } catch (e: Exception) { emptyList<String>() }

        val allFiles = (filesInDir + assetsFiles).distinct()

        val json = JSONObject().apply {
            put("models", JSONArray().apply {
                allFiles.forEach { name ->
                    val loadedInfo = loadedModels[name]
                    put(JSONObject().apply {
                        put("name", name)
                        put("loaded", loadedInfo != null)
                        put("type", loadedInfo?.type?.name ?: guessModelType(name))
                        put("source", if (filesInDir.contains(name)) "filesDir" else "assets")
                        loadedInfo?.let {
                            put("path", it.path)
                            put("accelerator", it.accelerator.name)
                        }
                    })
                }
            })
            put("count", allFiles.size)
            put("available_types", JSONArray(ModelType.values().map { it.name }))
        }
        return jsonResponse(json)
    }

    private fun guessModelType(filename: String): String {
        val lower = filename.lowercase()
        return when {
            lower.endsWith(".gguf") -> ModelType.LLM_INFERENCE.name
            lower.contains("birefnet") || lower.contains("rmbg") || lower.contains("segment") || lower.contains("deeplab") -> ModelType.SEGMENTATION.name
            lower.contains("detect") || lower.contains("ssd") || lower.contains("yolo") -> ModelType.OBJECT_DETECTION.name
            lower.contains("pose") || lower.contains("movenet") -> ModelType.POSE_ESTIMATION.name
            lower.contains("ocr") -> ModelType.OCR.name
            lower.contains("audio") || lower.contains("yamnet") -> ModelType.AUDIO_CLASSIFICATION.name
            else -> ModelType.IMAGE_CLASSIFICATION.name
        }
    }

    private fun handleQueue(): Response {
        val json = JSONObject().apply {
            put("pending", requestQueue.size)
            put("total_processed", totalRequests)
            put("avg_inference_ms", if (totalRequests > 0) totalInferenceTimeMs / totalRequests else 0)
        }
        return jsonResponse(json)
    }

    private fun handleDetect(session: IHTTPSession): Response {
        val body = parseBody(session)
        val startTime = System.nanoTime()

        if (objectDetector == null) {
            val modelPath = body.optString("model", "ssd_mobilenet_v2.tflite")
            val forceAccel = body.optString("accelerator", "").let {
                if (it.isNotEmpty()) AcceleratorType.valueOf(it.uppercase()) else null
            }
            objectDetector = VideoObjectDetector(context).also {
                it.initialize(modelPath, forceAccelerator = forceAccel)
            }
        }

        val bitmap = decodeImageFromBody(session, body)
            ?: return jsonError(Response.Status.BAD_REQUEST, "Imagem não fornecida. Use 'image' (base64) ou upload multipart.")

        val minConfidence = body.optDouble("confidence", 0.5).toFloat()

        val detections = objectDetector!!.detectFrame(bitmap)
            .filter { it.confidence >= minConfidence }

        val elapsedMs = (System.nanoTime() - startTime) / 1_000_000f
        totalInferenceTimeMs += elapsedMs.toLong()

        val json = JSONObject().apply {
            put("detections", JSONArray().apply {
                detections.forEach { det ->
                    put(JSONObject().apply {
                        put("label", det.label)
                        put("confidence", det.confidence)
                        put("bbox", JSONObject().apply {
                            put("left", det.boundingBox.left)
                            put("top", det.boundingBox.top)
                            put("right", det.boundingBox.right)
                            put("bottom", det.boundingBox.bottom)
                        })
                    })
                }
            })
            put("count", detections.size)
            put("inference_ms", elapsedMs)
            put("accelerator", objectDetector!!.getAcceleratorUsed())
            put("image_size", "${bitmap.width}x${bitmap.height}")
        }

        return jsonResponse(json)
    }

    private fun handleClassify(session: IHTTPSession): Response {
        val body = parseBody(session)
        val startTime = System.nanoTime()

        if (imageClassifier == null) {
            val modelPath = body.optString("model", "efficientnet_lite0.tflite")
            imageClassifier = ImageClassifier(context).also { it.initialize(modelPath) }
        }

        val bitmap = decodeImageFromBody(session, body)
            ?: return jsonError(Response.Status.BAD_REQUEST, "Imagem não fornecida")

        val topK = body.optInt("top_k", 5)
        val results = imageClassifier!!.classify(bitmap, topK)

        val elapsedMs = (System.nanoTime() - startTime) / 1_000_000f
        totalInferenceTimeMs += elapsedMs.toLong()

        val json = JSONObject().apply {
            put("classifications", JSONArray().apply {
                results.forEach { r ->
                    put(JSONObject().apply {
                        put("label", r.label)
                        put("confidence", r.confidence)
                    })
                }
            })
            put("inference_ms", elapsedMs)
            put("accelerator", imageClassifier!!.getAcceleratorUsed())
        }

        return jsonResponse(json)
    }

    private fun handleLLMGenerate(session: IHTTPSession): Response {
        val body = parseBody(session)
        val startTime = System.nanoTime()
        val modelPath = body.optString("model", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
        val prompt = body.optString("prompt", "")
        if (prompt.isEmpty()) return jsonError(Response.Status.BAD_REQUEST, "Campo 'prompt' é obrigatório")
        val maxTokens = body.optInt("max_tokens", 256)

        val text: String
        val tokensGenerated: Int
        val tokensPerSec: Float

        if (modelPath.endsWith(".gguf", ignoreCase = true)) {
            if (llamaRunner == null || llamaLoadedModel != modelPath) {
                llamaRunner?.release()
                llamaRunner = LlamaRunner(context).also { it.load(modelPath) }
                llamaLoadedModel = modelPath
            }
            val result = llamaRunner!!.generate(prompt, maxTokens)
            text = result.text; tokensGenerated = result.tokensGenerated; tokensPerSec = result.tokensPerSecond
        } else {
            if (llmRunner == null) llmRunner = LocalLLMRunner(context).also { it.initialize(modelPath) }
            val response = llmRunner!!.generate(prompt, maxTokens, 0.7f)
            text = response.text; tokensGenerated = response.tokensGenerated; tokensPerSec = response.tokensPerSecond
        }

        val elapsedMs = (System.nanoTime() - startTime) / 1_000_000f
        totalInferenceTimeMs += elapsedMs.toLong()

        val json = JSONObject().apply {
            put("text", text)
            put("tokens_generated", tokensGenerated)
            put("tokens_per_second", tokensPerSec)
            put("total_ms", elapsedMs)
            put("model", modelPath)
        }
        return jsonResponse(json)
    }

    private fun handleLLMChat(session: IHTTPSession): Response {
        val body = parseBody(session)

        val messages = body.optJSONArray("messages")
            ?: return jsonError(Response.Status.BAD_REQUEST, "Campo 'messages' é obrigatório")

        val chatPrompt = buildString {
            for (i in 0 until messages.length()) {
                val msg = messages.getJSONObject(i)
                val role = msg.getString("role")
                val content = msg.getString("content")
                append("<|$role|>\n$content\n")
            }
            append("<|assistant|>\n")
        }

        val maxTokens = body.optInt("max_tokens", 256)
        val modelPath = body.optString("model", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

        val startTime = System.nanoTime()
        val replyText: String

        if (modelPath.endsWith(".gguf", ignoreCase = true)) {
            if (llamaRunner == null || llamaLoadedModel != modelPath) {
                llamaRunner?.release()
                llamaRunner = LlamaRunner(context).also { it.load(modelPath) }
                llamaLoadedModel = modelPath
            }
            val msgList = mutableListOf<Map<String, String>>()
            for (i in 0 until messages.length()) {
                val m = messages.getJSONObject(i)
                msgList.add(mapOf("role" to m.getString("role"), "content" to m.getString("content")))
            }
            replyText = llamaRunner!!.chat(msgList, maxTokens).text
        } else {
            if (llmRunner == null) llmRunner = LocalLLMRunner(context).also { it.initialize(modelPath) }
            replyText = llmRunner!!.generate(chatPrompt, maxTokens, 0.7f).text
        }

        val elapsedMs = (System.nanoTime() - startTime) / 1_000_000f

        val json = JSONObject().apply {
            put("message", JSONObject().apply {
                put("role", "assistant")
                put("content", replyText)
            })
            put("total_ms", elapsedMs)
            put("model", modelPath)
        }

        return jsonResponse(json)
    }

    private fun handleRawInference(session: IHTTPSession): Response {
        val body = parseBody(session)
        val startTime = System.nanoTime()

        val modelPath = body.optString("model", "")
        if (modelPath.isEmpty()) {
            return jsonError(Response.Status.BAD_REQUEST, "Campo 'model' é obrigatório")
        }

        val modelType = try {
            ModelType.valueOf(body.optString("model_type", "IMAGE_CLASSIFICATION").uppercase())
        } catch (e: Exception) {
            ModelType.IMAGE_CLASSIFICATION
        }

        val forceAccel = body.optString("accelerator", "").let {
            if (it.isNotEmpty()) try { AcceleratorType.valueOf(it.uppercase()) } catch (e: Exception) { null }
            else null
        }

        val result = delegateFactory.createInterpreter(
            modelPath = modelPath,
            modelType = modelType,
            forceAccelerator = forceAccel
        )

        val inputArray = body.optJSONArray("input")
        val inputShapeArray = body.optJSONArray("input_shape")

        if (inputArray == null) {
            return jsonError(Response.Status.BAD_REQUEST, "Campo 'input' é obrigatório")
        }

        val inputData = FloatArray(inputArray.length()) { inputArray.getDouble(it).toFloat() }
        val inputBuffer = java.nio.ByteBuffer.allocateDirect(inputData.size * 4)
            .order(java.nio.ByteOrder.nativeOrder())
        inputData.forEach { inputBuffer.putFloat(it) }
        inputBuffer.rewind()

        val interpreter = result.interpreter
        val outputTensor = interpreter.getOutputTensor(0)
        val outputShape = outputTensor.shape()
        val outputSize = outputShape.fold(1) { acc, i -> acc * i }
        val outputBuffer = java.nio.ByteBuffer.allocateDirect(outputSize * 4)
            .order(java.nio.ByteOrder.nativeOrder())

        interpreter.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()

        val outputData = FloatArray(outputSize)
        outputBuffer.asFloatBuffer().get(outputData)

        val elapsedMs = (System.nanoTime() - startTime) / 1_000_000f
        totalInferenceTimeMs += elapsedMs.toLong()

        interpreter.close()
        delegateFactory.releaseAll()

        val json = JSONObject().apply {
            put("output", JSONArray(outputData.toList()))
            put("output_shape", JSONArray(outputShape.toList()))
            put("inference_ms", elapsedMs)
            put("accelerator", result.delegateName)
            put("model", modelPath)
        }

        return jsonResponse(json)
    }

    private fun handleLoadModel(session: IHTTPSession): Response {
        val body = parseBody(session)

        val name = body.optString("name", "")
        val path = body.optString("path", "")
        val typeName = body.optString("type", "IMAGE_CLASSIFICATION")

        if (name.isEmpty() || path.isEmpty()) {
            return jsonError(Response.Status.BAD_REQUEST, "Campos 'name' e 'path' são obrigatórios")
        }

        val type = try {
            ModelType.valueOf(typeName.uppercase())
        } catch (e: Exception) {
            return jsonError(Response.Status.BAD_REQUEST, "Tipo inválido: $typeName. Use: ${ModelType.values().joinToString()}")
        }

        val accel = detector.getBestAccelerator(type)

        when (type) {
            ModelType.OBJECT_DETECTION, ModelType.VIDEO_DETECTION -> {
                objectDetector?.release()
                objectDetector = VideoObjectDetector(context).also { it.initialize(path) }
            }
            ModelType.IMAGE_CLASSIFICATION -> {
                imageClassifier?.release()
                imageClassifier = ImageClassifier(context).also { it.initialize(path) }
            }
            ModelType.LLM_INFERENCE -> {
                llmRunner?.release()
                llmRunner = LocalLLMRunner(context).also { it.initialize(path) }
            }
            else -> {}
        }

        loadedModels[name] = LoadedModelInfo(name, path, type, accel)

        val json = JSONObject().apply {
            put("loaded", true)
            put("name", name)
            put("path", path)
            put("type", type.name)
            put("accelerator", accel.name)
        }

        return jsonResponse(json)
    }

    private fun handleUnloadModel(modelName: String): Response {
        val info = loadedModels.remove(modelName)
        if (info != null) {
            when (info.type) {
                ModelType.OBJECT_DETECTION, ModelType.VIDEO_DETECTION -> {
                    objectDetector?.release(); objectDetector = null
                }
                ModelType.IMAGE_CLASSIFICATION -> {
                    imageClassifier?.release(); imageClassifier = null
                }
                ModelType.LLM_INFERENCE -> {
                    llmRunner?.release(); llmRunner = null
                    llamaRunner?.release(); llamaRunner = null
                }
                else -> {}
            }
        }

        val file = File(context.filesDir, modelName)
        val deleted = if (file.exists()) file.delete() else false

        if (info == null && !deleted) {
            return jsonError(Response.Status.NOT_FOUND, "Modelo '$modelName' não encontrado")
        }

        val json = JSONObject().apply {
            put("deleted", true)
            put("name", modelName)
            put("file_removed", deleted)
        }
        return jsonResponse(json)
    }

    private fun handleRembg(session: IHTTPSession): Response {
        val body = parseBody(session)
        val startTime = System.nanoTime()
        val modelPath = body.optString("model", "rmbg2.tflite")

        val bitmap = decodeImageFromBody(session, body)
            ?: return jsonError(Response.Status.BAD_REQUEST, "Image required: provide 'image' as base64")

        val result: Bitmap
        val acceleratorUsed: String

        if (modelPath.endsWith(".onnx", ignoreCase = true)) {
            if (ortBackgroundRemover == null || modelPath != ortBackgroundRemoverModel) {
                ortBackgroundRemover?.release()
                ortBackgroundRemoverModel = modelPath
                ortBackgroundRemover = OrtBackgroundRemover(context).also { it.initialize(modelPath) }
            }
            result = ortBackgroundRemover!!.removeBackground(bitmap)
            acceleratorUsed = ortBackgroundRemover!!.getAcceleratorUsed()
        } else {
            val forceAccel = body.optString("accelerator", "GPU").let {
                if (it.isBlank()) AcceleratorType.GPU
                else runCatching { AcceleratorType.valueOf(it.uppercase()) }.getOrDefault(AcceleratorType.GPU)
            }
            if (backgroundRemover == null || forceAccel != backgroundRemoverAccelerator) {
                backgroundRemover?.release()
                backgroundRemoverAccelerator = forceAccel
                backgroundRemover = BackgroundRemover(context).also { it.initialize(modelPath, forceAccel) }
            }
            result = backgroundRemover!!.removeBackground(bitmap)
            acceleratorUsed = backgroundRemover!!.getAcceleratorUsed()
        }

        val out = ByteArrayOutputStream()
        result.compress(Bitmap.CompressFormat.PNG, 100, out)
        val b64 = Base64.encodeToString(out.toByteArray(), Base64.NO_WRAP)

        val elapsedMs = (System.nanoTime() - startTime) / 1_000_000f

        val json = JSONObject().apply {
            put("result", b64)
            put("format", "png")
            put("width", result.width)
            put("height", result.height)
            put("inference_ms", elapsedMs)
            put("accelerator", acceleratorUsed)
        }
        return jsonResponse(json)
    }

    private fun handleUploadModel(session: IHTTPSession): Response {
        val files = mutableMapOf<String, String>()
        session.parseBody(files)

        val tmpPath = files["file"] ?: files["model"]
            ?: return jsonError(Response.Status.BAD_REQUEST, "Envie o arquivo como 'file' ou 'model'")

        val targetName = session.parms["name"] ?: "uploaded_model.onnx"
        val targetFile = File(context.filesDir, targetName)
        File(tmpPath).copyTo(targetFile, overwrite = true)

        val json = JSONObject().apply {
            put("uploaded", true)
            put("name", targetName)
            put("path", targetFile.absolutePath)
            put("size_mb", targetFile.length() / (1024.0 * 1024.0))
        }

        return jsonResponse(json)
    }

    private fun handleBenchmark(session: IHTTPSession): Response {
        val body = parseBody(session)

        val modelPath = body.optString("model", "model.tflite")
        val typeName = body.optString("type", "IMAGE_CLASSIFICATION")
        val shapeArray = body.optJSONArray("input_shape")

        val type = try { ModelType.valueOf(typeName.uppercase()) } catch (e: Exception) { ModelType.IMAGE_CLASSIFICATION }
        val shape = if (shapeArray != null) {
            IntArray(shapeArray.length()) { shapeArray.getInt(it) }
        } else intArrayOf(1, 224, 224, 3)

        val results = runBlocking {
            benchmarkRunner.benchmarkModel(modelPath, type, shape)
        }

        val json = JSONObject().apply {
            put("model", modelPath)
            put("type", type.name)
            put("results", JSONArray().apply {
                results.forEach { r ->
                    put(JSONObject().apply {
                        put("accelerator", r.accelerator.name)
                        put("delegate", r.delegateName)
                        put("success", r.success)
                        put("avg_ms", r.avgInferenceMs)
                        put("min_ms", r.minInferenceMs)
                        put("max_ms", r.maxInferenceMs)
                        put("fps", r.fps)
                        put("memory_mb", r.memoryUsageMb)
                        put("warmup_ms", r.warmupTimeMs)
                        if (!r.success) put("error", r.error)
                    })
                }
            })
            put("best", results.filter { it.success }.minByOrNull { it.avgInferenceMs }?.let {
                JSONObject().apply {
                    put("accelerator", it.accelerator.name)
                    put("fps", it.fps)
                    put("avg_ms", it.avgInferenceMs)
                }
            })
        }

        return jsonResponse(json)
    }

    private fun handleDocs(): Response {
        val html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>AI Inference Server</title>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { font-family: system-ui, -apple-system, sans-serif; background: #0a0a12; color: #e0e0e0; padding: 2rem; }
                h1 { color: #00e5ff; font-size: 1.8rem; margin-bottom: 0.5rem; }
                h2 { color: #76ff03; font-size: 1.2rem; margin: 1.5rem 0 0.5rem; }
                .subtitle { color: #888; font-size: 0.9rem; margin-bottom: 2rem; }
                .endpoint { background: #1a1a2e; border-radius: 8px; padding: 1rem; margin: 0.5rem 0; border-left: 3px solid #00e5ff; }
                .method { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; margin-right: 8px; }
                .get { background: #0f6e56; color: #fff; }
                .post { background: #534ab7; color: #fff; }
                .delete { background: #993c1d; color: #fff; }
                .path { font-family: monospace; color: #00e5ff; }
                .desc { color: #aaa; font-size: 0.85rem; margin-top: 4px; }
                code { background: #2a2a3e; padding: 2px 6px; border-radius: 4px; font-size: 0.85rem; }
                pre { background: #1a1a2e; padding: 1rem; border-radius: 8px; overflow-x: auto; margin: 0.5rem 0; font-size: 0.85rem; }
                .status { display: inline-block; padding: 4px 12px; border-radius: 12px; background: #0f6e56; color: #fff; font-size: 0.8rem; }
            </style>
        </head>
        <body>
            <h1>AI Inference Server</h1>
            <p class="subtitle">
                <span class="status">running</span>
                ${android.os.Build.MODEL} — ${android.os.Build.SOC_MODEL} — porta $listeningPort
            </p>
            
            <h2>Status & Hardware</h2>
            <div class="endpoint"><span class="method get">GET</span> <span class="path">/api/v1/status</span><div class="desc">Info do servidor, uptime, métricas</div></div>
            <div class="endpoint"><span class="method get">GET</span> <span class="path">/api/v1/hardware</span><div class="desc">Aceleradores detectados (NPU, GPU, DSP, CPU)</div></div>
            <div class="endpoint"><span class="method get">GET</span> <span class="path">/api/v1/models</span><div class="desc">Modelos carregados</div></div>
            <div class="endpoint"><span class="method get">GET</span> <span class="path">/api/v1/queue</span><div class="desc">Status da fila de processamento</div></div>

            <h2>Inferência</h2>
            <div class="endpoint"><span class="method post">POST</span> <span class="path">/api/v1/detect</span><div class="desc">Detecção de objetos em imagem (base64 ou multipart)</div></div>
            <div class="endpoint"><span class="method post">POST</span> <span class="path">/api/v1/classify</span><div class="desc">Classificação de imagem</div></div>
            <div class="endpoint"><span class="method post">POST</span> <span class="path">/api/v1/llm/generate</span><div class="desc">Geração de texto com LLM local</div></div>
            <div class="endpoint"><span class="method post">POST</span> <span class="path">/api/v1/llm/chat</span><div class="desc">Chat multi-turno com LLM local</div></div>
            <div class="endpoint"><span class="method post">POST</span> <span class="path">/api/v1/inference/raw</span><div class="desc">Inferência genérica com modelo custom</div></div>

            <h2>Gerenciamento</h2>
            <div class="endpoint"><span class="method post">POST</span> <span class="path">/api/v1/models/load</span><div class="desc">Carregar modelo</div></div>
            <div class="endpoint"><span class="method post">POST</span> <span class="path">/api/v1/models/upload</span><div class="desc">Upload de modelo .tflite</div></div>
            <div class="endpoint"><span class="method delete">DELETE</span> <span class="path">/api/v1/models/{name}</span><div class="desc">Descarregar modelo</div></div>
            <div class="endpoint"><span class="method post">POST</span> <span class="path">/api/v1/benchmark</span><div class="desc">Benchmark de aceleradores</div></div>

            <h2>Exemplo rápido</h2>
            <pre>
# Verificar hardware
curl http://PHONE_IP:$listeningPort/api/v1/hardware

# Detecção de objetos
curl -X POST http://PHONE_IP:$listeningPort/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{"image": "/9j/4AAQ...(base64)...", "confidence": 0.5}'

# Chat com LLM
curl -X POST http://PHONE_IP:$listeningPort/api/v1/llm/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Olá!"}]}'
            </pre>
        </body>
        </html>
        """.trimIndent()

        return newFixedLengthResponse(Response.Status.OK, "text/html", html)
    }

    private fun parseBody(session: IHTTPSession): JSONObject {
        val contentLength = session.headers["content-length"]?.toLongOrNull() ?: 0
        if (contentLength > MAX_UPLOAD_SIZE) {
            throw IllegalArgumentException("Payload excede limite de ${MAX_UPLOAD_SIZE / 1024 / 1024}MB")
        }

        val contentType = session.headers["content-type"] ?: ""

        return if (contentType.contains("application/json")) {
            val files = mutableMapOf<String, String>()
            session.parseBody(files)
            val postData = files["postData"] ?: "{}"
            JSONObject(postData)
        } else if (contentType.contains("multipart")) {
            val files = mutableMapOf<String, String>()
            session.parseBody(files)
            JSONObject().apply {
                session.parms.forEach { (key, value) -> put(key, value) }
                files.forEach { (key, path) -> put("_file_$key", path) }
            }
        } else {
            JSONObject()
        }
    }

    private fun decodeImageFromBody(session: IHTTPSession, body: JSONObject): Bitmap? {
        val base64Image = body.optString("image", "")
        if (base64Image.isNotEmpty()) {
            val imageBytes = Base64.decode(
                base64Image.replace("data:image/[^;]+;base64,".toRegex(), ""),
                Base64.DEFAULT
            )
            return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        }

        val filePath = body.optString("_file_image", "")
        if (filePath.isNotEmpty()) {
            return BitmapFactory.decodeFile(filePath)
        }

        return null
    }

    private fun checkAuth(session: IHTTPSession): Boolean {
        val headerKey = session.headers["x-api-key"]
        val authHeader = session.headers["authorization"]
        val queryKey = session.parms["api_key"]

        return headerKey == apiKey ||
               authHeader == "Bearer $apiKey" ||
               queryKey == apiKey
    }

    private fun jsonResponse(json: JSONObject, status: Response.Status = Response.Status.OK): Response {
        return newFixedLengthResponse(status, "application/json", json.toString(2))
    }

    private fun jsonError(status: Response.Status, message: String): Response {
        val json = JSONObject().apply {
            put("error", true)
            put("message", message)
            put("status", status.requestStatus)
        }
        return newFixedLengthResponse(status, "application/json", json.toString(2))
    }

    private fun Response.applyCors(headers: Map<String, String>): Response {
        headers.forEach { (key, value) -> addHeader(key, value) }
        return this
    }

    fun stopServer() {
        stop()
        objectDetector?.release()
        imageClassifier?.release()
        llmRunner?.release()
        llamaRunner?.release()
        backgroundRemover?.release()
        delegateFactory.releaseAll()
        loadedModels.clear()
        Log.i(TAG, "Servidor parado")
    }
}
