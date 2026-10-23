package com.aidetector.llm

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.io.File

/** Executa inferência de modelos LLM em formato GGUF via llama.cpp. */
class LlamaRunner(private val context: Context) {

    companion object {
        private const val TAG = "LlamaRunner"
        private const val DEFAULT_CTX    = 2048
        private const val DEFAULT_THREADS = 4

        init {
            System.loadLibrary("llama_jni")
        }
    }

    private var loadedModel: String? = null

    private external fun nativeLoad(path: String, nCtx: Int, nThreads: Int): Boolean
    private external fun nativeGenerate(prompt: String, maxTokens: Int): String
    private external fun nativeFree()
    private external fun nativeModelInfo(): String

    fun load(modelName: String, nCtx: Int = DEFAULT_CTX, nThreads: Int = DEFAULT_THREADS) {
        if (loadedModel == modelName) return

        val file = File(context.filesDir, modelName)
        if (!file.exists()) throw IllegalStateException("Model not found: ${file.absolutePath}")

        Log.i(TAG, "Loading $modelName (${file.length() / 1024 / 1024} MB)...")
        val ok = nativeLoad(file.absolutePath, nCtx, nThreads)
        if (!ok) throw RuntimeException("llama.cpp failed to load $modelName")
        loadedModel = modelName
        Log.i(TAG, "Loaded OK — ${modelInfo()}")
    }

    fun generate(prompt: String, maxTokens: Int = 256): GenerateResult {
        val start = System.nanoTime()
        val text = nativeGenerate(prompt, maxTokens)
        val ms = (System.nanoTime() - start) / 1_000_000f
        val tokensGenerated = text.trim().split(Regex("\\s+")).size
        return GenerateResult(
            text            = text,
            tokensGenerated = tokensGenerated,
            tokensPerSecond = if (ms > 0) tokensGenerated / (ms / 1000f) else 0f,
            totalMs         = ms,
            model           = loadedModel ?: "unknown"
        )
    }

    fun chat(messages: List<Map<String, String>>, maxTokens: Int = 256): GenerateResult {
        val trimmed = if (messages.size > 8) messages.takeLast(8) else messages
        val prompt = buildChatMLPrompt(trimmed)
        return generate(prompt, maxTokens)
    }

    fun modelInfo(): Map<String, Any> {
        val json = JSONObject(nativeModelInfo())
        return mapOf(
            "n_params"     to json.optLong("n_params"),
            "n_vocab"      to json.optInt("n_vocab"),
            "n_ctx_train"  to json.optInt("n_ctx_train"),
            "model"        to (loadedModel ?: "none")
        )
    }

    fun release() {
        nativeFree()
        loadedModel = null
    }

    private fun buildChatMLPrompt(messages: List<Map<String, String>>): String {
        val sb = StringBuilder()
        for (msg in messages) {
            val role    = msg["role"]    ?: "user"
            val content = msg["content"] ?: ""
            sb.append("<|im_start|>$role\n$content<|im_end|>\n")
        }
        sb.append("<|im_start|>assistant\n")
        return sb.toString()
    }
}

data class GenerateResult(
    val text:            String,
    val tokensGenerated: Int,
    val tokensPerSecond: Float,
    val totalMs:         Float,
    val model:           String
)
