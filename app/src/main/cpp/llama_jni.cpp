#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include "llama.h"
#include "ggml.h"

#define TAG "LlamaJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static llama_model*   g_model   = nullptr;
static llama_context* g_ctx     = nullptr;
static llama_sampler* g_sampler = nullptr;

static void log_callback(ggml_log_level level, const char* text, void*) {
    if (level == GGML_LOG_LEVEL_ERROR) LOGE("%s", text);
    else                               LOGI("%s", text);
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_aidetector_llm_LlamaRunner_nativeLoad(
        JNIEnv* env, jobject,
        jstring jpath, jint n_ctx, jint n_threads) {

    llama_log_set(log_callback, nullptr);
    llama_backend_init();

    const char* path = env->GetStringUTFChars(jpath, nullptr);
    LOGI("Loading model: %s  ctx=%d threads=%d", path, n_ctx, n_threads);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  // CPU only — driver Vulkan Adreno incompatível nesta versão do llama.cpp

    g_model = llama_model_load_from_file(path, mparams);
    env->ReleaseStringUTFChars(jpath, path);

    if (!g_model) { LOGE("Failed to load model"); return JNI_FALSE; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = n_ctx;
    cparams.n_threads = n_threads;
    cparams.n_batch   = 512;

    g_ctx = llama_init_from_model(g_model, cparams);
    if (!g_ctx) { LOGE("Failed to create context"); return JNI_FALSE; }

    g_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_temp(0.7f));
    llama_sampler_chain_add(g_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    const llama_vocab* vocab = llama_model_get_vocab(g_model);
    LOGI("Model loaded OK. Vocab size: %d", llama_vocab_n_tokens(vocab));
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL
Java_com_aidetector_llm_LlamaRunner_nativeGenerate(
        JNIEnv* env, jobject,
        jstring jprompt, jint max_tokens) {

    if (!g_model || !g_ctx) return env->NewStringUTF("[ERROR: model not loaded]");

    const llama_vocab* vocab = llama_model_get_vocab(g_model);
    const char* prompt = env->GetStringUTFChars(jprompt, nullptr);

    const int n = -llama_tokenize(vocab, prompt, strlen(prompt), nullptr, 0, true, true);
    std::vector<llama_token> tokens(n);
    llama_tokenize(vocab, prompt, strlen(prompt), tokens.data(), tokens.size(), true, true);
    env->ReleaseStringUTFChars(jprompt, prompt);

    const int n_ctx = llama_n_ctx(g_ctx);
    const int max_prompt = n_ctx - max_tokens - 8;
    if (max_prompt <= 0) {
        return env->NewStringUTF("[ERROR: context too small for this max_tokens]");
    }
    if ((int)tokens.size() > max_prompt) {
        tokens.erase(tokens.begin(), tokens.begin() + (tokens.size() - max_prompt));
        LOGI("Prompt truncated to %d tokens (ctx=%d)", (int)tokens.size(), n_ctx);
    }

    llama_memory_clear(llama_get_memory(g_ctx), false);
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(g_ctx, batch)) {
        LOGE("llama_decode failed on prompt");
        return env->NewStringUTF("[ERROR: decode failed]");
    }

    std::string result;
    result.reserve(max_tokens * 4);
    int n_cur = (int)tokens.size();

    for (int i = 0; i < max_tokens; i++) {
        if (n_cur >= n_ctx - 1) break;
        llama_token token = llama_sampler_sample(g_sampler, g_ctx, -1);
        if (llama_vocab_is_eog(vocab, token)) break;

        char buf[256] = {};
        int nb = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        if (nb > 0) result.append(buf, nb);

        n_cur++;
        llama_batch next = llama_batch_get_one(&token, 1);
        if (llama_decode(g_ctx, next)) break;
    }

    return env->NewStringUTF(result.c_str());
}

JNIEXPORT void JNICALL
Java_com_aidetector_llm_LlamaRunner_nativeFree(JNIEnv*, jobject) {
    if (g_sampler) { llama_sampler_free(g_sampler); g_sampler = nullptr; }
    if (g_ctx)     { llama_free(g_ctx);              g_ctx     = nullptr; }
    if (g_model)   { llama_model_free(g_model);      g_model   = nullptr; }
    llama_backend_free();
    LOGI("Model freed");
}

JNIEXPORT jstring JNICALL
Java_com_aidetector_llm_LlamaRunner_nativeModelInfo(JNIEnv* env, jobject) {
    if (!g_model) return env->NewStringUTF("{}");
    const llama_vocab* vocab = llama_model_get_vocab(g_model);
    char buf[256];
    snprintf(buf, sizeof(buf),
             "{\"n_params\":%lld,\"n_vocab\":%d,\"n_ctx_train\":%d}",
             (long long)llama_model_n_params(g_model),
             llama_vocab_n_tokens(vocab),
             llama_model_n_ctx_train(g_model));
    return env->NewStringUTF(buf);
}

} // extern "C"
