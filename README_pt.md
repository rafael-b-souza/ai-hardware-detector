# AI Hardware Detector & On-Device LLM Runtime (Português)

Framework em Kotlin e C++ (NDK) para detecção de aceleradores de hardware em dispositivos Android (Snapdragon NPU, Adreno GPU, DSP Hexagon e CPU ARM) com runtime integrado para inferência local de modelos GGUF via `llama.cpp`, TensorFlow Lite e ONNX Runtime.

Desenvolvido e validado em silício real (Samsung Galaxy S22 com Snapdragon 8 Gen 1, Galaxy S21 com Snapdragon 888 e Galaxy Note 8 com Snapdragon 835).

---

## Benchmark & Matriz Comparativa de Inferência (llama.cpp)

Testes de inferência executados localmente no **Samsung Galaxy S22 (Qualcomm Snapdragon 8 Gen 1, 8 GB RAM, Android 14 / One UI 6.1)** utilizando o modelo de raciocínio **Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled (Q4_K_M, ~1.2 GB)** via NDK com `llama.cpp` (build r27c / C++17).

| Acelerador | Backend / Driver | Formato / Quant | Velocidade (tok/s) | TTFT (Prompt Eval) | Consumo de RAM | Estabilidade & Termal | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CPU ARM** (Kryo: Cortex-X2 + A710) | `ggml-cpu` (NEON + `dotprod` + `i8mm`) | Q4_K_M / INT8 | **7.70 tok/s** | ~115 ms (32 tokens) | ~1.4 GB (mmap nativo) | Estável. 4 threads alocadas nos cores Gold/Prime evitam thermal runaway. | **Produção (Recomendado)** |
| **GPU Adreno 730** | `ggml-vulkan` (Qualcomm Vulkan ICD) | FP16 / Q4_K_M | *Falha no load* | N/A | Alocação rejeitada | **Crash (SIGSEGV)**. Driver proprietário Adreno falha ao alocar heap local de tensores. | Incompatível (Vendor bug) |
| **GPU Adreno 730** | OpenCL / TFLite GPU Delegate | FP16 / FP32 | ~18-24 tok/s (est.) | ~85 ms | ~1.6 GB | Estável para visão (MobileNet, BiRefNet); requer wrapper OpenCL customizado no llama. | Suporte Parcial (Vision/TFLite) |
| **NPU Hexagon 780** | Qualcomm QNN / HTP v73 (DSP) | INT4 / INT8 | ~28-35 tok/s (teórico) | <40 ms | <1.0 GB (TCM dedicado) | Eficiência máxima (mW/token); requer quantização offline via Qualcomm AI Hub. | Roadmap (HTP v73 direct) |

### Diagnóstico Técnico: Por que a CPU ARMv8.4-a superou a GPU Vulkan no Android

1. **Intrinsics ARMv8.4-a / ARMv9 (`+dotprod+i8mm`)**:
   Ao compilar o backend `ggml-cpu` com `-march=armv8.4-a+dotprod+i8mm -O3`, o compilador Clang emite instruções vetoriais `SDOT`/`UDOT` e `SMMLA` (MatMul INT8). Isso executa 4 operações inteiras por ciclo em cada pista SIMD de 128 bits dos núcleos Cortex-X2 e Cortex-A710, sustentando **7.70 tok/s** reais em um modelo de 2 bilhões de parâmetros com raciocínio ativo (bloco `<think>` de 300+ tokens gerado sem engasgos).
2. **Qualcomm Adreno Vulkan Driver SIGSEGV (`ggml_backend_buft_alloc_buffer`)**:
   Embora o backend Vulkan do `llama.cpp` compile com sucesso contra os headers Vulkan 1.3 (`vulkan.hpp`), o driver Vulkan proprietário da Qualcomm no Android (One UI com SELinux restritivo) retorna ponteiro nulo em alocações de `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` para buffers grandes de tensores GGUF mmapeados. A tentativa de dereferência gera `SIGSEGV (SEGV_MAPERR) em 0x0`. Até que o driver da Samsung/Qualcomm corrija essa alocação ou se utilize um driver alternativo (Mesa Turnip via root), o caminho de CPU NEON multithread é a única rota 100% determinística e sem crashes.
3. **Gerenciamento de Memória Sem Heap Java (OOM Prevention)**:
   Modelos GGUF e ONNX são mapeados diretamente via syscalls `mmap()` pelo runtime nativo C++ (`filesDir`). O heap da JVM permanece abaixo de 60 MB, impedindo que o Android LMK (Low Memory Killer) aborte o processo.

---

## Arquitetura de Detecção e Inferência NDK/JNI

```
+-----------------------------------------------------------------------------------------+
|                                    APPLICATION LAYER                                    |
|   +---------------------------------------------+  +--------------------------------+   |
|   |         Jetpack Compose App UI              |  |   NanoHTTPD REST API (:8080)   |   |
|   |  - Monitoramento de aceleradores e TOPS     |  |   - POST /api/v1/llm/generate  |   |
|   |  - Controle de threads, ctx e inferência    |  |   - POST /api/v1/models/upload |   |
|   |  - Visualizador de segmentação e chat local |  |   - GET  /api/v1/models        |   |
|   +---------------------------------------------+  +--------------------------------+   |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                          HARDWARE DETECTION & ROUTING ENGINE                            |
|   +--------------------------+  +------------------------+  +-----------------------+   |
|   | HardwareDetector.kt      |  | DelegateFactory.kt     |  | BenchmarkRunner.kt    |   |
|   | - Lê SoC / Board / CPU   |  | - Fallback Chain:      |  | - Medição de latência |   |
|   | - Varre /vendor/lib64    |  |   NPU -> GPU -> CPU    |  | - Taxa de tokens/s    |   |
|   | - Avalia TOPS e compat.  |  | - TFLite / ORT routing |  | - FPS em vídeo real   |   |
|   +--------------------------+  +------------------------+  +-----------------------+   |
+-----------------------------------------------------------------------------------------+
             |                                                |
             v (GGUF LLM)                                     v (TFLite & ONNX)
+------------------------------------------+  +-------------------------------------------+
|             JNI BRIDGE LAYER             |  |           VISION / EMBEDDINGS             |
|   com.aidetector.llm.LlamaRunner.kt      |  |  +-------------------------------------+  |
|                  | (JNI Invocation)      |  |  | TensorFlow Lite (2.14)              |  |
|                  v                       |  |  | - NNAPI Delegate (Hexagon DSP)      |  |
|   llama_jni.cpp (C++17 Shared Lib)       |  |  | - GPU Delegate (Adreno OpenCL)      |  |
|   - nativeLoad(path, n_ctx, threads)     |  |  | - XNNPACK CPU Delegate              |  |
|   - nativeGenerate(prompt, max_tokens)   |  |  +-------------------------------------+  |
|   - nativeModelInfo()                    |  |  +-------------------------------------+  |
|   - nativeFree()                         |  |  | ONNX Runtime Android (1.17)         |  |
|                  |                       |  |  | - NNAPI Execution Provider (EP)     |  |
|                  v                       |  |  | - CPU EP com mmap direto            |  |
|   llama.cpp Core Engine                  |  |  +-------------------------------------+  |
|   - llama_model_load_from_file()         |  +-------------------------------------------+
|   - llama_sampler_chain (top_k/p/temp)   |                        |
|   - ggml-cpu (NEON, dotprod, i8mm)       |                        |
+------------------------------------------+                        |
                     \                                             /
                      \                                           /
                       v                                         v
+-----------------------------------------------------------------------------------------+
|                                LINUX KERNEL & HARDWARE HAL                              |
|   +------------------------------+  +--------------------+  +-----------------------+   |
|   | CPU Cores (Snapdragon)       |  | GPU (Adreno 730)   |  | NPU / DSP             |   |
|   | - 1x Cortex-X2 @ 3.0 GHz     |  | - OpenCL 2.0 / 3.0 |  | - Hexagon 780         |   |
|   | - 3x Cortex-A710 @ 2.5 GHz   |  | - Vulkan 1.3       |  | - HTP v73 (27 TOPS)   |   |
|   | - 4x Cortex-A510 @ 1.8 GHz   |  |                    |  | - libQnnHtp.so        |   |
|   +------------------------------+  +--------------------+  +-----------------------+   |
+-----------------------------------------------------------------------------------------+
```

---

## Dispositivos Catalogados & Suporte de Hardware

| Dispositivo | SoC / Chipset | NPU Dedicada | GPU | Acelerador Ideal | LLM Recomendado |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Samsung Galaxy S22** | Snapdragon 8 Gen 1 | Hexagon 780 (27 TOPS) | Adreno 730 | NPU (Visão) / CPU NEON (LLM) | Qwen3.5-2B Q4_K_M (7.7 tok/s) |
| **Samsung Galaxy S21** | Snapdragon 888 | Hexagon 780 (26 TOPS) | Adreno 660 | NPU (Visão) / CPU NEON (LLM) | Gemma 2B INT8 / Phi-2 INT4 |
| **Samsung Galaxy Note 8** | Snapdragon 835 | Não possui NPU | Adreno 540 | GPU OpenCL / CPU | SmolLM-135M / DistilGPT2 |
| **Variantes Exynos (S21/S22)** | Exynos 2100 / 2200 | Samsung NPU (22–26 TOPS) | Mali-G78 / Xclipse 920 | NPU (via NNAPI) | Gemma 2B INT8 |

---

## Guia Prático: Android Devs & Comunidade r/LocalLLaMA

### 1. Pré-requisitos de Toolchain

- **Android SDK Platform 34** (minSdk 28 — Android 9.0+ para suporte NNAPI/Vulkan).
- **Android NDK r27c** (`27.2.12479018` ou superior).
- **CMake 3.22.1+** e **Ninja**.
- Dispositivo com depuração USB ou depuração sem fio habilitada.

### 2. Compilação da Biblioteca Nativa e do APK

```bash
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### 3. Deploy do Modelo no Smartphone via HTTP REST

O aplicativo integra um **servidor HTTP autônomo (NanoHTTPD na porta 8080)** que permite o upload direto para o `filesDir` interno:

```bash
# Obtenha o IP do smartphone na rede Wi-Fi (ex: 192.168.3.13)
curl -X POST http://192.168.3.13:8080/api/v1/models/upload \
  -F "file=@./models/Qwen3.5-2B.Q4_K_M.gguf" \
  -F "name=Qwen3.5-2B.Q4_K_M.gguf"
```

### 4. Executando Inferência Local

```bash
curl -X POST http://192.168.3.13:8080/api/v1/llm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-2B.Q4_K_M.gguf",
    "prompt": "<|im_start|>user\nQual a diferença entre uma NPU e uma GPU em processamento de tensores?<|im_end|>\n<|im_start|>assistant\n",
    "max_tokens": 350,
    "temperature": 0.7
  }'
```
