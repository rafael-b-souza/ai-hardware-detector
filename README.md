# AI Hardware Detector & On-Device LLM Runtime

[![Build Status](https://github.com/rafael-b-souza/ai-hardware-detector/actions/workflows/build-ndk.yml/badge.svg)](https://github.com/rafael-b-souza/ai-hardware-detector/actions/workflows/build-ndk.yml)
[![Kotlin](https://img.shields.io/badge/Kotlin-1.9.0-purple.svg)](https://kotlinlang.org)
[![C++17 NDK](https://img.shields.io/badge/NDK-r27c%20(C%2B%2B17)-blue.svg)](https://developer.android.com/ndk)
[![llama.cpp](https://img.shields.io/badge/Engine-llama.cpp%20GGUF-green.svg)](https://github.com/ggerganov/llama.cpp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An open-source Android framework in **Kotlin & C++17 (NDK)** designed to automatically detect hardware accelerators (Qualcomm Snapdragon NPU, Adreno GPU, Hexagon DSP, and ARM Cortex CPU) and route models to their optimal backend. Features an embedded **`llama.cpp`** runtime for on-device GGUF reasoning models, TensorFlow Lite (NNAPI/GPU Delegate), and ONNX Runtime.

Validated on real silicon: **Samsung Galaxy S22 (Snapdragon 8 Gen 1)**, **Galaxy S21 (Snapdragon 888)**, and **Galaxy Note 8 (Snapdragon 835)**.

*Read this in [Português](README_pt.md).*

---

## 🚀 Benchmark & Inference Matrix (`llama.cpp` on Snapdragon 8 Gen 1)

Inference executed on-device on a **Samsung Galaxy S22 (Qualcomm Snapdragon 8 Gen 1, 8 GB RAM, Android 14 / One UI 6.1)** running **Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled (Q4_K_M, ~1.2 GB)** via NDK with `llama.cpp` (build r27c / C++17).

| Accelerator | Backend / Driver | Format / Quant | Speed (tok/s) | TTFT (Prompt Eval) | RAM Footprint | Thermal & Stability | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ARM CPU** (Kryo: Cortex-X2 + A710) | `ggml-cpu` (NEON + `dotprod` + `i8mm`) | Q4_K_M / INT8 | **7.70 tok/s** | ~115 ms (32 tokens) | ~1.4 GB (native `mmap`) | Stable. 4 threads locked to Prime/Gold cores prevent thermal throttling. | **Production (Recommended)** |
| **Adreno 730 GPU** | `ggml-vulkan` (Qualcomm Vulkan ICD) | FP16 / Q4_K_M | *Load Failure* | N/A | Allocation rejected | **Crash (SIGSEGV)**. Adreno proprietary driver fails allocating `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` on large GGUF tensor maps. | Incompatible (Vendor Driver Bug) |
| **Adreno 730 GPU** | OpenCL / TFLite GPU Delegate | FP16 / FP32 | ~18-24 tok/s (est.) | ~85 ms | ~1.6 GB | Stable for computer vision (MobileNet, BiRefNet); requires custom OpenCL wrapper for llama. | Partial Support (Vision / TFLite) |
| **Hexagon 780 NPU** | Qualcomm QNN / HTP v73 (DSP) | INT4 / INT8 | ~28-35 tok/s (theor.) | <40 ms | <1.0 GB (TCM memory) | Maximum efficiency (mW/token); requires offline quantization via Qualcomm AI Hub. | Roadmap (HTP v73 direct) |

### 🔬 Technical Deep-Dive: Why ARMv8.4-a CPU Outperforms Vulkan on Android

1. **ARMv8.4-a / ARMv9 Vector Intrinsics (`+dotprod+i8mm`)**:
   Compiling `ggml-cpu` with `-march=armv8.4-a+dotprod+i8mm -O3` allows Clang to emit vector instructions (`SDOT`/`UDOT` and `SMMLA` INT8 matrix multiply). This achieves 4 integer operations per cycle on each 128-bit SIMD lane across the Cortex-X2 and Cortex-A710 cores, sustaining **7.70 tok/s** on a 2B reasoning model while generating multi-hundred token `<think>` chains without stutter.
2. **Qualcomm Adreno Vulkan Driver SIGSEGV (`ggml_backend_buft_alloc_buffer`)**:
   While the `llama.cpp` Vulkan backend compiles cleanly against Vulkan 1.3 headers, Qualcomm's proprietary driver on stock Android (One UI with restrictive SELinux) returns `null` when allocating device-local memory (`VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`) for large memory-mapped GGUF weights. Dereferencing causes `SIGSEGV (SEGV_MAPERR at 0x0)`. Until OEM drivers fix this allocation behavior, the multi-threaded NEON CPU path remains the only deterministic, crash-free option.
3. **Zero Java Heap Overhead (Out-Of-Memory Prevention)**:
   GGUF and ONNX models are mapped directly using native `mmap()` syscalls in C++ (`filesDir`). The JVM heap remains under 60 MB, preventing the Android Low Memory Killer (LMK) from terminating background generation sessions.

---

## 🏗️ Architecture

```
+-----------------------------------------------------------------------------------------+
|                                    APPLICATION LAYER                                    |
|   +---------------------------------------------+  +--------------------------------+   |
|   |         Jetpack Compose App UI              |  |   NanoHTTPD REST API (:8080)   |   |
|   |  - Hardware monitoring & TOPS benchmark     |  |   - POST /api/v1/llm/generate  |   |
|   |  - Thread affinity, context & sampling      |  |   - POST /api/v1/models/upload |   |
|   |  - On-device vision & chat interface        |  |   - GET  /api/v1/models        |   |
|   +---------------------------------------------+  +--------------------------------+   |
+-----------------------------------------------------------------------------------------+
                                             |
                                             v
+-----------------------------------------------------------------------------------------+
|                          HARDWARE DETECTION & ROUTING ENGINE                            |
|   +--------------------------+  +------------------------+  +-----------------------+   |
|   | HardwareDetector.kt      |  | DelegateFactory.kt     |  | BenchmarkRunner.kt    |   |
|   | - SoC / Board detection  |  | - Fallback Chain:      |  | - Real-time tok/s     |   |
|   | - Scans /vendor/lib64    |  |   NPU -> GPU -> CPU    |  | - Prompt eval TTFT    |   |
|   | - Estimates TOPS         |  | - TFLite / ORT routing |  | - Vision inference FPS|   |
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
|                  v                       |  |  | - CPU EP with direct mmap           |  |
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

## 📱 Hardware Support Matrix

| Device | SoC / Platform | Dedicated NPU | GPU | Optimal Accelerator | Recommended LLM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Samsung Galaxy S22** | Snapdragon 8 Gen 1 | Hexagon 780 (27 TOPS) | Adreno 730 | NPU (Vision) / CPU NEON (LLM) | Qwen3.5-2B Q4_K_M (7.7 tok/s) |
| **Samsung Galaxy S21** | Snapdragon 888 | Hexagon 780 (26 TOPS) | Adreno 660 | NPU (Vision) / CPU NEON (LLM) | Gemma 2B INT8 / Phi-2 INT4 |
| **Samsung Galaxy Note 8** | Snapdragon 835 | None | Adreno 540 | GPU OpenCL / CPU | SmolLM-135M / DistilGPT2 |
| **Exynos Variants (S21/S22)** | Exynos 2100 / 2200 | Samsung NPU (22–26 TOPS) | Mali-G78 / Xclipse 920 | NPU (via NNAPI) | Gemma 2B INT8 |

---

## 🛠️ Quickstart: Building & Running

### 1. Requirements

- **Android SDK Platform 34** (minSdk 28 — Android 9.0+).
- **Android NDK r27c** (`27.2.12479018` or higher).
- **CMake 3.22.1+** and **Ninja**.

### 2. Build the APK

```bash
./gradlew assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### 3. Upload GGUF Models directly over Wi-Fi (Bypassing Android 11+ SELinux)

Modern Android devices enforce strict scoped storage, making large `adb push` commands cumbersome. This app runs an embedded **NanoHTTPD REST server on port 8080**, allowing direct uploads into the app's sandboxed `filesDir`:

```bash
# Upload model directly to the phone IP (e.g. 192.168.3.13)
curl -X POST http://192.168.3.13:8080/api/v1/models/upload \
  -F "file=@./models/Qwen3.5-2B.Q4_K_M.gguf" \
  -F "name=Qwen3.5-2B.Q4_K_M.gguf"
```

### 4. Execute On-Device Inference via REST

Turn your smartphone into a local inference server for any machine on your LAN:

```bash
curl -X POST http://192.168.3.13:8080/api/v1/llm/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-2B.Q4_K_M.gguf",
    "prompt": "<|im_start|>user\nExplain the difference between NPU and GPU tensor cores in under 50 words.<|im_end|>\n<|im_start|>assistant\n",
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

**Response Payload:**
```json
{
  "tokens_generated": 68,
  "tokens_per_second": 7.703,
  "total_ms": 8827.6,
  "model": "Qwen3.5-2B.Q4_K_M.gguf",
  "text": "NPUs use systolic arrays optimized for low-precision INT8/INT4 matrix multiply-accumulate operations with minimal power draw. GPUs feature parallel SIMD/SIMT shader cores optimized for floating-point calculations, delivering high peak throughput at the cost of higher thermal and energy overhead."
}
```

### 5. Multi-Device Cluster & Python Load Balancer

Included in `client/ai_client.py`: a distributed client that auto-discovers multiple Android phones on the LAN, probes their TOPS rating, and dynamically load balances batch inference across devices!

```bash
python client/ai_client.py --hosts 192.168.3.13:8080 192.168.3.14:8080 --batch prompts.jsonl
```

---

## ⚡ Thermal & Thread Optimization Guidelines

- **Thread Affinity (`nThreads = 4`)**: On 1+3+4 octa-core chips (1 Prime Cortex-X2 + 3 Gold Cortex-A710 + 4 Silver Cortex-A510), setting `nThreads = 4` achieves 100% saturation of high-IPC cores without choking the OS UI thread or overheating low-efficiency Silver cores.
- **Context Size (`n_ctx = 2048`)**: Keeps the KV cache within manageable bounds, preventing memory bloat and preserving battery life during continuous multi-turn generation.

---

## 📄 License

MIT License. Contributions and PRs welcome!
