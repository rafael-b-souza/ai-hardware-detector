# AI Hardware Detector

Detecção e seleção de aceleradores de hardware para inferência de modelos de IA no Android (NPU, GPU, DSP e CPU).

## Dispositivos Testados

| Dispositivo | Chipset | NPU | GPU | DSP | Melhor Para |
|------------|---------|-----|-----|-----|-------------|
| **Galaxy S22** | Snapdragon 8 Gen 1 | ✅ Hexagon 780 (27 TOPS) | ✅ Adreno 730 | ✅ Hexagon | Todos os modelos |
| **Galaxy S21** | Snapdragon 888 | ✅ Hexagon 780 (26 TOPS) | ✅ Adreno 660 | ✅ Hexagon | Todos os modelos |
| **Galaxy Note 8** | Snapdragon 835 | ❌ Sem NPU | ✅ Adreno 540 | ⚠️ Limitado | Modelos leves |

## Arquitetura

```
┌─────────────────────────────────────────┐
│              SUA APP / UI               │
├─────────────────────────────────────────┤
│          ModelRunners                    │
│  VideoDetector │ LLMRunner │ Classifier │
├─────────────────────────────────────────┤
│          DelegateFactory                │
│  Cria Interpreter com melhor delegate   │
│  Fallback automático: NPU→GPU→CPU      │
├─────────────────────────────────────────┤
│          HardwareDetector               │
│  Detecta: NPU, GPU, DSP, CPU           │
│  Identifica chipset e capacidades       │
├─────────────────────────────────────────┤
│          BenchmarkRunner                │
│  Testa performance real de cada accel.  │
└─────────────────────────────────────────┘
```

## Uso Rápido

### 1. Detectar Hardware
```kotlin
val detector = HardwareDetector(context)
val accelerators = detector.detectAllAccelerators()

accelerators.forEach { accel ->
    println("${accel.type}: ${accel.name} - ${accel.estimatedTOPS} TOPS")
}
```

### 2. Seleção de Acelerador
```kotlin
val bestForVideo = detector.getBestAccelerator(ModelType.VIDEO_DETECTION)
val bestForLLM = detector.getBestAccelerator(ModelType.LLM_INFERENCE)
```

### 3. Criar Interpreter
```kotlin
val factory = DelegateFactory(context)
val result = factory.createInterpreter(
    modelPath = "ssd_mobilenet_v2.tflite",
    modelType = ModelType.OBJECT_DETECTION
)

println("Usando: ${result.delegateName}")
```

### 4. Detecção de Objetos em Vídeo
```kotlin
val detector = VideoObjectDetector(context)
detector.initialize("ssd_mobilenet_v2.tflite")

val detections = detector.detectFrame(bitmap)
detections.forEach { det ->
    println("${det.label}: ${det.confidence * 100}% em ${det.boundingBox}")
}
```

### 5. LLM Local
```kotlin
val llm = LocalLLMRunner(context)
llm.initialize("gemma_2b_int8.tflite")

val response = llm.generate(
    prompt = "Explique machine learning em 3 frases:",
    maxTokens = 128,
    temperature = 0.7f
)
println("${response.text}")
println("${response.tokensPerSecond} tokens/s via ${response.acceleratorUsed}")
```

### 6. Benchmark
```kotlin
val benchmark = BenchmarkRunner(context)
val results = benchmark.benchmarkModel(
    modelPath = "model.tflite",
    modelType = ModelType.OBJECT_DETECTION
)

results.forEach { r ->
    println("${r.delegateName}: ${r.avgInferenceMs}ms (${r.fps} FPS)")
}
```

## Modelos Recomendados

### Para S22 / S21 (com NPU)
| Tarefa | Modelo | Tamanho | Performance Esperada |
|--------|--------|---------|---------------------|
| Detecção de objetos | EfficientDet-Lite2 | 7.2 MB | ~30 FPS |
| Detecção de vídeo | SSD MobileNet V2 | 6.9 MB | ~60 FPS |
| LLM | Gemma 2B INT8 | 1.4 GB | ~10 tok/s |
| LLM | Phi-2 INT4 | 1.5 GB | ~8 tok/s |
| Classificação | EfficientNet-Lite0 | 4.7 MB | ~120 FPS |
| Segmentação | DeepLab V3 | 2.7 MB | ~25 FPS |
| Pose | MoveNet Lightning | 3.6 MB | ~50 FPS |

### Para Note 8 (sem NPU, GPU only)
| Tarefa | Modelo | Tamanho | Performance Esperada |
|--------|--------|---------|---------------------|
| Detecção de objetos | SSD MobileNet V1 | 4.3 MB | ~20 FPS |
| Classificação | MobileNet V2 | 3.5 MB | ~40 FPS |
| LLM | SmolLM 135M | 270 MB | ~5 tok/s |
| LLM | DistilGPT2 | 320 MB | ~3 tok/s |

## Como Executar

1. Clone o repositório
2. Abra no Android Studio
3. Posicione os modelos `.tflite` em `app/src/main/assets/`
4. Compile e instale no dispositivo

## Matriz de Seleção de Acelerador

```
Tipo de Modelo          → Prioridade de Acelerador
─────────────────────────────────────────────────
Classificação de Imagem → NPU > GPU > CPU
Detecção de Objetos     → NPU > GPU > CPU
Detecção em Vídeo       → NPU > GPU > CPU
Inferência LLM          → GPU > NPU > CPU
Estimativa de Pose      → NPU > GPU > CPU
Segmentação             → NPU > GPU > CPU
OCR                     → NPU > DSP > GPU > CPU
Classificação de Áudio  → DSP > NPU > CPU
```

## Notas de Operação

- **Fallback**: Se o delegate NPU falhar em uma operação, o runtime tenta GPU e em seguida CPU.
- **Cache NNAPI**: A compilação inicial do modelo leva cerca de 2–5s; execuções subsequentes utilizam o cache gerado em disco.
- **Quantização (INT8)**: Reduz consumo de memória e latência de inferência no subsistema Hexagon (NPU/DSP).
- **Modelos de linguagem (>2B parâmetros)**: Requerem ao menos 4 GB de memória RAM livre.
- **Galaxy Note 8**: Limitado a 6 GB de RAM e GPU Adreno 540; utilizar modelos quantizados abaixo de 500 MB.
- **Variantes Exynos**: Modelos com SoC Exynos utilizam a Samsung NPU em vez do Hexagon da Qualcomm.
