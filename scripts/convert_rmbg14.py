"""
Baixa BiRefNet-general ONNX e converte para FP16 para uso no ONNX Runtime Android.

Input e output são preservados em float32 (keep_io_types=True) para manter o pipeline
de pré e pós-processamento.
"""

import urllib.request
from pathlib import Path

ROOT      = Path(__file__).parent.parent
ONNX_FP32 = ROOT / "birefnet-general.onnx"
ONNX_FP16 = ROOT / "birefnet-general-fp16.onnx"

ONNX_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-general-epoch_244.onnx"


def main():
    if ONNX_FP32.exists():
        print(f"FP32 já existe: {ONNX_FP32}")
    else:
        print("Baixando BiRefNet-general FP32...")
        def progress(count, block, total):
            pct = min(count * block * 100 // total, 100)
            print(f"\r  {pct}%", end="", flush=True)
        urllib.request.urlretrieve(ONNX_URL, ONNX_FP32, reporthook=progress)
        print(f"\nBaixado: {ONNX_FP32} ({ONNX_FP32.stat().st_size // 1024 // 1024} MB)")

    print("\nConvertendo FP32 → FP16...")
    import onnx
    from onnxconverter_common import convert_float_to_float16

    model = onnx.load(str(ONNX_FP32))
    model_fp16 = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(ONNX_FP16))

    size_mb = ONNX_FP16.stat().st_size / 1024 / 1024
    print(f"Salvo: {ONNX_FP16} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
