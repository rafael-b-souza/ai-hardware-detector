"""
Converte BRIA RMBG-1.4 ONNX para TFLite para inferência no Android.
"""

from pathlib import Path

ASSETS = Path(__file__).parent.parent / "app/src/main/assets"
OUTPUT  = ASSETS / "rmbg2.tflite"
ONNX_TMP = ASSETS / "rmbg_tmp.onnx"


def main():
    print("Downloading RMBG-1.4 ONNX...")
    from huggingface_hub import hf_hub_download
    onnx_path = hf_hub_download(
        repo_id="briaai/RMBG-1.4",
        filename="onnx/model.onnx",
        local_dir=str(ASSETS),
    )
    print(f"Downloaded: {onnx_path}")

    print("\nConverting ONNX → TFLite...")
    import subprocess, sys, shutil

    result = subprocess.run(
        [sys.executable, "-c", "import onnx2tf; onnx2tf.convert("
         f"input_onnx_file_path='{onnx_path}',"
         f"output_folder_path='{ASSETS}/onnx2tf_out',"
         "non_verbose=True)"],
        capture_output=True, text=True
    )

    out_dir = ASSETS / "onnx2tf_out"
    tflites = list(out_dir.glob("*.tflite")) if out_dir.exists() else []

    if tflites:
        tflites[0].rename(OUTPUT)
    else:
        print("Fallback para onnx-tf...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnx-tf", "tensorflow-cpu"])
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        model = onnx.load(onnx_path)
        tf_rep = prepare(model)
        saved_dir = str(ASSETS / "saved_model_tmp")
        tf_rep.export_graph(saved_dir)
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
        tflite_model = converter.convert()
        OUTPUT.write_bytes(tflite_model)
        shutil.rmtree(saved_dir, ignore_errors=True)

    print(f"Saved: {OUTPUT} ({OUTPUT.stat().st_size // 1024 // 1024} MB)")
    shutil.rmtree(out_dir, ignore_errors=True)
    Path(onnx_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
