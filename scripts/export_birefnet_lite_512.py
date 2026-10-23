"""
Exporta BiRefNet-lite (ZhengPeng7/BiRefNet_lite) para ONNX FP32 em resolução fixa 512x512.

Dimensões fixas evitam incompatibilidades com operações de Reshape no Swin Transformer.
"""

import sys
import torch
import torch.onnx.symbolic_helper as sym_help
import onnx
from pathlib import Path
from transformers import AutoModelForImageSegmentation


# Mapeamento de torchvision::deform_conv2d para ONNX DeformConv (opset 18+)
@sym_help.parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "b")
def _deform_conv2d_onnx(g, input, weight, offset, mask, bias,
                         stride_h, stride_w, pad_h, pad_w,
                         dil_h, dil_w, n_weight_grps, n_offset_grps, use_mask):
    weight_sizes = weight.type().sizes()
    kernel_h, kernel_w = weight_sizes[2], weight_sizes[3]
    return g.op(
        "DeformConv",
        input, weight, offset, bias, mask,
        dilations_i=[dil_h, dil_w],
        group_i=n_weight_grps,
        kernel_shape_i=[kernel_h, kernel_w],
        offset_group_i=n_offset_grps,
        pads_i=[pad_h, pad_w, pad_h, pad_w],
        strides_i=[stride_h, stride_w],
    )

torch.onnx.register_custom_op_symbolic(
    "torchvision::deform_conv2d", _deform_conv2d_onnx, opset_version=1
)

OUTPUT = Path(__file__).parent.parent / "birefnet-lite-fp32.onnx"
REPO   = "ZhengPeng7/BiRefNet_lite"
H, W   = 512, 512


def _patch_deform_conv():
    """
    Substitui chamadas funcionais diretas a deform_conv2d por conv2d antes do trace.
    Necessário porque o ONNX Runtime Android não possui kernel para DeformConv.
    """
    import sys
    import torch.nn.functional as F
    import torchvision.ops as tvops
    import torchvision.ops.deform_conv as _dc

    def _as_regular_conv(input, offset, weight, bias=None,
                         stride=(1, 1), padding=(0, 0), dilation=(1, 1), mask=None):
        if isinstance(stride, int):   stride   = (stride,   stride)
        if isinstance(padding, int):  padding  = (padding,  padding)
        if isinstance(dilation, int): dilation = (dilation, dilation)
        return F.conv2d(input, weight, bias, stride, padding, dilation)

    tvops.deform_conv2d = _as_regular_conv
    _dc.deform_conv2d   = _as_regular_conv

    patched = []
    for mod_name, mod in list(sys.modules.items()):
        if mod is not None and getattr(mod, 'deform_conv2d', None) is not None:
            try:
                mod.deform_conv2d = _as_regular_conv
                patched.append(mod_name)
            except (AttributeError, TypeError):
                pass

    print(f"  deform_conv2d substituída em: {patched}")


def main():
    print(f"=== BiRefNet-lite ONNX Export ({H}x{W}) ===\n")

    print(f"Carregando {REPO}...")
    model = AutoModelForImageSegmentation.from_pretrained(REPO, trust_remote_code=True)
    model.eval()
    print(f"  Parâmetros: {sum(p.numel() for p in model.parameters()):,}")

    _patch_deform_conv()

    dummy = torch.randn(1, 3, H, W)

    print("\nRodando trace...")
    with torch.no_grad():
        outputs = model(dummy)

    n_outputs = len(outputs) if isinstance(outputs, (list, tuple)) else 1
    print(f"  Outputs encontrados: {n_outputs}")

    input_names  = ["input_image"]
    output_names = [f"output_{i}" for i in range(n_outputs)]

    print(f"\nExportando ONNX → {OUTPUT}...")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(OUTPUT),
            opset_version=19,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=None,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
        )

    print("\nValidando com onnx.checker...")
    model_onnx = onnx.load(str(OUTPUT))
    onnx.checker.check_model(model_onnx)
    print("  Modelo válido")

    size_mb = OUTPUT.stat().st_size / 1024 / 1024
    graph   = model_onnx.graph

    print(f"\n{'='*50}")
    print(f"Arquivo : {OUTPUT}")
    print(f"Tamanho : {size_mb:.1f} MB")
    print(f"Nós     : {len(graph.node)}")

    print("\nInputs:")
    for inp in graph.input:
        shape = [d.dim_value if d.HasField('dim_value') else d.dim_param
                 for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {shape}")

    print("\nOutputs:")
    for out in graph.output:
        shape = [d.dim_value if d.HasField('dim_value') else d.dim_param
                 for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {shape}")


if __name__ == "__main__":
    main()
