import argparse
import yaml
import torch
import torch.nn as nn
from ml_collections import ConfigDict
from utils import get_model_from_config
import os
import gc


def convert_to_fp16(input_path, output_path):
    """Convert ONNX model to FP16."""
    from onnxruntime.transformers import float16
    import onnx
    
    print(f"Converting to FP16: {input_path} -> {output_path}")
    model = onnx.load(input_path)
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, output_path)
    print(f"FP16 conversion complete!")


def quantize_int8(input_path, output_path):
    """Quantize ONNX model to INT8 using dynamic quantization."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    print(f"Quantizing to INT8: {input_path} -> {output_path}")
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8
    )
    print(f"INT8 quantization complete!")


def quantize_int4(input_path, output_path):
    """Quantize ONNX model to INT4 using weight-only quantization."""
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    
    print(f"Quantizing to INT4: {input_path} -> {output_path}")
    quantizer = MatMulNBitsQuantizer(
        model=input_path,
        block_size=128,
        is_symmetric=False,
    )
    quantizer.process()
    quantizer.model.save_model_to_file(output_path)
    print(f"INT4 quantization complete!")


def quantize_fp8(input_path, output_path):
    """Quantize ONNX model to FP8 (E4M3FN) using dynamic quantization."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    
    print(f"Quantizing to FP8: {input_path} -> {output_path}")
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QFLOAT8E4M3FN
    )
    print(f"FP8 quantization complete!")


def export_to_onnx(args):
    # Load config
    with open(args.config_path) as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

    # Disable flash attention for ONNX export to avoid opset issues and excessive RAM
    config.model.flash_attn = False

    # Initialize model
    model = get_model_from_config(args.model_type, config)
    
    # Load checkpoint if provided
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading checkpoint from {args.model_path}")
        state_dict = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        del state_dict
        gc.collect()
    else:
        print("No checkpoint provided or found, exporting initialized model")

    model.eval()

    channels = 2 if config.model.stereo else 1
    # Use a smaller size for export to save RAM, dynamic axes will handle larger inputs
    # 44100 is 1 second at 44.1kHz, enough for STFT with n_fft=2048
    dummy_input_size = 44100 
    dummy_input = torch.randn(1, channels, dummy_input_size)
    
    # Determine output paths
    if args.output_path:
        base_output = args.output_path
        if base_output.endswith('.onnx'):
            base_output = base_output[:-5]
    else:
        model_name = os.path.splitext(os.path.basename(args.model_path))[0] if args.model_path else "mel_band_roformer"
        base_output = model_name
    
    fp32_output = f"{base_output}.onnx"

    print(f"Exporting to {fp32_output}...")
    
    # Disable dynamo for export compatibility
    # Set dynamo=False explicitly
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                fp32_output,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size', 2: 'time_steps'},
                    'output': {0: 'batch_size', 2: 'time_steps'}
                },
                opset_version=17,
                do_constant_folding=True,
                dynamo=False  # Important: Use legacy exporter for complex STFT operations
            )
        print("Export successful!")
        
        # Apply quantization if requested
        if args.fp16:
            fp16_output = f"{base_output}_fp16.onnx"
            convert_to_fp16(fp32_output, fp16_output)
            print(f"FP16 model saved to: {fp16_output}")
        
        if args.int8:
            int8_output = f"{base_output}_int8.onnx"
            quantize_int8(fp32_output, int8_output)
            print(f"INT8 model saved to: {int8_output}")
        
        if args.int4:
            int4_output = f"{base_output}_int4.onnx"
            quantize_int4(fp32_output, int4_output)
            print(f"INT4 model saved to: {int4_output}")
        
        if args.fp8:
            fp8_output = f"{base_output}_fp8.onnx"
            quantize_fp8(fp32_output, fp8_output)
            print(f"FP8 model saved to: {fp8_output}")
            
    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Mel-Band Roformer model to ONNX format with optional quantization")
    parser.add_argument("--config_path", type=str, default="configs/config_vocals_mel_band_roformer.yaml", help="Path to config file")
    parser.add_argument("--model_type", type=str, default="mel_band_roformer", help="Model type")
    parser.add_argument("--model_path", type=str, default="", help="Path to PyTorch checkpoint")
    parser.add_argument("--output_path", type=str, default="", help="Output ONNX file path (without extension for quantized variants)")
    
    # Quantization options
    parser.add_argument("--fp16", action="store_true", help="Also export FP16 quantized model (roughly half the size)")
    parser.add_argument("--int8", action="store_true", help="Also export INT8 quantized model (roughly quarter the size)")
    parser.add_argument("--int4", action="store_true", help="Also export INT4 quantized model (roughly 1/8 the size)")
    parser.add_argument("--fp8", action="store_true", help="Also export FP8 (E4M3FN) quantized model")
    
    args = parser.parse_args()
    export_to_onnx(args)
