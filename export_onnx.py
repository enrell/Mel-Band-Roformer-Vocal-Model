import argparse
import yaml
import torch
import torch.nn as nn
from ml_collections import ConfigDict
from utils import get_model_from_config
import os
import gc

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
    
    output_path = args.output_path
    if not output_path:
        model_name = os.path.splitext(os.path.basename(args.model_path))[0] if args.model_path else "mel_band_roformer"
        output_path = f"{model_name}.onnx"

    print(f"Exporting to {output_path}...")
    
    # Disable dynamo for export compatibility
    # Set dynamo=False explicitly
    try:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
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
    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/config_vocals_mel_band_roformer.yaml", help="Path to config file")
    parser.add_argument("--model_type", type=str, default="mel_band_roformer", help="Model type")
    parser.add_argument("--model_path", type=str, default="", help="Path to PyTorch checkpoint")
    parser.add_argument("--output_path", type=str, default="", help="Output ONNX file path")
    
    args = parser.parse_args()
    export_to_onnx(args)
