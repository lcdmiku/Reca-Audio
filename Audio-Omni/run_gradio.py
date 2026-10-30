from audio_omni import get_pretrained_model
from audio_omni.interface.gradio import create_ui
import json 

import torch

def main(args):
    torch.manual_seed(42)

    interface = create_ui(
        model_config_path = args.model_config, 
        ckpt_path=args.ckpt_path, 
        pretrained_name=args.pretrained_name, 
        pretransform_ckpt_path=args.pretransform_ckpt_path,
        model_half=args.model_half,
        conditioner_device=args.cond_device,
        dtype=args.dtype,
    )
    interface.queue()
    interface.launch(share=args.share, server_name="0.0.0.0", server_port=args.server_port, auth=(args.username, args.password) if args.username is not None else None)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run gradio interface')
    parser.add_argument('--pretrained-name', type=str, help='Name of pretrained model', required=False)
    parser.add_argument('--model-config', type=str, help='Path to model config', required=False)
    parser.add_argument('--ckpt-path', type=str, help='Path to model checkpoint', required=False)
    parser.add_argument('--pretransform-ckpt-path', type=str, help='Optional to model pretransform checkpoint', required=False)
    parser.add_argument('--share', action='store_true', help='Create a publicly shareable link', required=False)
    parser.add_argument('--server-port', type=int, help='Port to run the gradio interface on', required=False, default=7860)
    parser.add_argument('--username', type=str, help='Gradio username', required=False)
    parser.add_argument('--password', type=str, help='Gradio password', required=False)
    parser.add_argument('--model-half', action='store_true', help='Whether to use half precision', required=False)
    parser.add_argument('--cond-device', type=str, default=None,
                        help='把 Qwen 理解塔放到另一张卡（组件并行），如 cuda:1')
    parser.add_argument('--dtype', type=str, default=None, choices=['float16', 'float32'],
                        help='生成头（DiT）精度；float16 可省一半权重显存')
    args = parser.parse_args()
    main(args)