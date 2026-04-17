"""WaveFlow-UIE single-image inference.

Entry point: python -m waveflow_uie.sample --checkpoint <path> --input <image_or_dir> --steps 5
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np
import torch
import yaml

from waveflow_uie.models.waveflow import WaveFlowUIE
from waveflow_uie.utils.ema import EMAHelper
from waveflow_uie.utils.seed import set_seed


def load_model(checkpoint_path: str, device: torch.device) -> WaveFlowUIE:
    """Load model with EMA weights from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct model config from checkpoint if available
    model = WaveFlowUIE()
    model.load_state_dict(ckpt['model_state_dict'])

    # Apply EMA weights to velocity_unet
    if 'ema_state_dict' in ckpt:
        ema_state = ckpt['ema_state_dict']
        for name, param in model.velocity_unet.named_parameters():
            if name in ema_state['shadow']:
                param.data.copy_(ema_state['shadow'][name].to(device))

    model.to(device)
    model.eval()
    return model


def load_image(path: str) -> torch.Tensor:
    """Load image as (1, 3, H, W) tensor in [0, 1], padded to even dims."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Pad to even dimensions
    h, w = img.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, (h, w)


def save_image(tensor: torch.Tensor, path: str, original_size: tuple) -> None:
    """Save (1, 3, H, W) tensor as image, cropping to original size."""
    img = tensor[0].cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
    h, w = original_size
    img = img[:h, :w]
    img = (img * 255.0).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    cv2.imwrite(path, img)


def enhance(args):
    """Run enhancement on single image or directory."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    # Collect input images
    if os.path.isdir(args.input):
        exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(args.input, ext)))
        paths = sorted(paths)
    else:
        paths = [args.input]

    os.makedirs(args.output, exist_ok=True)
    print(f"Enhancing {len(paths)} image(s) with {args.steps} {args.solver} steps")

    total_time = 0.0
    for path in paths:
        img_tensor, original_size = load_image(path)
        img_tensor = img_tensor.to(device)

        start = time.time()
        with torch.no_grad():
            enhanced = model.sample(img_tensor, num_steps=args.steps, solver=args.solver)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        total_time += elapsed

        # Save output
        basename = os.path.basename(path)
        out_path = os.path.join(args.output, basename)
        save_image(enhanced, out_path, original_size)
        print(f"  {basename}: {elapsed:.3f}s")

    avg_time = total_time / len(paths) if paths else 0
    print(f"\nDone. Average: {avg_time:.3f}s/image. Output: {args.output}")


def main():
    parser = argparse.ArgumentParser(description='WaveFlow-UIE Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--input', type=str, required=True, help='Input image or directory')
    parser.add_argument('--output', type=str, default='results/enhanced', help='Output directory')
    parser.add_argument('--steps', type=int, default=5, help='Number of ODE steps')
    parser.add_argument('--solver', type=str, default='euler', choices=['euler', 'midpoint'])
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    enhance(args)


if __name__ == '__main__':
    main()
