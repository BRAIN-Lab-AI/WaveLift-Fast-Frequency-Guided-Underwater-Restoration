"""No-reference evaluation for U45, C60, and other unpaired underwater test sets.

Runs WaveFlow-UIE inference on a folder of degraded images and computes
UCIQE/UIQM on both the input and the enhanced output, so 'before vs after'
comparisons land in a single CSV.

Entry point:
  python -m waveflow_uie.eval_no_reference \\
    --checkpoint experiments/<run>/checkpoints/best.pt \\
    --input data/U45 \\
    --output results/u45_both.csv \\
    --enhanced-dir results/u45_both/images \\
    --steps 5 --solver euler --resize 256
"""

import argparse
import csv
import glob
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from waveflow_uie.sample import load_model
from waveflow_uie.utils.metrics import calculate_uciqe, calculate_uiqm
from waveflow_uie.utils.seed import set_seed


def _to_uint8_rgb(arr_float: np.ndarray) -> np.ndarray:
    """Convert HWC float32 [0,1] RGB to HWC uint8 RGB."""
    return (np.clip(arr_float, 0.0, 1.0) * 255.0).astype(np.uint8)


def _score(img_rgb_uint8: np.ndarray) -> dict:
    """Compute no-reference metrics on a uint8 RGB image."""
    img_bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
    return {
        'uciqe': calculate_uciqe(img_bgr),
        'uiqm': calculate_uiqm(img_rgb_uint8),
    }


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed, deterministic=True)

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(args.input, ext)))
    paths = sorted(paths)
    assert paths, f"No images found in {args.input}"

    if args.enhanced_dir:
        os.makedirs(args.enhanced_dir, exist_ok=True)

    print(f"Evaluating {len(paths)} images at {args.resize}x{args.resize} "
          f"with {args.steps} {args.solver} steps")

    rows = []
    sums = {'uciqe_in': 0.0, 'uiqm_in': 0.0, 'uciqe_out': 0.0, 'uiqm_out': 0.0}
    total_time = 0.0

    for idx, path in enumerate(paths):
        # Load + resize input
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"  [WARN] failed to load {path}, skipping")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if args.resize > 0:
            img_rgb = cv2.resize(img_rgb, (args.resize, args.resize),
                                 interpolation=cv2.INTER_AREA)
        else:
            # Pad to even for DWT
            h, w = img_rgb.shape[:2]
            if h % 2 or w % 2:
                img_rgb = np.pad(img_rgb, ((0, h % 2), (0, w % 2), (0, 0)), mode='reflect')

        lq_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

        # Inference
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            enhanced = model.sample(lq_tensor, num_steps=args.steps, solver=args.solver)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start
        total_time += elapsed

        out_rgb = enhanced[0].cpu().numpy().transpose(1, 2, 0)
        in_uint8 = _to_uint8_rgb(img_rgb)
        out_uint8 = _to_uint8_rgb(out_rgb)

        m_in = _score(in_uint8)
        m_out = _score(out_uint8)

        sums['uciqe_in'] += m_in['uciqe']
        sums['uiqm_in'] += m_in['uiqm']
        sums['uciqe_out'] += m_out['uciqe']
        sums['uiqm_out'] += m_out['uiqm']

        if args.enhanced_dir:
            out_bgr = cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args.enhanced_dir, os.path.basename(path)), out_bgr)

        rows.append({
            'image': os.path.basename(path),
            'time_ms': round(elapsed * 1000, 2),
            'uciqe_in': round(m_in['uciqe'], 4),
            'uiqm_in': round(m_in['uiqm'], 4),
            'uciqe_out': round(m_out['uciqe'], 4),
            'uiqm_out': round(m_out['uiqm'], 4),
        })

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(paths)}] UCIQE: {m_in['uciqe']:.3f} -> {m_out['uciqe']:.3f} | "
                  f"UIQM: {m_in['uiqm']:.3f} -> {m_out['uiqm']:.3f}")

    n = len(rows)
    avg = {k: v / n for k, v in sums.items()}
    avg_time = (total_time / n) * 1000

    print(f"\n{'='*60}")
    print(f"Dataset: {args.input}  ({n} images)")
    print(f"  Input  UCIQE: {avg['uciqe_in']:.4f} | UIQM: {avg['uiqm_in']:.4f}")
    print(f"  Output UCIQE: {avg['uciqe_out']:.4f} | UIQM: {avg['uiqm_out']:.4f}")
    print(f"  Delta  UCIQE: {avg['uciqe_out'] - avg['uciqe_in']:+.4f} | "
          f"UIQM: {avg['uiqm_out'] - avg['uiqm_in']:+.4f}")
    print(f"  Avg inference time: {avg_time:.1f} ms/image")

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        fieldnames = ['image', 'time_ms', 'uciqe_in', 'uiqm_in', 'uciqe_out', 'uiqm_out']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({
            'image': 'AVERAGE',
            'time_ms': round(avg_time, 2),
            'uciqe_in': round(avg['uciqe_in'], 4),
            'uiqm_in': round(avg['uiqm_in'], 4),
            'uciqe_out': round(avg['uciqe_out'], 4),
            'uiqm_out': round(avg['uiqm_out'], 4),
        })

    print(f"\nResults: {args.output}")
    if args.enhanced_dir:
        print(f"Enhanced images: {args.enhanced_dir}")


def main():
    parser = argparse.ArgumentParser(description='WaveFlow-UIE No-Reference Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input', type=str, required=True, help='Folder of degraded images')
    parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    parser.add_argument('--enhanced-dir', type=str, default=None,
                        help='Optional folder to save enhanced images')
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--solver', type=str, default='euler', choices=['euler', 'midpoint'])
    parser.add_argument('--resize', type=int, default=256,
                        help='Resize all images to NxN (0 = keep native size). Default: 256.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
