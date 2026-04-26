"""WF-Diff baseline reproduction for apples-to-apples comparison.

Loads WF-Diff pretrained weights, runs inference on the test set,
and computes metrics using the same pipeline as WaveFlow-UIE evaluation.

Entry point:
  python -m waveflow_uie.baseline_wfdiff \\
    --weights pretrained/net_g_405000_UIEB.pth \\
    --dataset uieb_test \\
    --output results/wfdiff_baseline/
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
import torch

from waveflow_uie.evaluate import DATASET_PRESETS, compute_metrics
from waveflow_uie.utils.seed import set_seed


def run_baseline(args):
    """Run WF-Diff baseline evaluation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed, deterministic=True)

    # Add project root to path for basicsr imports
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    # Import WF-Diff components
    from basicsr.archs import build_network

    # DWT/IWT from WF-Diff (batch-concat version)
    from basicsr.archs.Padiff_arch.wavelet import DWT, IWT

    # Build WF-Diff network
    network_opt = {
        'type': 'WfDiffx2',
        'in_channel': 6,
        'out_channel': 3,
        'inner_channel': 48,
        'norm_groups': 24,
        'with_time_emb': True,
        'schedule_opt': {
            'schedule': 'linear',
            'n_timestep': 2000,
            'linear_start': 1e-6,
            'linear_end': 1e-2,
        },
        'sample_proc': 'ddim',
        'local_ensemble': True,
        'feat_unfold': True,
        'cell_decode': True,
        'ppg_input_channels': 3,
    }

    print("Building WF-Diff network...")
    net_g = build_network(network_opt)
    net_g = net_g.to(device)

    # Load pretrained weights
    print(f"Loading weights from {args.weights}")
    state = torch.load(args.weights, map_location=device, weights_only=False)
    if 'params' in state:
        state = state['params']
    elif 'params_ema' in state:
        state = state['params_ema']
    net_g.load_state_dict(state, strict=True)
    net_g.eval()

    # Dataset
    if args.dataset in DATASET_PRESETS:
        ds_cfg = DATASET_PRESETS[args.dataset]
    else:
        ds_cfg = {'input_dir': args.input_dir, 'target_dir': args.target_dir}

    # Load test images manually (avoid BasicSR dataset complexity)
    import glob
    input_paths = sorted(glob.glob(os.path.join(ds_cfg['input_dir'], '*')))
    target_paths = sorted(glob.glob(os.path.join(ds_cfg['target_dir'], '*')))
    assert len(input_paths) == len(target_paths), "Mismatch between input and target counts"

    # Output directory
    img_dir = os.path.join(args.output, 'images')
    os.makedirs(img_dir, exist_ok=True)

    dwt = DWT()
    idwt = IWT()
    metric_names = ['psnr', 'ssim', 'lpips', 'uciqe', 'uiqm']
    totals = {m: 0.0 for m in metric_names}
    total_time = 0.0
    per_image_results = []

    print(f"Running inference on {len(input_paths)} images...")

    for idx, (inp_path, tgt_path) in enumerate(zip(input_paths, target_paths)):
        # Load images
        img_lq = cv2.imread(inp_path, cv2.IMREAD_COLOR)
        img_gt = cv2.imread(tgt_path, cv2.IMREAD_COLOR)

        img_lq_rgb = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_gt_rgb = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if args.resize > 0:
            img_lq_rgb = cv2.resize(img_lq_rgb, (args.resize, args.resize), interpolation=cv2.INTER_AREA)
            img_gt_rgb = cv2.resize(img_gt_rgb, (args.resize, args.resize), interpolation=cv2.INTER_AREA)
        else:
            # Pad to multiple of 8
            h, w = img_lq_rgb.shape[:2]
            new_h = h - (h % 8) if h % 8 != 0 else h
            new_w = w - (w % 8) if w % 8 != 0 else w
            if new_h != h or new_w != w:
                img_lq_rgb = cv2.resize(img_lq_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                img_gt_rgb = cv2.resize(img_gt_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # To tensor CHW
        lq_tensor = torch.from_numpy(img_lq_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
        gt_tensor = torch.from_numpy(img_gt_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

        # WF-Diff inference
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            out1, out2ll, _, out2high, _, _, _ = net_g(lq_tensor, None)
            n = out1.shape[0]
            out1dwt = dwt(out1)
            out1LL, out1high0 = out1dwt[:n], out1dwt[n:]
            finalLL = out1LL + out2ll
            finalHH = out1high0 + out2high
            output = idwt(torch.cat((finalLL, finalHH), dim=0))
        torch.cuda.synchronize()
        elapsed = time.time() - start
        total_time += elapsed

        # Convert to numpy
        pred_np = (output[0].cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        gt_np = (gt_tensor[0].cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)

        # Resize if shapes don't match
        if pred_np.shape != gt_np.shape:
            pred_np = cv2.resize(pred_np, (gt_np.shape[1], gt_np.shape[0]), interpolation=cv2.INTER_AREA)

        # Compute metrics
        metrics = compute_metrics(pred_np, gt_np)
        for m in metric_names:
            totals[m] += metrics[m]

        img_name = os.path.basename(inp_path)
        per_image_results.append({'image': img_name, 'time_ms': elapsed * 1000, **metrics})

        # Save enhanced image
        out_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(img_dir, img_name), out_bgr)

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(input_paths)}] PSNR: {metrics['psnr']:.2f}")

    # Averages
    count = len(input_paths)
    avg = {m: totals[m] / max(count, 1) for m in metric_names}
    avg_time = (total_time / max(count, 1)) * 1000

    print(f"\nWF-Diff Baseline Results ({count} images):")
    print(f"  PSNR:  {avg['psnr']:.2f}")
    print(f"  SSIM:  {avg['ssim']:.4f}")
    print(f"  LPIPS: {avg['lpips']:.4f}")
    print(f"  UCIQE: {avg['uciqe']:.4f}")
    print(f"  UIQM:  {avg['uiqm']:.4f}")
    print(f"  Time:  {avg_time:.1f}ms/image")

    # Save CSV
    csv_path = os.path.join(args.output, 'wfdiff_baseline_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['image', 'time_ms'] + metric_names
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_results)

        # Write average row
        writer.writerow({
            'image': 'AVERAGE',
            'time_ms': round(avg_time, 2),
            **{m: round(avg[m], 4) for m in metric_names},
        })

    print(f"Results saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='WF-Diff Baseline Evaluation')
    parser.add_argument('--weights', type=str, default='pretrained/net_g_405000_UIEB.pth',
                        help='Path to WF-Diff pretrained weights')
    parser.add_argument('--dataset', type=str, default='uieb_test',
                        help='Dataset preset (uieb_test, lsui_test)')
    parser.add_argument('--input-dir', type=str, default=None, help='Custom input directory')
    parser.add_argument('--target-dir', type=str, default=None, help='Custom target directory')
    parser.add_argument('--output', type=str, default='results/wfdiff_baseline',
                        help='Output directory')
    parser.add_argument('--resize', type=int, default=0,
                        help='Resize all images to NxN before inference (0 = keep native resolution). '
                             'Standard for benchmarking: 256.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    run_baseline(args)


if __name__ == '__main__':
    main()
