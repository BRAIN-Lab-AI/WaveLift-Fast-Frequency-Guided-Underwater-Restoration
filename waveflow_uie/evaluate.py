"""WaveFlow-UIE evaluation with multi-step and multi-solver support.

Entry point:
  python -m waveflow_uie.evaluate \\
    --checkpoint <path> --dataset uieb_test \\
    --steps 1,2,5,10,20 --solvers euler,midpoint \\
    --output results/step_ablation.csv
"""

import argparse
import csv
import os
import time

import cv2
import numpy as np
import torch

from waveflow_uie.data.dataset import PairedUIEDataset
from waveflow_uie.sample import load_model
from waveflow_uie.utils.seed import set_seed


# Dataset path presets
DATASET_PRESETS = {
    'uieb_test': {
        'input_dir': 'data/UIEB/test/input',
        'target_dir': 'data/UIEB/test/target',
    },
    'lsui_test': {
        'input_dir': 'data/LSUI/test/input',
        'target_dir': 'data/LSUI/test/GT',
    },
    'both_test': {
        'input_dir': 'data/Both/test/input',
        'target_dir': 'data/Both/test/target',
    },
}


def compute_metrics(pred_np: np.ndarray, gt_np: np.ndarray) -> dict:
    """Compute all metrics on a single image pair.

    Args:
        pred_np: Predicted image, uint8 HWC RGB [0, 255].
        gt_np: Ground truth image, uint8 HWC RGB [0, 255].
    """
    from waveflow_uie.utils.metrics import (
        calculate_lpips,
        calculate_psnr,
        calculate_ssim,
        calculate_uciqe,
        calculate_uiqm,
    )

    # PSNR and SSIM need matching shapes
    if pred_np.shape != gt_np.shape:
        pred_np = cv2.resize(pred_np, (gt_np.shape[1], gt_np.shape[0]), interpolation=cv2.INTER_AREA)

    # Our metrics module accepts RGB input with is_bgr=False
    psnr = calculate_psnr(pred_np, gt_np, crop_border=2, test_y_channel=True, is_bgr=False)
    ssim = calculate_ssim(pred_np, gt_np, crop_border=2, test_y_channel=True, is_bgr=False)

    # LPIPS needs BGR
    pred_bgr = cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR)
    gt_bgr = cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR)
    lpips_val = calculate_lpips(pred_bgr, gt_bgr)

    # UCIQE expects BGR, UIQM expects RGB
    uciqe = calculate_uciqe(pred_bgr)
    uiqm = calculate_uiqm(pred_np)

    return {
        'psnr': psnr,
        'ssim': ssim,
        'lpips': lpips_val,
        'uciqe': uciqe,
        'uiqm': uiqm,
    }


def evaluate(args):
    """Run evaluation across multiple step counts and solvers."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed, deterministic=True)

    # Parse step counts and solvers
    steps_list = [int(s) for s in args.steps.split(',')]
    solvers = [s.strip() for s in args.solvers.split(',')]

    # Dataset
    if args.dataset in DATASET_PRESETS:
        ds_cfg = DATASET_PRESETS[args.dataset]
    else:
        ds_cfg = {'input_dir': args.input_dir, 'target_dir': args.target_dir}

    dataset = PairedUIEDataset(
        input_dir=ds_cfg['input_dir'],
        target_dir=ds_cfg['target_dir'],
        is_train=False,
        enlarge_ratio=1,
    )

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    # Results storage
    all_results = []
    metric_names = ['psnr', 'ssim', 'lpips', 'uciqe', 'uiqm']
    all_metric_names = metric_names + ['fid']

    from waveflow_uie.utils.metrics import FIDCalculator

    for solver in solvers:
        for num_steps in steps_list:
            print(f"\n{'='*60}")
            print(f"Evaluating: solver={solver}, steps={num_steps}")
            print(f"{'='*60}")

            totals = {m: 0.0 for m in metric_names}
            total_time = 0.0
            count = 0
            per_image_results = []
            fid_calc = FIDCalculator(device=device)

            for idx in range(len(dataset)):
                batch = dataset[idx]
                lq = batch['lq'].unsqueeze(0).to(device)
                gt = batch['gt'].unsqueeze(0).to(device)

                # Optional: resize to fixed size for fair benchmarking
                if args.resize > 0:
                    lq = torch.nn.functional.interpolate(
                        lq, size=(args.resize, args.resize), mode='bilinear', align_corners=False
                    )
                    gt = torch.nn.functional.interpolate(
                        gt, size=(args.resize, args.resize), mode='bilinear', align_corners=False
                    )

                # Inference with timing
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start = time.time()
                with torch.no_grad():
                    enhanced = model.sample(lq, num_steps=num_steps, solver=solver)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.time() - start
                total_time += elapsed

                # Convert to numpy
                pred_np = (enhanced[0].cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                gt_np = (gt[0].cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)

                metrics = compute_metrics(pred_np, gt_np)
                for m in metric_names:
                    totals[m] += metrics[m]
                count += 1

                # Accumulate features for FID
                fid_calc.update_real(gt_np)
                fid_calc.update_fake(pred_np)

                img_name = os.path.basename(batch['lq_path'])
                per_image_results.append({'image': img_name, 'time_ms': elapsed * 1000, **metrics})

                if (idx + 1) % 10 == 0:
                    print(f"  [{idx+1}/{len(dataset)}] PSNR: {metrics['psnr']:.2f}")

            # Averages
            avg = {m: totals[m] / max(count, 1) for m in metric_names}
            avg_time = (total_time / max(count, 1)) * 1000  # ms

            # FID computed across all images
            print("  Computing FID...")
            fid_score = fid_calc.compute()
            avg['fid'] = fid_score

            result = {
                'solver': solver,
                'steps': num_steps,
                'inference_time_ms': round(avg_time, 2),
                **{m: round(avg[m], 4) for m in all_metric_names},
            }
            all_results.append(result)

            print(f"\n  Average: PSNR={avg['psnr']:.2f} | SSIM={avg['ssim']:.4f} | "
                  f"LPIPS={avg['lpips']:.4f} | UCIQE={avg['uciqe']:.4f} | "
                  f"UIQM={avg['uiqm']:.4f} | FID={fid_score:.2f} | Time={avg_time:.1f}ms")

            # Save per-image results
            if args.save_images:
                img_dir = os.path.join(
                    os.path.dirname(args.output) or 'results',
                    f'{solver}_{num_steps}steps'
                )
                os.makedirs(img_dir, exist_ok=True)
                # Re-run to save (or save during first pass — simpler to reuse)

    # Save summary CSV
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        fieldnames = ['solver', 'steps', 'inference_time_ms'] + all_metric_names
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{'='*60}")
    print("Summary results:")
    print(f"{'Solver':<10} {'Steps':<6} {'Time(ms)':<10} {'PSNR':<8} {'SSIM':<8} "
          f"{'LPIPS':<8} {'UCIQE':<8} {'UIQM':<8} {'FID':<8}")
    print('-' * 76)
    for r in all_results:
        print(f"{r['solver']:<10} {r['steps']:<6} {r['inference_time_ms']:<10.1f} "
              f"{r['psnr']:<8.2f} {r['ssim']:<8.4f} {r['lpips']:<8.4f} "
              f"{r['uciqe']:<8.4f} {r['uiqm']:<8.4f} {r['fid']:<8.2f}")
    print(f"\nResults saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description='WaveFlow-UIE Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--dataset', type=str, default='uieb_test',
                        help='Dataset name preset (uieb_test, lsui_test) or use --input-dir/--target-dir')
    parser.add_argument('--input-dir', type=str, default=None, help='Custom input directory')
    parser.add_argument('--target-dir', type=str, default=None, help='Custom target directory')
    parser.add_argument('--steps', type=str, default='5', help='Comma-separated step counts (e.g., 1,2,5,10,20)')
    parser.add_argument('--solvers', type=str, default='euler', help='Comma-separated solvers (euler,midpoint)')
    parser.add_argument('--output', type=str, default='results/evaluation.csv', help='Output CSV path')
    parser.add_argument('--save-images', action='store_true', help='Save enhanced images')
    parser.add_argument('--resize', type=int, default=0,
                        help='Resize all images to NxN before inference (0 = keep original size). '
                             'Standard for benchmarking: 256.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
