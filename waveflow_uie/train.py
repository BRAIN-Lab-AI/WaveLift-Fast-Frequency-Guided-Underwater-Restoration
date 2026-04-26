"""WaveFlow-UIE training loop.

Entry point: python -m waveflow_uie.train --config configs/waveflow_uie_uieb.yaml [--seed 42]

Features:
  - FP16 mixed precision with gradient scaling
  - Gradient accumulation (effective batch = batch_size * grad_accum)
  - EMA with warmup (0.99 -> 0.999)
  - Linear LR warmup + CosineAnnealingLR
  - Gradient clipping
  - TensorBoard logging
  - Checkpoint save/resume
  - NaN recovery (reload checkpoint + reduce LR)
  - Validation with PSNR/SSIM
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from waveflow_uie.data.dataset import PairedUIEDataset
from waveflow_uie.losses import WaveFlowLoss
from waveflow_uie.models.waveflow import WaveFlowUIE
from waveflow_uie.utils.ema import EMAHelper
from waveflow_uie.utils.seed import set_seed


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_model(config: dict) -> WaveFlowUIE:
    """Build WaveFlow-UIE model from config."""
    model_cfg = config.get('model', {})
    return WaveFlowUIE(
        unet_kwargs=model_cfg.get('velocity_unet', {}),
        physics_kwargs=model_cfg.get('physics_prior', {}),
    )


def build_loss(config: dict, device: torch.device) -> WaveFlowLoss:
    """Build composite loss from config."""
    loss_cfg = config.get('training', {}).get('losses', {})
    physics_cfg = config.get('model', {}).get('physics_prior', {})
    return WaveFlowLoss(
        cfm_weight=loss_cfg.get('cfm_weight', 1.0),
        freq_weight=loss_cfg.get('freq_weight', 0.5),
        lpips_weight=loss_cfg.get('lpips_weight', 0.1),
        lab_weight=loss_cfg.get('lab_weight', 0.05),
        physics_weight=loss_cfg.get('physics_weight', 0.1),
        freq_hf_weight=loss_cfg.get('freq_hf_weight', 2.0),
        physics_enabled=physics_cfg.get('enabled', True),
    ).to(device)


def save_checkpoint(
    path: str,
    model: WaveFlowUIE,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    ema: EMAHelper,
    step: int,
    best_psnr: float,
) -> None:
    """Save training checkpoint."""
    torch.save({
        'step': step,
        'best_psnr': best_psnr,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'ema_state_dict': ema.state_dict(),
    }, path)


def load_checkpoint(
    path: str,
    model: WaveFlowUIE,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    ema: EMAHelper,
    device: torch.device,
) -> tuple:
    """Load training checkpoint. Returns (step, best_psnr)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    scaler.load_state_dict(ckpt['scaler_state_dict'])
    ema.load_state_dict(ckpt['ema_state_dict'], device)
    return ckpt['step'], ckpt.get('best_psnr', 0.0)


def find_latest_checkpoint(ckpt_dir: str) -> str:
    """Find the most recent checkpoint in a directory."""
    ckpts = sorted(Path(ckpt_dir).glob('checkpoint_*.pth'))
    return str(ckpts[-1]) if ckpts else None


from waveflow_uie.utils.metrics import calculate_psnr as _calculate_psnr
from waveflow_uie.utils.metrics import calculate_ssim as _calculate_ssim


@torch.no_grad()
def validate(
    model: WaveFlowUIE,
    val_loader: DataLoader,
    device: torch.device,
    num_steps: int = 5,
) -> dict:
    """Run validation and compute PSNR/SSIM."""
    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    for batch in val_loader:
        lq = batch['lq'].to(device)
        gt = batch['gt'].to(device)

        enhanced = model.sample(lq, num_steps=num_steps, solver='euler')

        # Convert to numpy [0, 255] uint8 HWC for metric computation
        for i in range(lq.shape[0]):
            pred_np = (enhanced[i].cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            gt_np = (gt[i].cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)

            total_psnr += _calculate_psnr(pred_np, gt_np, crop_border=2, test_y_channel=True, is_bgr=False)
            total_ssim += _calculate_ssim(pred_np, gt_np, crop_border=2, test_y_channel=True, is_bgr=False)
            count += 1

    model.train()
    return {
        'psnr': total_psnr / max(count, 1),
        'ssim': total_ssim / max(count, 1),
    }


def train(config: dict, seed: int = 42, resume: str = None) -> None:
    """Main training function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

    train_cfg = config.get('training', {})
    data_cfg = config.get('data', {})
    val_cfg = config.get('validation', {})
    log_cfg = config.get('logging', {})
    exp_name = config.get('experiment', {}).get('name', 'waveflow_uie')

    total_iters = train_cfg.get('total_iters', 200000)
    grad_accum = train_cfg.get('grad_accum_steps', 2)
    warmup_iters = train_cfg.get('warmup_iters', 5000)
    grad_clip = train_cfg.get('grad_clip', 1.0)
    use_fp16 = train_cfg.get('fp16', True)

    print_freq = log_cfg.get('print_freq', 100)
    save_freq = log_cfg.get('save_checkpoint_freq', 10000)
    sample_freq = log_cfg.get('sample_freq', 1000)
    val_freq_early = val_cfg.get('val_freq_early', 1000)
    val_freq_late = val_cfg.get('val_freq_late', 5000)
    val_freq_threshold = val_cfg.get('val_freq_threshold', 20000)

    # Directories
    exp_dir = os.path.join('experiments', exp_name)
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    tb_dir = os.path.join(exp_dir, 'tb_logger')
    sample_dir = os.path.join(exp_dir, 'samples')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(tb_dir)

    # Dataset
    train_dataset = PairedUIEDataset(
        input_dir=data_cfg['train_input'],
        target_dir=data_cfg['train_target'],
        patch_size=data_cfg.get('patch_size', 256),
        is_train=True,
        enlarge_ratio=data_cfg.get('dataset_enlarge_ratio', 10),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.get('batch_size', 4),
        shuffle=True,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=True,
        drop_last=True,
    )

    val_dataset = PairedUIEDataset(
        input_dir=data_cfg['val_input'],
        target_dir=data_cfg['val_target'],
        patch_size=data_cfg.get('patch_size', 256),
        is_train=False,
        enlarge_ratio=1,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Model
    model = build_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Loss
    criterion = build_loss(config, device)

    # Optimizer
    opt_cfg = train_cfg.get('optimizer', {})
    betas = tuple(opt_cfg.get('betas', [0.9, 0.999]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg.get('lr', 2e-4),
        weight_decay=opt_cfg.get('weight_decay', 1e-4),
        betas=betas,
    )

    # Scheduler (applied after warmup)
    sched_cfg = train_cfg.get('scheduler', {})
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_iters - warmup_iters,
        eta_min=sched_cfg.get('eta_min', 1e-6),
    )

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)

    # EMA
    ema_cfg = train_cfg.get('ema', {})
    ema = EMAHelper(
        model.velocity_unet,
        target_decay=ema_cfg.get('target_decay', 0.999),
        warmup_decay=ema_cfg.get('warmup_decay', 0.99),
        warmup_iters=ema_cfg.get('warmup_iters', 1000),
    )

    # Resume
    global_step = 0
    best_psnr = 0.0
    base_lr = opt_cfg.get('lr', 2e-4)

    if resume:
        ckpt_path = resume
    else:
        ckpt_path = find_latest_checkpoint(ckpt_dir)

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Resuming from {ckpt_path}")
        global_step, best_psnr = load_checkpoint(
            ckpt_path, model, optimizer, scheduler, scaler, ema, device
        )
        print(f"  Resumed at step {global_step}, best PSNR: {best_psnr:.2f}")

    # Log config
    writer.add_text('config', yaml.dump(config), global_step)
    writer.add_text('seed', str(seed), global_step)

    print(f"Starting training from step {global_step} to {total_iters}")
    print(f"  Batch size: {data_cfg.get('batch_size', 4)} x {grad_accum} = "
          f"{data_cfg.get('batch_size', 4) * grad_accum} effective")
    print(f"  FP16: {use_fp16}, Grad clip: {grad_clip}")
    print(f"  Train images: {len(train_dataset) // train_dataset.enlarge_ratio}, "
          f"Val images: {len(val_dataset)}")

    # Training loop
    model.train()
    optimizer.zero_grad()
    data_iter = iter(train_loader)
    nan_recovery_count = 0
    max_nan_recoveries = 5

    while global_step < total_iters:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        lq = batch['lq'].to(device, non_blocking=True)
        gt = batch['gt'].to(device, non_blocking=True)

        # Learning rate warmup
        if global_step < warmup_iters:
            lr = base_lr * (global_step + 1) / warmup_iters
            for pg in optimizer.param_groups:
                pg['lr'] = lr

        # Forward pass
        with torch.amp.autocast('cuda', enabled=use_fp16):
            outputs = model.training_step(lq, gt)
            losses = criterion(outputs, lq, gt)
            loss = losses['total'] / grad_accum

        # NaN check
        if not torch.isfinite(loss):
            nan_recovery_count += 1
            print(f"\n[WARNING] NaN/Inf loss at step {global_step} "
                  f"(recovery {nan_recovery_count}/{max_nan_recoveries})")
            print(f"  lq stats: min={lq.min():.4f}, max={lq.max():.4f}, mean={lq.mean():.4f}")

            if nan_recovery_count > max_nan_recoveries:
                print("[ERROR] Too many NaN recoveries. Stopping.")
                break

            # Reload last checkpoint and reduce LR
            ckpt_path = find_latest_checkpoint(ckpt_dir)
            if ckpt_path:
                global_step, best_psnr = load_checkpoint(
                    ckpt_path, model, optimizer, scheduler, scaler, ema, device
                )
                base_lr *= 0.1
                print(f"  Reloaded checkpoint at step {global_step}, new base LR: {base_lr:.2e}")
            else:
                print("[ERROR] No checkpoint to recover from. Stopping.")
                break

            optimizer.zero_grad()
            model.train()
            continue

        # Backward
        scaler.scale(loss).backward()

        # Optimizer step (every grad_accum steps)
        if (global_step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # EMA update
            ema.update(model.velocity_unet, global_step)

            # Scheduler step (after warmup)
            if global_step >= warmup_iters:
                scheduler.step()

        # Logging
        if global_step % print_freq == 0:
            lr_current = optimizer.param_groups[0]['lr']
            loss_str = ' | '.join(f"{k}: {v.item():.4f}" for k, v in losses.items())
            print(f"[Step {global_step:>7d}/{total_iters}] {loss_str} | lr: {lr_current:.2e}")

            for k, v in losses.items():
                writer.add_scalar(f'loss/{k}', v.item(), global_step)
            writer.add_scalar('train/lr', lr_current, global_step)

            if (global_step + 1) % grad_accum == 0:
                writer.add_scalar('train/grad_norm', grad_norm.item(), global_step)

            # Log physics prior stats
            if outputs.get('t_map') is not None:
                t_mean = outputs['t_map'].mean().item()
                writer.add_scalar('physics/transmission_mean', t_mean, global_step)

        # Sample images
        if global_step > 0 and global_step % sample_freq == 0:
            model.eval()
            with torch.no_grad():
                sample_lq = lq[:2]
                sample_gt = gt[:2]
                sample_enhanced = model.sample(sample_lq, num_steps=5, solver='euler')

                # Log images to TensorBoard
                writer.add_images('samples/input', sample_lq.clamp(0, 1), global_step)
                writer.add_images('samples/enhanced', sample_enhanced.clamp(0, 1), global_step)
                writer.add_images('samples/ground_truth', sample_gt.clamp(0, 1), global_step)
            model.train()

        # Validation
        val_freq = val_freq_early if global_step < val_freq_threshold else val_freq_late
        if global_step > 0 and global_step % val_freq == 0:
            # Use EMA weights for validation
            backup = ema.apply(model.velocity_unet)
            metrics = validate(model, val_loader, device, num_steps=5)
            ema.restore(model.velocity_unet, backup)

            print(f"  [Val] PSNR: {metrics['psnr']:.2f} | SSIM: {metrics['ssim']:.4f}")
            writer.add_scalar('val/psnr', metrics['psnr'], global_step)
            writer.add_scalar('val/ssim', metrics['ssim'], global_step)

            if metrics['psnr'] > best_psnr:
                best_psnr = metrics['psnr']
                save_checkpoint(
                    os.path.join(ckpt_dir, 'best.pth'),
                    model, optimizer, scheduler, scaler, ema, global_step, best_psnr,
                )
                print(f"  New best PSNR: {best_psnr:.2f}")

        # Save checkpoint
        if global_step > 0 and global_step % save_freq == 0:
            save_checkpoint(
                os.path.join(ckpt_dir, f'checkpoint_{global_step:07d}.pth'),
                model, optimizer, scheduler, scaler, ema, global_step, best_psnr,
            )
            # Also save as latest
            save_checkpoint(
                os.path.join(ckpt_dir, 'latest.pth'),
                model, optimizer, scheduler, scaler, ema, global_step, best_psnr,
            )

        global_step += 1

    # Final save
    save_checkpoint(
        os.path.join(ckpt_dir, f'checkpoint_{global_step:07d}.pth'),
        model, optimizer, scheduler, scaler, ema, global_step, best_psnr,
    )
    writer.close()
    print(f"\nTraining complete. Best PSNR: {best_psnr:.2f}")


def main():
    parser = argparse.ArgumentParser(description='WaveFlow-UIE Training')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume from')
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, seed=args.seed, resume=args.resume)


if __name__ == '__main__':
    main()
