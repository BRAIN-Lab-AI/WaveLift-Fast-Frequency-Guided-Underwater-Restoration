"""Unit tests for model components."""

import torch
import pytest


def test_physics_prior_shapes():
    from waveflow_uie.models.physics_prior import PhysicsPriorNet

    net = PhysicsPriorNet(in_channels=3, hidden_channels=32, enabled=True)
    x = torch.randn(2, 3, 256, 256)
    physics_cond, t_map, a_map = net(x, target_size=(128, 128))

    assert physics_cond.shape == (2, 4, 128, 128)
    assert t_map.shape == (2, 1, 256, 256)
    assert a_map.shape == (2, 3, 256, 256)

    # Values should be in [0, 1] (sigmoid)
    assert t_map.min() >= 0.0 and t_map.max() <= 1.0
    assert a_map.min() >= 0.0 and a_map.max() <= 1.0


def test_physics_prior_disabled():
    from waveflow_uie.models.physics_prior import PhysicsPriorNet

    net = PhysicsPriorNet(enabled=False)
    x = torch.randn(2, 3, 256, 256)
    physics_cond, t_map, a_map = net(x, target_size=(128, 128))

    assert physics_cond.shape == (2, 4, 128, 128)
    assert physics_cond.abs().max() == 0.0


def test_physics_prior_param_count():
    from waveflow_uie.models.physics_prior import PhysicsPriorNet

    net = PhysicsPriorNet(in_channels=3, hidden_channels=32)
    total = sum(p.numel() for p in net.parameters())
    print(f"Physics prior params: {total:,}")
    assert total < 100_000, f"Expected <100K params, got {total}"


def test_velocity_unet_shapes():
    from waveflow_uie.models.velocity_unet import VelocityUNet

    model = VelocityUNet(
        in_channels=28, out_channels=12, base_channels=64,
        channel_mults=(1, 2, 4, 8), num_res_blocks=2,
        attention_resolutions=(16, 8),
    )
    x = torch.randn(1, 28, 128, 128)
    t = torch.rand(1)
    with torch.no_grad():
        out = model(x, t)
    assert out.shape == (1, 12, 128, 128)


def test_velocity_unet_param_range():
    from waveflow_uie.models.velocity_unet import VelocityUNet

    model = VelocityUNet()
    total = sum(p.numel() for p in model.parameters())
    print(f"UNet params: {total:,} ({total/1e6:.1f}M)")
    assert 30_000_000 < total < 100_000_000, f"Params out of expected range: {total}"


def test_velocity_unet_gradient_flow():
    from waveflow_uie.models.velocity_unet import VelocityUNet

    model = VelocityUNet(
        in_channels=28, out_channels=12, base_channels=32,
        channel_mults=(1, 2, 4), num_res_blocks=1,
        attention_resolutions=(8,),
    )
    x = torch.randn(1, 28, 32, 32)
    t = torch.rand(1)
    out = model(x, t)
    loss = out.sum()
    loss.backward()

    # All parameters should have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_waveflow_training_step():
    from waveflow_uie.models.waveflow import WaveFlowUIE

    model = WaveFlowUIE(
        unet_kwargs={
            'in_channels': 28, 'out_channels': 12, 'base_channels': 32,
            'channel_mults': (1, 2), 'num_res_blocks': 1,
            'attention_resolutions': (),
        },
    )
    lq = torch.randn(2, 3, 64, 64)
    gt = torch.randn(2, 3, 64, 64)
    outputs = model.training_step(lq, gt)

    assert outputs['v_pred'].shape == (2, 12, 32, 32)
    assert outputs['v_target'].shape == (2, 12, 32, 32)
    assert outputs['pred_rgb'].shape == (2, 3, 64, 64)
    assert outputs['physics_cond'].shape == (2, 4, 32, 32)
    assert outputs['t_map'].shape == (2, 1, 64, 64)
    assert outputs['a_map'].shape == (2, 3, 64, 64)


def test_waveflow_sample():
    from waveflow_uie.models.waveflow import WaveFlowUIE

    model = WaveFlowUIE(
        unet_kwargs={
            'in_channels': 28, 'out_channels': 12, 'base_channels': 32,
            'channel_mults': (1, 2), 'num_res_blocks': 1,
            'attention_resolutions': (),
        },
    )
    model.eval()
    lq = torch.randn(1, 3, 64, 64).clamp(0, 1)

    # Euler
    enhanced = model.sample(lq, num_steps=3, solver='euler')
    assert enhanced.shape == (1, 3, 64, 64)
    assert enhanced.min() >= 0.0 and enhanced.max() <= 1.0

    # Midpoint
    enhanced_mp = model.sample(lq, num_steps=2, solver='midpoint')
    assert enhanced_mp.shape == (1, 3, 64, 64)


def test_ema_helper():
    from waveflow_uie.utils.ema import EMAHelper

    model = torch.nn.Linear(10, 5)
    ema = EMAHelper(model, target_decay=0.999, warmup_decay=0.99, warmup_iters=100)

    # Warmup decay
    assert ema.get_decay(50) == 0.99
    # Target decay
    assert ema.get_decay(200) == 0.999

    # Update
    model.weight.data.fill_(1.0)
    ema.update(model, step=0)

    # Apply and restore
    backup = ema.apply(model)
    ema.restore(model, backup)
    assert torch.allclose(model.weight.data, torch.ones_like(model.weight.data))


def test_ema_state_dict():
    from waveflow_uie.utils.ema import EMAHelper

    model = torch.nn.Linear(10, 5)
    ema = EMAHelper(model)

    # Save/load roundtrip
    state = ema.state_dict()
    ema2 = EMAHelper(model)
    ema2.load_state_dict(state, torch.device('cpu'))
    assert ema2.target_decay == ema.target_decay


def test_seed_reproducibility():
    from waveflow_uie.utils.seed import set_seed

    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    assert torch.equal(a, b)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
