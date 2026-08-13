#!/usr/bin/env python3
"""
Smoke Test for Dual-Stream VTLA Training
快速测试训练流程是否正常
"""
import sys
import torch
from pathlib import Path

# 添加项目根目录到path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.dual_stream_vtla_policy import DualStreamVTLAPolicy, build_dual_stream_vtla_model


def test_model_creation():
    """测试模型创建"""
    print("="*60)
    print("Test 1: Model Creation")
    print("="*60)

    class Args:
        state_dim = 14
        chunk_size = 50
        camera_names = ['cam_high']
        tactile_names = ['tac_left', 'tac_right']
        hidden_dim = 512
        nheads = 8
        dim_feedforward = 2048
        enc_layers = 4
        dec_layers = 6
        dropout = 0.1
        pre_norm = False

        # 双流配置
        shared_encoder = True
        shared_decoder = False
        enable_cross_stream = False
        cross_stream_layers = []
        fusion_type = 'gated'
        use_contact_routing = False
        use_cvae = True
        latent_dim = 32

        # Backbone
        tactile_backbone = 'resnet34'
        tactile_latent_dim = 512
        pretrained_backbones = True
        backbone = 'resnet18'
        lr_backbone = 1e-5
        lr_vision_backbone = 1e-5  # 添加这个属性
        masks = False
        dilation = False
        position_embedding = 'sine'

        # 损失权重
        kl_weight = 10.0
        pad_weight = 1.0
        l1_reduction = 'valid_mean'
        aux_vision_weight = 0.0
        aux_tactile_weight = 0.0

    args = Args()

    try:
        # 直接传递args对象，不转换为dict
        model = build_dual_stream_vtla_model(args)
        print("✓ Model created successfully")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Total parameters: {total_params / 1e6:.2f}M")
        print(f"✓ Trainable parameters: {trainable_params / 1e6:.2f}M")

        return model
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward_pass(model):
    """测试前向传播"""
    print("\n" + "="*60)
    print("Test 2: Forward Pass")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    B = 2
    state_dim = 14
    chunk_size = 50

    # 准备输入
    qpos = torch.randn(B, state_dim).to(device)
    cam_image = torch.randn(B, 1, 3, 256, 256).to(device)  # 1 camera
    tac_image = torch.randn(B, 2, 3, 256, 256).to(device)  # 2 tactile sensors

    try:
        # 推理模式
        with torch.no_grad():
            actions_pred, is_pad_pred = model(qpos, cam_image, tac_image)

        print(f"✓ Inference successful")
        print(f"  Actions shape: {actions_pred.shape}")
        print(f"  Is_pad shape: {is_pad_pred.shape}")

        assert actions_pred.shape == (B, chunk_size, state_dim), "Action shape mismatch"
        assert is_pad_pred.shape == (B, chunk_size, 1), "Is_pad shape mismatch"
        print(f"✓ Output shapes correct")

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_training_pass(model):
    """测试训练模式前向传播"""
    print("\n" + "="*60)
    print("Test 3: Training Pass")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # 创建Policy包装器
    class Args:
        kl_weight = 10.0
        pad_weight = 1.0
        l1_reduction = 'valid_mean'
        aux_vision_weight = 0.0
        aux_tactile_weight = 0.0

    from models.dual_stream_vtla_policy import DualStreamVTLAPolicy

    # 临时包装
    policy = type('Policy', (), {'model': model, 'kl_weight': 10.0, 'pad_weight': 1.0,
                                   'l1_reduction': 'valid_mean', 'aux_vision_weight': 0.0,
                                   'aux_tactile_weight': 0.0})()
    policy.model = model

    B = 2
    state_dim = 14
    chunk_size = 50

    qpos = torch.randn(B, state_dim).to(device)
    cam_image = torch.randn(B, 1, 3, 256, 256).to(device)
    tac_image = torch.randn(B, 2, 3, 256, 256).to(device)
    actions = torch.randn(B, chunk_size, state_dim).to(device)
    is_pad = torch.zeros(B, chunk_size, dtype=torch.bool).to(device)

    try:
        # 训练模式
        model.train()
        actions_pred, is_pad_pred, (mu, logvar), components = model(
            qpos, cam_image, tac_image, actions, is_pad,
            return_components=True
        )

        print(f"✓ Training forward successful")
        print(f"  Actions shape: {actions_pred.shape}")
        print(f"  Latent mu shape: {mu.shape}")
        print(f"  Latent logvar shape: {logvar.shape}")

        # 检查components
        if 'fusion_weights' in components and components['fusion_weights'] is not None:
            print(f"  Fusion weights shape: {components['fusion_weights'].shape}")
            print(f"  Vision weight mean: {components['vision_weight'].mean():.3f}")
            print(f"  Tactile weight mean: {components['tactile_weight'].mean():.3f}")

        print(f"✓ Components returned correctly")

        # 测试反向传播
        from torch.nn import functional as F
        loss = F.l1_loss(actions_pred, actions)
        loss.backward()
        print(f"✓ Backward pass successful")

    except Exception as e:
        print(f"✗ Training pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_checkpoint_save_load():
    """测试checkpoint保存和加载"""
    print("\n" + "="*60)
    print("Test 4: Checkpoint Save/Load")
    print("="*60)

    import tempfile
    import os

    # 创建临时模型
    print("Creating model...")
    model = test_model_creation()
    if model is None:
        return False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # 保存checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, 'test_checkpoint.ckpt')

        try:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': 10,
            }
            torch.save(checkpoint, ckpt_path)
            print(f"✓ Checkpoint saved to {ckpt_path}")

            # 加载checkpoint
            loaded_ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(loaded_ckpt['model_state_dict'])
            print(f"✓ Checkpoint loaded successfully")
            print(f"  Epoch: {loaded_ckpt['epoch']}")

        except Exception as e:
            print(f"✗ Checkpoint save/load failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    return True


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("Dual-Stream VTLA Smoke Test")
    print("="*60 + "\n")

    # Test 1: 模型创建
    model = test_model_creation()
    if model is None:
        print("\n✗ Smoke test FAILED at model creation")
        return 1

    # Test 2: 推理前向传播
    if not test_forward_pass(model):
        print("\n✗ Smoke test FAILED at forward pass")
        return 1

    # Test 3: 训练前向传播
    if not test_training_pass(model):
        print("\n✗ Smoke test FAILED at training pass")
        return 1

    # Test 4: Checkpoint保存加载
    if not test_checkpoint_save_load():
        print("\n✗ Smoke test FAILED at checkpoint save/load")
        return 1

    # 所有测试通过
    print("\n" + "="*60)
    print("✓ All tests PASSED!")
    print("="*60)
    print("\nYou can now proceed with full training:")
    print("  ./scripts/training/start_dual_stream_training.sh insert_HDMI 0")
    print("="*60 + "\n")

    return 0


if __name__ == '__main__':
    exit(main())
