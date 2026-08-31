from __future__ import annotations

import pytest
import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.optim.schedulers import load_scheduler_state, save_scheduler_state

from lerobot_policy_vtla import (
    VTLAConfig,
    VTLAPolicy,
    WarmupStableCosineDecaySchedulerConfig,
)
from scripts.training.validate_checkpoint import prepare_dataset_sample_for_policy

IMAGE_NAMES = [
    "observation.images.front_camera",
    "observation.images.left_hand_camera",
    "observation.images.right_hand_camera",
    "observation.images.left_tactile_left_camera",
    "observation.images.left_tactile_right_camera",
    "observation.images.right_tactile_left_camera",
    "observation.images.right_tactile_right_camera",
]


def make_config() -> VTLAConfig:
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(28,)),
        **{
            name: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64))
            for name in IMAGE_NAMES
        },
    }
    return VTLAConfig(
        input_features=input_features,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(16,))},
        device="cpu",
        dtype="float32",
        chunk_size=4,
        n_action_steps=4,
        image_size=(64, 64),
        vision_backbone_weights=None,
        tactile_backbone_weights=None,
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_encoder_layers=1,
        n_decoder_layers=1,
        cross_attention_layers=1,
        n_vae_encoder_layers=1,
        temporal_ensemble_coeff=None,
    )


def make_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    batch = {
        "observation.state": torch.randn(batch_size, 28),
        "action": torch.randn(batch_size, 4, 16),
        "action_is_pad": torch.tensor([[False, False, False, False], [False, False, True, True]]),
    }
    batch.update({name: torch.randn(batch_size, 3, 64, 64) for name in IMAGE_NAMES})
    return batch


def test_policy_supports_distinct_state_and_action_dimensions() -> None:
    policy = VTLAPolicy(make_config())
    batch = make_batch()
    policy.train()
    loss, metrics = policy(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics.keys() >= {"l1_loss", "kld_loss", "contact_probability"}
    assert any(parameter.grad is not None for parameter in policy.parameters())


def test_policy_select_action_returns_one_unnormalized_shape() -> None:
    policy = VTLAPolicy(make_config())
    observation = make_batch()
    observation.pop("action")
    observation.pop("action_is_pad")
    action = policy.select_action(observation)
    assert action.shape == (2, 16)
    assert torch.isfinite(action).all()


def test_production_lr_schedule_boundaries() -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=1e-5)
    config = WarmupStableCosineDecaySchedulerConfig()
    scheduler = config.build(optimizer, num_training_steps=30_000)

    def lr_at(step: int) -> float:
        return 1e-5 * scheduler.lr_lambdas[0](step)

    assert lr_at(0) == pytest.approx(1e-5 / 1_001)
    assert lr_at(1_000) == pytest.approx(1e-5)
    assert lr_at(27_000) == pytest.approx(1e-5)
    assert lr_at(28_500) == pytest.approx(5.5e-6)
    assert lr_at(30_000) == pytest.approx(1e-6)


def test_lr_schedule_scales_for_short_runs_and_resumes(tmp_path) -> None:
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=1e-5)
    config = WarmupStableCosineDecaySchedulerConfig()
    scheduler = config.build(optimizer, num_training_steps=10)

    assert config.scaled_phase_steps(10) == (0, 9)
    assert config.scaled_phase_steps(1) == (0, 0)
    assert scheduler.lr_lambdas[0](0) == pytest.approx(1.0)
    assert scheduler.lr_lambdas[0](9) == pytest.approx(1.0)
    assert scheduler.lr_lambdas[0](10) == pytest.approx(0.1)

    optimizer.step()
    scheduler.step()
    optimizer.step()
    scheduler.step()
    save_scheduler_state(scheduler, tmp_path)

    restored_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(()))], lr=1e-5)
    restored = config.build(restored_optimizer, num_training_steps=10)
    load_scheduler_state(restored, tmp_path)
    assert restored.last_epoch == scheduler.last_epoch
    assert restored.get_last_lr() == pytest.approx(scheduler.get_last_lr())


def test_uint8_camera_conversion_preserves_dynamic_range() -> None:
    camera_key = IMAGE_NAMES[0]
    sample = {
        camera_key: torch.tensor([[[0, 128, 255]]], dtype=torch.uint8),
        "observation.state": torch.zeros(28),
    }
    prepared = prepare_dataset_sample_for_policy(sample, [camera_key])

    assert prepared[camera_key].dtype == torch.float32
    assert prepared[camera_key].min().item() == pytest.approx(0.0)
    assert prepared[camera_key].max().item() == pytest.approx(1.0)
    assert (prepared[camera_key].max() - prepared[camera_key].min()).item() > 0
