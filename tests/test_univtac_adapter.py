import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).parents[1]
UNIVTAC_ROOT = Path("/home/rmc/workspace/UniVTAC")
ADAPTER_ROOT = PROJECT_ROOT / "univtac_adapter"
for path in (str(ADAPTER_ROOT), str(UNIVTAC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from policy.VTLA.deploy_policy import Policy, _image_tensor, _to_univtac_qpos


def test_image_tensor_matches_imagenet_normalization():
    image = torch.zeros(12, 18, 3, dtype=torch.uint8)
    output = _image_tensor(image, torch.device("cpu"))
    expected = -torch.tensor([0.485, 0.456, 0.406]) / torch.tensor([0.229, 0.224, 0.225])
    assert output.shape == (3, 256, 256)
    assert torch.allclose(output[:, 0, 0], expected)


def test_nine_dimensional_action_maps_to_univtac_gripper_api():
    action = torch.arange(9, dtype=torch.float32)
    mapped = _to_univtac_qpos(action)
    assert mapped.shape == (8,)
    assert torch.equal(mapped[:7], action[:7])
    assert mapped[-1].item() == action[7].item()


def test_simulator_receives_a_normal_tensor_outside_inference_mode():
    class DummyModel:
        def __call__(self, qpos, cameras, tactile):
            assert torch.is_inference(qpos)
            return torch.zeros(1, 1, 9)

    class DummyTask:
        device = torch.device("cpu")

        def take_action(self, action, action_type):
            assert action_type == "qpos"
            assert not torch.is_inference(action)
            action[0] = 1.0

    policy = Policy.__new__(Policy)
    policy.model = DummyModel()
    policy.action_step = 0
    policy.normalized_action_clip = 5.0
    policy.joint_mean = torch.zeros(9)
    policy.joint_std = torch.ones(9)
    policy.encode_obs = lambda observation: (
        torch.zeros(1, 9),
        torch.zeros(1, 2, 3, 4, 4),
        torch.zeros(1, 2, 3, 4, 4),
    )
    policy.eval(DummyTask(), {})
