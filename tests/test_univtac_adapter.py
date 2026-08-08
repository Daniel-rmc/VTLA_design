import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).parents[1]
UNIVTAC_ROOT = Path("/home/rmc/workspace/UniVTAC")
ADAPTER_ROOT = PROJECT_ROOT / "univtac_adapter"
for path in (str(ADAPTER_ROOT), str(UNIVTAC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from policy.VTLA.deploy_policy import _image_tensor, _to_univtac_qpos


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
