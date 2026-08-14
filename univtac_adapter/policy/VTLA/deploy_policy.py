"""Deploy a VTLA checkpoint through UniVTAC's unified evaluation API."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import cv2
import numpy as np
import torch

from policy._base_policy import BasePolicy


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CAMERA_KEYS = {
    "cam_high": "head",
    "cam_wrist": "wrist",
}
TACTILE_KEYS = {
    "tac_left": ("left_tactile", "left_gsmini"),
    "tac_right": ("right_tactile", "right_gsmini"),
}


def _image_tensor(
    image: torch.Tensor,
    device: torch.device,
    normalize: bool = True,
) -> torch.Tensor:
    """Match the resize and optional normalization used by VTLADataset."""
    image = torch.as_tensor(image)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image.squeeze(0)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Expected an HWC RGB image, got {tuple(image.shape)}")
    image_array = np.ascontiguousarray(image[..., :3].detach().cpu().numpy())
    image_array = cv2.resize(image_array, (256, 256), interpolation=cv2.INTER_AREA)
    image = (
        torch.from_numpy(np.ascontiguousarray(image_array))
        .permute(2, 0, 1)
        .to(device=device, dtype=torch.float32)
        / 255.0
    )
    if normalize:
        mean = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = image.new_tensor(IMAGENET_STD).view(3, 1, 1)
        image = (image - mean) / std
    return image


def _resolve_tactile(observation: Mapping, name: str) -> torch.Tensor:
    tactile = observation["tactile"]
    for sensor_key in TACTILE_KEYS[name]:
        if sensor_key in tactile:
            # Training consumed raw tactile RGB, not marker-overlay RGB.
            return tactile[sensor_key]["rgb"]
    raise KeyError(f"Could not find UniVTAC tactile sensor for {name}: {tuple(tactile)}")


def _to_univtac_qpos(action: torch.Tensor) -> torch.Tensor:
    """Keep native 8D actions and support historical 9D checkpoints."""
    if action.numel() == 8:
        return action
    if action.numel() != 9:
        raise ValueError(f"Expected an 8D or 9D qpos action, got {action.numel()}D")
    return torch.cat((action[:7], action[7:8]))


class Policy(BasePolicy):
    """VTLA adapter implementing UniVTAC's ``Policy`` contract."""

    def __init__(self, args: Mapping):
        project_dir = Path(args.get("project_dir", Path(__file__).parents[3])).resolve()
        checkpoint_path = Path(args["checkpoint"]).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"VTLA checkpoint not found: {checkpoint_path}")

        project_str = str(project_dir)
        if project_str not in sys.path:
            sys.path.insert(0, project_str)

        from models.vtla import VTLAPolicy

        requested_device = args.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(requested_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("The VTLA UniVTAC adapter requested CUDA, but CUDA is unavailable")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        run_config = checkpoint.get("run_config", {})
        training = dict(run_config.get("training", {}))
        if not training:
            raise KeyError("Checkpoint does not contain run_config.training")

        # All backbone weights are restored strictly below. Avoid an unnecessary
        # tactile ImageNet download when reconstructing the architecture.
        training["pretrained_backbones"] = False
        model_args = SimpleNamespace(**training)
        stage = args.get("stage", training.get("stage", "stage2"))
        self.model = VTLAPolicy(model_args, stage=stage)

        state_dict = checkpoint["model_state_dict"]
        if state_dict and next(iter(state_dict)).startswith("module."):
            state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device).eval()

        stats = checkpoint.get("dataset_stats") or run_config.get("dataset_stats")
        if not stats:
            raise KeyError("Checkpoint does not contain dataset normalization statistics")
        self.qpos_mean = torch.as_tensor(
            stats.get("qpos_mean", stats["joint_mean"]),
            dtype=torch.float32,
            device=self.device,
        )
        self.qpos_std = torch.as_tensor(
            stats.get("qpos_std", stats["joint_std"]),
            dtype=torch.float32,
            device=self.device,
        )
        self.action_mean = torch.as_tensor(
            stats.get("action_mean", stats["joint_mean"]),
            dtype=torch.float32,
            device=self.device,
        )
        self.action_std = torch.as_tensor(
            stats.get("action_std", stats["joint_std"]),
            dtype=torch.float32,
            device=self.device,
        )
        self.joint_mean = self.qpos_mean
        self.joint_std = self.qpos_std
        self.camera_names = list(training["camera_names"])
        self.tactile_names = list(training["tactile_names"])
        self.normalize_tactile = bool(training.get("normalize_tactile", True))
        self.state_dim = int(training["state_dim"])
        self.joint_indices = [
            int(index) for index in stats.get("joint_indices", range(self.state_dim))
        ]
        self.action_step = int(args.get("action_step", 0))
        self.temporal_agg = bool(args.get("temporal_agg", training.get("temporal_agg", False)))
        self.temporal_agg_k = float(args.get("temporal_agg_k", 0.01))
        self.action_history = []
        self.timestep = 0
        self.normalized_action_clip = float(args.get("normalized_action_clip", 5.0))
        self.task_name = str(args.get("task_name", "unknown"))

        if self.qpos_mean.numel() != self.state_dim or self.action_mean.numel() != self.state_dim:
            raise ValueError(
                f"Checkpoint stats have qpos={self.qpos_mean.numel()} and "
                f"action={self.action_mean.numel()} dimensions, "
                f"but the model expects {self.state_dim}"
            )
        if len(self.joint_indices) != self.state_dim:
            raise ValueError(
                f"Checkpoint selects {len(self.joint_indices)} raw joint columns, "
                f"but the model expects {self.state_dim}"
            )
        if not 0 <= self.action_step < int(training["chunk_size"]):
            raise ValueError(f"action_step is outside the predicted action chunk: {self.action_step}")
        unknown_cameras = set(self.camera_names) - set(CAMERA_KEYS)
        unknown_tactile = set(self.tactile_names) - set(TACTILE_KEYS)
        if unknown_cameras or unknown_tactile:
            raise ValueError(
                f"Unsupported configured sensors: cameras={unknown_cameras}, tactile={unknown_tactile}"
            )

        print(
            f"Loaded VTLA epoch {checkpoint.get('epoch', 'unknown')} from {checkpoint_path}; "
            f"cameras={self.camera_names}, tactile={self.tactile_names}, stage={stage}, "
            f"control={self.state_dim}D indices={self.joint_indices}"
        )

    def encode_obs(self, observation: Mapping) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qpos = torch.as_tensor(
            observation["embodiment"]["joint"], dtype=torch.float32, device=self.device
        ).flatten()
        if not self.joint_indices or max(self.joint_indices) >= qpos.numel():
            raise ValueError(
                f"UniVTAC observation has {qpos.numel()} joint positions, "
                f"but VTLA selects raw columns {self.joint_indices}"
            )
        qpos = qpos[self.joint_indices]
        qpos_mean = getattr(self, 'qpos_mean', self.joint_mean)
        qpos_std = getattr(self, 'qpos_std', self.joint_std)
        qpos = ((qpos - qpos_mean) / qpos_std).unsqueeze(0)

        camera_images = [
            _image_tensor(
                observation["observation"][CAMERA_KEYS[name]]["rgb"], self.device
            )
            for name in self.camera_names
        ]
        tactile_images = [
            _image_tensor(
                _resolve_tactile(observation, name),
                self.device,
                normalize=bool(getattr(self, 'normalize_tactile', True)),
            )
            for name in self.tactile_names
        ]
        return qpos, torch.stack(camera_images).unsqueeze(0), torch.stack(tactile_images).unsqueeze(0)

    def eval(self, task, observation):
        # Keep inference mode strictly around VTLA. Wrapping task.take_action()
        # would cause IsaacLab state tensors to become inference tensors, which
        # cannot be updated in-place during the next environment reset.
        with torch.inference_mode():
            qpos, camera_images, tactile_images = self.encode_obs(observation)
            normalized_actions = self.model(qpos, camera_images, tactile_images)
            if getattr(self, 'temporal_agg', False):
                self.action_history.append((self.timestep, normalized_actions[0]))
                current_predictions = [
                    chunk[self.timestep - query_t]
                    for query_t, chunk in self.action_history
                    if 0 <= self.timestep - query_t < chunk.shape[0]
                ]
                stacked = torch.stack(current_predictions)
                weights = torch.exp(
                    -self.temporal_agg_k
                    * torch.arange(stacked.shape[0], device=self.device, dtype=stacked.dtype)
                )
                weights = weights / weights.sum()
                normalized_action = (stacked * weights.unsqueeze(1)).sum(0)
                self.action_history = [
                    (query_t, chunk)
                    for query_t, chunk in self.action_history
                    if self.timestep - query_t + 1 < chunk.shape[0]
                ]
            else:
                normalized_action = normalized_actions[0, self.action_step]
            normalized_action = normalized_action.clamp(
                -self.normalized_action_clip, self.normalized_action_clip
            )
            action_std = getattr(self, 'action_std', self.joint_std)
            action_mean = getattr(self, 'action_mean', self.joint_mean)
            action = normalized_action * action_std + action_mean

        # clone() outside inference mode produces a normal tensor for IsaacLab.
        action = action.clone()

        # New checkpoints already emit UniVTAC-native 8D commands. The helper
        # preserves deployment compatibility for historical 9D checkpoints.
        action = _to_univtac_qpos(action)
        action[-1] = action[-1].clamp(0.0, 0.039)
        task.take_action(action.to(task.device), action_type="qpos")
        self.timestep = getattr(self, 'timestep', 0) + 1

    def reset(self):
        self.action_history = []
        self.timestep = 0
        return None
