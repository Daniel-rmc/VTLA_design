"""Deploy a Dual-Stream VTLA checkpoint through UniVTAC's evaluation API."""

from __future__ import annotations

import os
import sys
from pathlib import Path

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


def _image_tensor(image: torch.Tensor, device: torch.device, normalize: bool = True) -> torch.Tensor:
    image = torch.as_tensor(image)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image.squeeze(0)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Expected image with shape [H, W, C], got {tuple(image.shape)}")
    image = image[..., :3].to(device=device, dtype=torch.float32)
    if image.max() > 2.0:
        image = image / 255.0
    image = image.permute(2, 0, 1).unsqueeze(0)
    image = torch.nn.functional.interpolate(
        image,
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    if normalize:
        mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device, dtype=torch.float32).view(3, 1, 1)
        image = (image - mean) / std
    return image


def _pick(observation: dict, keys: tuple[str, ...] | list[str] | str):
    if isinstance(keys, str):
        keys = (keys,)
    for key in keys:
        if key in observation:
            return observation[key]
    raise KeyError(f"None of observation keys {keys!r} were found; available={sorted(observation.keys())}")


def _resolve_tactile(observation: dict, tactile_name: str):
    tac_obs = observation["tactile"]
    for key in TACTILE_KEYS.get(tactile_name, (tactile_name,)):
        if key in tac_obs:
            value = tac_obs[key]
            if isinstance(value, dict):
                return value.get("rgb", value.get("rgb_marker"))
            return value
    raise KeyError(f"Could not resolve tactile sensor {tactile_name!r}; available={sorted(tac_obs.keys())}")


def _to_univtac_qpos(action: torch.Tensor) -> torch.Tensor:
    if action.numel() == 8:
        return action
    if action.numel() == 9:
        return torch.cat([action[:7], action[-1:]], dim=0)
    raise ValueError(f"Expected 8D or 9D action, got shape {tuple(action.shape)}")

def _save_rgb(path: Path, image_chw: torch.Tensor, normalized: bool) -> None:
    image = image_chw.detach().float().cpu()
    if normalized:
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        image = image * std + mean
    image = image.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    bgr = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


class Policy(BasePolicy):
    def __init__(self, args):
        super().__init__(args)

        project_dir = Path(args.get("project_dir", "/home/rmc/workspace/VTLA_design")).resolve()
        checkpoint_path = Path(args["checkpoint"]).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"DualStream checkpoint not found: {checkpoint_path}")

        project_str = str(project_dir)
        if project_str not in sys.path:
            sys.path.insert(0, project_str)

        from models.dual_stream import DualStreamVTLAPolicy

        requested_device = args.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(requested_device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("The DualStream UniVTAC adapter requested CUDA, but CUDA is unavailable")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        training = dict(checkpoint.get("args", {}))
        if not training:
            raise KeyError("Checkpoint does not contain training args")
        training.update(args.get("policy_config", {}))
        training["device"] = str(self.device)

        self.model = DualStreamVTLAPolicy(training).to(self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
        if state_dict is None:
            raise KeyError("Checkpoint has neither model_state_dict nor state_dict")
        loading_status = self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded DualStream policy weights from {checkpoint_path}")
        print(f"Loading status: {loading_status}")
        self.model.eval()

        stats = checkpoint.get("dataset_stats", {})
        self.qpos_mean = torch.as_tensor(stats.get("qpos_mean", 0.0), dtype=torch.float32, device=self.device)
        self.qpos_std = torch.as_tensor(stats.get("qpos_std", 1.0), dtype=torch.float32, device=self.device)
        self.action_mean = torch.as_tensor(stats.get("action_mean", 0.0), dtype=torch.float32, device=self.device)
        self.action_std = torch.as_tensor(stats.get("action_std", 1.0), dtype=torch.float32, device=self.device)

        self.camera_names = list(training.get("camera_names", ["cam_high"]))
        self.tactile_names = list(training.get("tactile_names", ["tac_left", "tac_right"]))
        self.state_dim = int(training.get("state_dim", 9))
        self.chunk_size = int(training.get("chunk_size", 50))
        self.temporal_agg = bool(args.get("temporal_agg", True))
        self.action_step = int(args.get("action_step", 0))
        self.normalized_action_clip = args.get("normalized_action_clip", 5.0)
        self.deterministic_latent = bool(args.get("use_deterministic_latent", True))
        self.normalize_tactile = bool(training.get("normalize_tactile", True))
        self.max_timesteps = int(args.get("max_timesteps", 3000))
        self.action_history: list[torch.Tensor] = []
        self.timestep = 0
        dump_dir = os.environ.get("DUALSTREAM_DUMP_INPUTS")
        self.dump_inputs_dir = Path(dump_dir).expanduser().resolve() if dump_dir else None
        if self.dump_inputs_dir is not None:
            self.dump_inputs_dir.mkdir(parents=True, exist_ok=True)

    def _encode_observation(self, observation: dict):
        qpos = torch.as_tensor(
            observation["embodiment"]["joint"],
            dtype=torch.float32,
            device=self.device,
        )
        qpos = qpos[: self.state_dim]
        qpos = (qpos - self.qpos_mean) / self.qpos_std
        qpos = qpos.unsqueeze(0)

        cam_images = []
        camera_obs = observation["observation"]
        for camera_name in self.camera_names:
            camera_key = CAMERA_KEYS.get(camera_name, camera_name)
            cam_images.append(_image_tensor(camera_obs[camera_key]["rgb"], self.device))
        cam_image = torch.stack(cam_images, dim=0).unsqueeze(0)

        tac_images = []
        for tactile_name in self.tactile_names:
            tac_images.append(
                _image_tensor(
                    _resolve_tactile(observation, tactile_name),
                    self.device,
                    normalize=self.normalize_tactile,
                )
            )
        tac_image = torch.stack(tac_images, dim=0).unsqueeze(0)
        if self.dump_inputs_dir is not None and self.timestep == 0:
            _save_rgb(self.dump_inputs_dir / "cam_high_input.png", cam_image[0, 0], normalized=True)
            for index, tactile_name in enumerate(self.tactile_names):
                _save_rgb(
                    self.dump_inputs_dir / f"{tactile_name}_input.png",
                    tac_image[0, index],
                    normalized=self.normalize_tactile,
                )
        return qpos, cam_image, tac_image

    def eval(self, task, observation):
        qpos, cam_image, tac_image = self._encode_observation(observation)
        with torch.inference_mode():
            actions = self.model(
                qpos,
                cam_image,
                tac_image,
                deterministic_latent=self.deterministic_latent,
            )
            actions = actions[0]

            if self.temporal_agg:
                self.action_history.append(actions.detach())
                valid_actions = []
                for past_timestep, past_actions in enumerate(self.action_history):
                    action_index = self.timestep - past_timestep
                    if 0 <= action_index < past_actions.shape[0]:
                        valid_actions.append(past_actions[action_index])
                if not valid_actions:
                    normalized_action = actions[min(self.action_step, actions.shape[0] - 1)]
                else:
                    stacked = torch.stack(valid_actions, dim=0)
                    weights = torch.exp(-torch.arange(stacked.shape[0], device=self.device, dtype=torch.float32))
                    weights = weights / weights.sum()
                    normalized_action = (stacked * weights[:, None]).sum(dim=0)
            else:
                normalized_action = actions[min(self.action_step, actions.shape[0] - 1)]

            if self.normalized_action_clip is not None:
                normalized_action = normalized_action.clamp(
                    -float(self.normalized_action_clip), float(self.normalized_action_clip)
                )
            action = normalized_action * self.action_std + self.action_mean

        action = _to_univtac_qpos(action.clone())
        action[-1] = action[-1].clamp(0.0, 0.039)
        task.take_action(action.to(task.device), action_type="qpos")
        self.timestep += 1

    def reset(self):
        self.action_history = []
        self.timestep = 0
        return None
