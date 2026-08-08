"""HDF5 data loaders for the three VTLA training stages."""

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


IMAGE_SIZE = (256, 256)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

CAMERA_PATHS = {
    'cam_high': 'observation/head/rgb',
    'cam_wrist': 'observation/wrist/rgb',
}
TACTILE_PATHS = {
    'tac_left': 'tactile/left_tactile/rgb',
    'tac_right': 'tactile/right_tactile/rgb',
}


def decode_image(img_data: object) -> np.ndarray:
    """Decode either a JPEG byte string or an already-decoded HWC array."""
    is_encoded = isinstance(img_data, (bytes, np.bytes_))
    if isinstance(img_data, np.ndarray):
        is_encoded = is_encoded or img_data.dtype.kind in {'O', 'S'}

    if is_encoded:
        encoded = img_data.tobytes() if isinstance(img_data, np.ndarray) else bytes(img_data)
        image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("OpenCV failed to decode an encoded image")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = np.asarray(img_data)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an HWC RGB image, got shape {image.shape}")
    return image


def image_to_tensor(img_data: object, normalize: bool = True) -> torch.Tensor:
    """Decode and resize an image, returning a contiguous CHW float tensor."""
    image = cv2.resize(decode_image(img_data), IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
    if normalize:
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor


def _require_datasets(root: h5py.File, paths: Iterable[str], episode_path: Path) -> None:
    missing = [path for path in paths if path not in root]
    if missing:
        raise KeyError(f"{episode_path} is missing required datasets: {missing}")


class VTLADataset(Dataset):
    """One sample per trajectory timestep with a future joint-position chunk.

    The source files do not contain a dedicated robot action dataset. Until one
    is added, the next joint positions are used explicitly as an action proxy.
    """

    def __init__(
        self,
        dataset_dir: str,
        camera_names: Iterable[str],
        tactile_names: Iterable[str],
        chunk_size: int = 100,
        state_dim: Optional[int] = None,
        normalize_joints: bool = True,
        verbose: bool = True,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.camera_names = list(camera_names)
        self.tactile_names = list(tactile_names)
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.normalize_joints = normalize_joints

        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if not self.camera_names:
            raise ValueError("At least one camera must be configured")
        if not self.tactile_names:
            raise ValueError("At least one tactile sensor must be configured")

        hdf5_dir = self.dataset_dir / 'hdf5'
        if not hdf5_dir.exists():
            raise ValueError(f"HDF5 directory not found: {hdf5_dir}")

        self.episode_files = sorted(hdf5_dir.glob('*.hdf5'))
        if not self.episode_files:
            raise ValueError(f"No HDF5 files found in {hdf5_dir}")

        required_paths = [
            'embodiment/joint',
            *(CAMERA_PATHS.get(name, f'observation/{name}/rgb') for name in self.camera_names),
            *(TACTILE_PATHS.get(name, f'tactile/{name}/rgb') for name in self.tactile_names),
        ]
        self.samples = []
        all_joints = []

        for episode_path in self.episode_files:
            with h5py.File(episode_path, 'r') as root:
                _require_datasets(root, required_paths, episode_path)
                joints = np.asarray(root['embodiment/joint'], dtype=np.float32)
                if joints.ndim != 2:
                    raise ValueError(f"{episode_path}: joint data must be 2D, got {joints.shape}")
                if state_dim is not None and joints.shape[1] != state_dim:
                    raise ValueError(
                        f"{episode_path}: configured state_dim={state_dim}, "
                        f"but joint data has dimension {joints.shape[1]}"
                    )
                for path in required_paths[1:]:
                    if len(root[path]) != len(joints):
                        raise ValueError(
                            f"{episode_path}: {path} has {len(root[path])} frames, "
                            f"expected {len(joints)}"
                        )
                if len(joints) < 2:
                    continue
                all_joints.append(joints)
                self.samples.extend((episode_path, timestep) for timestep in range(len(joints) - 1))

        if not self.samples:
            raise ValueError("No trajectory contains enough frames to form a training sample")

        joint_values = np.concatenate(all_joints, axis=0)
        self.joint_mean = joint_values.mean(axis=0).astype(np.float32)
        self.joint_std = np.maximum(joint_values.std(axis=0), 1e-2).astype(np.float32)

        if verbose:
            print(
                f"Found {len(self.episode_files)} episodes and "
                f"{len(self.samples)} timestep samples"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def get_stats(self) -> Dict[str, np.ndarray]:
        return {
            'joint_mean': self.joint_mean.copy(),
            'joint_std': self.joint_std.copy(),
            'action_source': 'next embodiment/joint positions',
        }

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        if not self.normalize_joints:
            return values.astype(np.float32, copy=False)
        return ((values - self.joint_mean) / self.joint_std).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode_path, timestep = self.samples[idx]

        with h5py.File(episode_path, 'r') as root:
            joints = root['embodiment/joint']
            qpos = self._normalize(np.asarray(joints[timestep], dtype=np.float32))

            cam_images = []
            for name in self.camera_names:
                path = CAMERA_PATHS.get(name, f'observation/{name}/rgb')
                cam_images.append(image_to_tensor(root[path][timestep]))

            tac_images = []
            for name in self.tactile_names:
                path = TACTILE_PATHS.get(name, f'tactile/{name}/rgb')
                tac_images.append(image_to_tensor(root[path][timestep]))

            action_end = min(timestep + 1 + self.chunk_size, len(joints))
            future_joints = np.asarray(joints[timestep + 1:action_end], dtype=np.float32)
            future_joints = self._normalize(future_joints)

        valid_steps = len(future_joints)
        action_dim = future_joints.shape[1]
        actions = np.zeros((self.chunk_size, action_dim), dtype=np.float32)
        actions[:valid_steps] = future_joints
        is_pad = np.ones(self.chunk_size, dtype=bool)
        is_pad[:valid_steps] = False

        return {
            'qpos': torch.from_numpy(qpos),
            'cam_image': torch.stack(cam_images),
            'tac_image': torch.stack(tac_images),
            'actions': torch.from_numpy(actions),
            'is_pad': torch.from_numpy(is_pad),
        }


class TactilePretrainDataset(Dataset):
    """All available frames from all configured tactile sensors."""

    def __init__(self, dataset_dir: str, tactile_names: Iterable[str], verbose: bool = True):
        self.dataset_dir = Path(dataset_dir)
        self.tactile_names = list(tactile_names)
        hdf5_dir = self.dataset_dir / 'hdf5'
        self.episode_files = sorted(hdf5_dir.glob('*.hdf5'))
        if not self.episode_files:
            raise ValueError(f"No HDF5 files found in {hdf5_dir}")

        self.samples = []
        for episode_path in self.episode_files:
            with h5py.File(episode_path, 'r') as root:
                for name in self.tactile_names:
                    rgb_path = TACTILE_PATHS.get(name, f'tactile/{name}/rgb')
                    _require_datasets(root, [rgb_path], episode_path)
                    self.samples.extend(
                        (episode_path, name, timestep) for timestep in range(len(root[rgb_path]))
                    )

        if verbose:
            print(
                f"Found {len(self.episode_files)} episodes and "
                f"{len(self.samples)} tactile frames for pretraining"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        episode_path, tactile_name, timestep = self.samples[idx]
        rgb_path = TACTILE_PATHS.get(tactile_name, f'tactile/{tactile_name}/rgb')
        base_path = rgb_path.rsplit('/rgb', 1)[0]

        with h5py.File(episode_path, 'r') as root:
            raw_image = root[rgb_path][timestep]
            tactile_image = image_to_tensor(raw_image, normalize=True)
            rgb_target = image_to_tensor(raw_image, normalize=False)

            marker_path = f'{base_path}/marker'
            if marker_path in root:
                marker_data = np.asarray(root[marker_path][timestep], dtype=np.float32)
                if marker_data.ndim == 3:
                    marker_data = marker_data[0]
                marker = marker_data[:63] / np.array([320.0, 240.0], dtype=np.float32)
            else:
                marker = np.zeros((63, 2), dtype=np.float32)

            pose_path = f'{base_path}/pose'
            pose = (
                np.asarray(root[pose_path][timestep], dtype=np.float32)
                if pose_path in root else np.zeros(7, dtype=np.float32)
            )

            depth_path = f'{base_path}/depth'
            if depth_path in root:
                depth = np.asarray(root[depth_path][timestep], dtype=np.float32)
                depth = np.clip((depth - 24.0) / 10.0, 0.0, 1.0)[None]
            else:
                depth = np.zeros((1, 240, 320), dtype=np.float32)

        return {
            'tactile_image': tactile_image,
            'targets': {
                'rgb': rgb_target,
                'marker': torch.from_numpy(marker),
                'pose': torch.from_numpy(pose),
                'depth': torch.from_numpy(depth),
            },
        }


def create_dataloader(args, stage: str = 'stage2') -> DataLoader:
    """Create a single-process training data loader."""
    if stage == 'stage1':
        dataset = TactilePretrainDataset(args.dataset_dir, args.tactile_names)
    else:
        dataset = VTLADataset(
            args.dataset_dir,
            args.camera_names,
            args.tactile_names,
            args.chunk_size,
            state_dim=args.state_dim,
        )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
