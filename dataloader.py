"""HDF5 data loaders for the three VTLA training stages."""

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

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
TACTILE_PATH_CANDIDATES = {
    'tac_left': ('tactile/left_tactile/rgb', 'tactile/left_gsmini/rgb'),
    'tac_right': ('tactile/right_tactile/rgb', 'tactile/right_gsmini/rgb'),
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


def resolve_tactile_path(root: h5py.File, name: str, episode_path: Path) -> str:
    """Resolve local-collection and published GelSight Mini HDF5 key variants."""
    candidates = TACTILE_PATH_CANDIDATES.get(
        name, (TACTILE_PATHS.get(name, f'tactile/{name}/rgb'),)
    )
    for candidate in candidates:
        if candidate in root:
            return candidate
    raise KeyError(
        f"{episode_path} has no tactile RGB for {name}; checked {list(candidates)}"
    )


def _episode_sort_key(path: Path) -> Tuple[int, object]:
    """Sort numeric UniVTAC episode names numerically, then other names lexically."""
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.name)


def discover_episode_files(dataset_dir: str | Path) -> list[Path]:
    """Find episodes in either collected ``hdf5/`` or published ``clean/`` layout."""
    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory not found: {root}")

    nested = root / 'hdf5'
    hdf5_dir = nested if nested.is_dir() else root
    episode_files = sorted(hdf5_dir.glob('*.hdf5'), key=_episode_sort_key)
    if not episode_files:
        raise ValueError(
            f"No HDF5 files found in {root} (checked {nested} and {root})"
        )
    return episode_files


def split_episode_files(
    episode_files: Sequence[Path],
    val_fraction: float,
    seed: int,
    strata: Optional[Mapping[Path, str]] = None,
) -> tuple[list[Path], list[Path]]:
    """Create a deterministic episode-level train/validation split."""
    files = list(episode_files)
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if val_fraction == 0.0:
        return files, []
    if len(files) < 2:
        raise ValueError("At least two episodes are required for a validation split")

    rng = np.random.default_rng(seed)
    if strata is None:
        groups = {'all': files}
    else:
        groups = {}
        for path in files:
            if path not in strata:
                raise ValueError(f"No split stratum was provided for {path}")
            groups.setdefault(strata[path], []).append(path)

    val_set = set()
    for group_files in groups.values():
        if len(group_files) < 2:
            raise ValueError("Every split stratum must contain at least two episodes")
        val_count = max(1, int(round(len(group_files) * val_fraction)))
        val_count = min(val_count, len(group_files) - 1)
        permutation = rng.permutation(len(group_files))
        val_set.update(group_files[index] for index in permutation[:val_count])
    train_files = [path for path in files if path not in val_set]
    val_files = [path for path in files if path in val_set]
    return train_files, val_files


def infer_grasp_classify_label(episode_path: str | Path) -> str:
    """Identify which prism was active from the first recorded actor poses."""
    path = Path(episode_path)
    with h5py.File(path, 'r') as root:
        required = ['actor/rough_prism', 'actor/plain_prism']
        _require_datasets(root, required, path)
        rough_y = float(root['actor/rough_prism'][0, 1])
        plain_y = float(root['actor/plain_prism'][0, 1])
    rough_active = abs(rough_y) < 0.5
    plain_active = abs(plain_y) < 0.5
    if rough_active == plain_active:
        raise ValueError(
            f"Could not infer active prism in {path}: rough_y={rough_y}, plain_y={plain_y}"
        )
    return 'rough' if rough_active else 'plain'


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
        joint_indices: Optional[Iterable[int]] = None,
        episode_files: Optional[Sequence[str | Path]] = None,
        normalization_stats: Optional[Mapping[str, object]] = None,
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

        self.episode_files = (
            [Path(path).expanduser().resolve() for path in episode_files]
            if episode_files is not None else discover_episode_files(self.dataset_dir)
        )
        if not self.episode_files:
            raise ValueError("The selected episode list is empty")

        if joint_indices is None:
            self.joint_indices = list(range(state_dim)) if state_dim is not None else None
        else:
            self.joint_indices = [int(index) for index in joint_indices]
            if not self.joint_indices:
                raise ValueError("joint_indices cannot be empty")
            if len(set(self.joint_indices)) != len(self.joint_indices):
                raise ValueError(f"joint_indices contain duplicates: {self.joint_indices}")
            if min(self.joint_indices) < 0:
                raise ValueError(f"joint_indices must be non-negative: {self.joint_indices}")
        if state_dim is not None and self.joint_indices is not None:
            if len(self.joint_indices) != state_dim:
                raise ValueError(
                    f"state_dim={state_dim} but {len(self.joint_indices)} joint indices were selected"
                )

        camera_paths = [
            CAMERA_PATHS.get(name, f'observation/{name}/rgb') for name in self.camera_names
        ]
        self.samples = []
        all_joints = []
        self.raw_joint_dim = None
        self.tactile_paths_by_episode = {}

        for episode_path in self.episode_files:
            if not episode_path.is_file():
                raise ValueError(f"Episode file not found: {episode_path}")
            with h5py.File(episode_path, 'r') as root:
                tactile_paths = {
                    name: resolve_tactile_path(root, name, episode_path)
                    for name in self.tactile_names
                }
                self.tactile_paths_by_episode[episode_path] = tactile_paths
                required_paths = [
                    'embodiment/joint', *camera_paths, *tactile_paths.values()
                ]
                _require_datasets(root, required_paths, episode_path)
                joints = np.asarray(root['embodiment/joint'], dtype=np.float32)
                if joints.ndim != 2:
                    raise ValueError(f"{episode_path}: joint data must be 2D, got {joints.shape}")
                if self.raw_joint_dim is None:
                    self.raw_joint_dim = int(joints.shape[1])
                    if self.joint_indices is None:
                        self.joint_indices = list(range(self.raw_joint_dim))
                        self.state_dim = self.raw_joint_dim
                elif joints.shape[1] != self.raw_joint_dim:
                    raise ValueError(
                        f"{episode_path}: raw joint dimension {joints.shape[1]} differs from "
                        f"the first episode dimension {self.raw_joint_dim}"
                    )
                if max(self.joint_indices) >= joints.shape[1]:
                    raise ValueError(
                        f"{episode_path}: selected joint indices {self.joint_indices} exceed "
                        f"raw dimension {joints.shape[1]}"
                    )
                for path in required_paths[1:]:
                    if len(root[path]) != len(joints):
                        raise ValueError(
                            f"{episode_path}: {path} has {len(root[path])} frames, "
                            f"expected {len(joints)}"
                        )
                if len(joints) < 2:
                    continue
                all_joints.append(joints[:, self.joint_indices])
                self.samples.extend((episode_path, timestep) for timestep in range(len(joints) - 1))

        if not self.samples:
            raise ValueError("No trajectory contains enough frames to form a training sample")

        if normalization_stats is None:
            joint_values = np.concatenate(all_joints, axis=0)
            self.joint_mean = joint_values.mean(axis=0).astype(np.float32)
            self.joint_std = np.maximum(joint_values.std(axis=0), 1e-2).astype(np.float32)
            self.normalization_source = 'selected episodes'
        else:
            self.joint_mean = np.asarray(normalization_stats['joint_mean'], dtype=np.float32)
            self.joint_std = np.asarray(normalization_stats['joint_std'], dtype=np.float32)
            expected_shape = (len(self.joint_indices),)
            if self.joint_mean.shape != expected_shape or self.joint_std.shape != expected_shape:
                raise ValueError(
                    f"Normalization stats must have shape {expected_shape}, got "
                    f"mean={self.joint_mean.shape}, std={self.joint_std.shape}"
                )
            if np.any(self.joint_std <= 0):
                raise ValueError("Normalization standard deviations must be positive")
            self.normalization_source = 'provided training statistics'

        if verbose:
            print(
                f"Found {len(self.episode_files)} episodes and "
                f"{len(self.samples)} timestep samples; raw joints={self.raw_joint_dim}D, "
                f"model joints={len(self.joint_indices)}D indices={self.joint_indices}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def get_stats(self) -> Dict[str, object]:
        return {
            'joint_mean': self.joint_mean.copy(),
            'joint_std': self.joint_std.copy(),
            'action_source': 'next embodiment/joint positions',
            'raw_joint_dim': self.raw_joint_dim,
            'state_dim': len(self.joint_indices),
            'joint_indices': list(self.joint_indices),
            'control_layout': '7 arm joints + 1 gripper command' if self.joint_indices == list(range(8)) else None,
            'normalization_source': self.normalization_source,
        }

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        if not self.normalize_joints:
            return values.astype(np.float32, copy=False)
        return ((values - self.joint_mean) / self.joint_std).astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode_path, timestep = self.samples[idx]

        with h5py.File(episode_path, 'r') as root:
            joints = root['embodiment/joint']
            qpos = np.asarray(joints[timestep], dtype=np.float32)[self.joint_indices]
            qpos = self._normalize(qpos)

            cam_images = []
            for name in self.camera_names:
                path = CAMERA_PATHS.get(name, f'observation/{name}/rgb')
                cam_images.append(image_to_tensor(root[path][timestep]))

            tac_images = []
            for name in self.tactile_names:
                path = self.tactile_paths_by_episode[episode_path][name]
                tac_images.append(image_to_tensor(root[path][timestep]))

            action_end = min(timestep + 1 + self.chunk_size, len(joints))
            future_joints = np.asarray(
                joints[timestep + 1:action_end], dtype=np.float32
            )[:, self.joint_indices]
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
        self.episode_files = discover_episode_files(self.dataset_dir)

        self.samples = []
        for episode_path in self.episode_files:
            with h5py.File(episode_path, 'r') as root:
                for name in self.tactile_names:
                    rgb_path = resolve_tactile_path(root, name, episode_path)
                    self.samples.extend(
                        (episode_path, name, rgb_path, timestep)
                        for timestep in range(len(root[rgb_path]))
                    )

        if verbose:
            print(
                f"Found {len(self.episode_files)} episodes and "
                f"{len(self.samples)} tactile frames for pretraining"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        episode_path, tactile_name, rgb_path, timestep = self.samples[idx]
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
            joint_indices=getattr(args, 'joint_indices', None),
        )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
