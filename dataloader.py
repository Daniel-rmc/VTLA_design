"""
简单的数据加载器 - 基于UniVTAC格式
用于VTLA训练的演示
"""
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
from pathlib import Path
import cv2


def decode_image(img_data):
    """解码JPEG编码的图像数据"""
    if isinstance(img_data, bytes) or (isinstance(img_data, np.ndarray) and
                                       (img_data.dtype == np.object_ or img_data.dtype.type == np.bytes_)):
        # JPEG编码的图像，需要解码
        if isinstance(img_data, np.ndarray):
            img_data = img_data.tobytes() if img_data.shape == () else bytes(img_data)
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
        return img
    else:
        # 直接是numpy数组
        return np.array(img_data)


class VTLADataset(Dataset):
    """VTLA训练数据集"""

    def __init__(self, dataset_dir, camera_names, tactile_names, chunk_size=100):
        self.dataset_dir = Path(dataset_dir)
        self.camera_names = camera_names
        self.tactile_names = tactile_names
        self.chunk_size = chunk_size

        # 查找所有hdf5文件
        hdf5_dir = self.dataset_dir / 'hdf5'
        if not hdf5_dir.exists():
            raise ValueError(f"HDF5 directory not found: {hdf5_dir}")

        self.episode_files = sorted(list(hdf5_dir.glob('*.hdf5')))
        print(f"Found {len(self.episode_files)} episodes")

        if len(self.episode_files) == 0:
            raise ValueError(f"No HDF5 files found in {hdf5_dir}")

    def __len__(self):
        return len(self.episode_files)

    def __getitem__(self, idx):
        episode_path = self.episode_files[idx]

        with h5py.File(episode_path, 'r') as f:
            # 读取机器人状态 (从embodiment/joint)
            if 'embodiment/joint' in f:
                qpos = np.array(f['embodiment/joint'][0])  # 取第一帧
            else:
                qpos = np.zeros(14)  # 默认状态维度

            # 读取相机图像 (从observation/)
            cam_images = []
            # 映射相机名称
            camera_mapping = {
                'cam_high': 'observation/head/rgb',
                'cam_wrist': 'observation/wrist/rgb',
            }
            for cam_name in self.camera_names:
                hdf5_path = camera_mapping.get(cam_name, f'observation/{cam_name}/rgb')
                if hdf5_path in f:
                    img_data = f[hdf5_path][0]
                    img = decode_image(img_data)  # 解码JPEG
                    # 调整大小到256x256
                    img = cv2.resize(img, (256, 256))
                    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    cam_images.append(img)

            # 读取触觉图像 (从tactile/)
            tac_images = []
            tactile_mapping = {
                'tac_left': 'tactile/left_tactile/rgb',
                'tac_right': 'tactile/right_tactile/rgb',
            }
            for tac_name in self.tactile_names:
                hdf5_path = tactile_mapping.get(tac_name, f'tactile/{tac_name}/rgb')
                if hdf5_path in f:
                    img_data = f[hdf5_path][0]
                    img = decode_image(img_data)  # 解码JPEG
                    # 调整大小到256x256
                    img = cv2.resize(img, (256, 256))
                    img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    tac_images.append(img)

            # 读取动作序列 (使用embodiment/joint作为action proxy)
            if 'embodiment/joint' in f:
                actions = np.array(f['embodiment/joint'][:])
            else:
                actions = np.zeros((10, 14))  # 默认动作序列

            # 填充或截断到chunk_size
            episode_len = len(actions)
            if episode_len < self.chunk_size:
                # 填充
                actions_padded = np.zeros((self.chunk_size, actions.shape[1]))
                actions_padded[:episode_len] = actions
                is_pad = np.ones(self.chunk_size, dtype=bool)
                is_pad[:episode_len] = False
            else:
                # 截断
                actions_padded = actions[:self.chunk_size]
                is_pad = np.zeros(self.chunk_size, dtype=bool)

        return {
            'qpos': torch.from_numpy(qpos).float(),
            'cam_image': torch.stack(cam_images) if cam_images else torch.zeros(1, 3, 256, 256),
            'tac_image': torch.stack(tac_images) if tac_images else torch.zeros(1, 3, 256, 256),
            'actions': torch.from_numpy(actions_padded).float(),
            'is_pad': torch.from_numpy(is_pad),
        }


class TactilePretrainDataset(Dataset):
    """Stage 1: 触觉编码器预训练数据集"""

    def __init__(self, dataset_dir, tactile_names):
        self.dataset_dir = Path(dataset_dir)
        self.tactile_names = tactile_names

        hdf5_dir = self.dataset_dir / 'hdf5'
        self.episode_files = sorted(list(hdf5_dir.glob('*.hdf5')))
        print(f"Found {len(self.episode_files)} episodes for tactile pretraining")

    def __len__(self):
        return len(self.episode_files) * 10  # 每个episode采样多帧

    def __getitem__(self, idx):
        episode_idx = idx // 10
        frame_idx = idx % 10

        episode_path = self.episode_files[episode_idx % len(self.episode_files)]

        with h5py.File(episode_path, 'r') as f:
            # 读取触觉图像
            tac_name = self.tactile_names[0]
            tactile_mapping = {
                'tac_left': 'tactile/left_tactile/rgb',
                'tac_right': 'tactile/right_tactile/rgb',
            }
            hdf5_path = tactile_mapping.get(tac_name, f'tactile/{tac_name}/rgb')

            if hdf5_path in f:
                total_frames = len(f[hdf5_path])
                frame_idx = min(frame_idx, total_frames - 1)
                img_data = f[hdf5_path][frame_idx]
                img = decode_image(img_data)  # 解码JPEG
                img = cv2.resize(img, (256, 256))
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

                # 读取对应的marker和pose（如果有）
                marker_path = hdf5_path.replace('/rgb', '/marker')
                pose_path = hdf5_path.replace('/rgb', '/pose')

                if marker_path in f:
                    marker = torch.from_numpy(np.array(f[marker_path][frame_idx])).float()
                else:
                    marker = torch.zeros(63, 2)

                if pose_path in f:
                    pose = torch.from_numpy(np.array(f[pose_path][frame_idx])).float()
                else:
                    pose = torch.zeros(7)
            else:
                img = torch.zeros(3, 256, 256)
                marker = torch.zeros(63, 2)
                pose = torch.zeros(7)

        # 自监督目标
        targets = {
            'rgb': img,
            'marker': marker,
            'pose': pose,
        }

        return {
            'tactile_image': img,
            'targets': targets
        }


def create_dataloader(args, stage='stage2'):
    """创建数据加载器"""

    if stage == 'stage1':
        dataset = TactilePretrainDataset(
            args.dataset_dir,
            args.tactile_names
        )
    else:
        dataset = VTLADataset(
            args.dataset_dir,
            args.camera_names,
            args.tactile_names,
            args.chunk_size
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return dataloader


if __name__ == '__main__':
    # 测试数据加载器
    class Args:
        dataset_dir = '/home/rmc/workspace/UniVTAC/data/grasp_classify/demo'
        camera_names = ['cam_high']
        tactile_names = ['tac_left', 'tac_right']
        chunk_size = 50
        batch_size = 2
        num_workers = 0

    args = Args()

    print("Testing Stage 2 dataloader...")
    try:
        dataloader = create_dataloader(args, stage='stage2')

        for batch in dataloader:
            print("Batch keys:", batch.keys())
            print("qpos shape:", batch['qpos'].shape)
            print("cam_image shape:", batch['cam_image'].shape)
            print("tac_image shape:", batch['tac_image'].shape)
            print("actions shape:", batch['actions'].shape)
            print("is_pad shape:", batch['is_pad'].shape)
            break

        print("\n✓ Stage 2 dataloader test passed!")
    except Exception as e:
        print(f"\n✗ Stage 2 test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nTesting Stage 1 dataloader...")
    try:
        dataloader = create_dataloader(args, stage='stage1')

        for batch in dataloader:
            print("Batch keys:", batch.keys())
            print("tactile_image shape:", batch['tactile_image'].shape)
            print("targets keys:", batch['targets'].keys())
            break

        print("\n✓ Stage 1 dataloader test passed!")
    except Exception as e:
        print(f"\n✗ Stage 1 test failed: {e}")
        import traceback
        traceback.print_exc()
