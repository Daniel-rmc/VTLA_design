import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from dataloader import TactilePretrainDataset, VTLADataset
from models.action_heads import DualPathActionHead
from models.vtla_policy import get_2d_sinusoid_encoding


def _write_dataset(root: Path) -> None:
    hdf5_dir = root / 'hdf5'
    hdf5_dir.mkdir(parents=True)
    image = np.full((4, 16, 20, 3), 128, dtype=np.uint8)
    joints = np.arange(12, dtype=np.float32).reshape(4, 3)
    marker = np.zeros((4, 2, 64, 2), dtype=np.float32)

    with h5py.File(hdf5_dir / '0.hdf5', 'w') as stream:
        stream.create_dataset('embodiment/joint', data=joints)
        stream.create_dataset('observation/head/rgb', data=image)
        stream.create_dataset('observation/wrist/rgb', data=image)
        for side in ('left_tactile', 'right_tactile'):
            stream.create_dataset(f'tactile/{side}/rgb', data=image)
            stream.create_dataset(f'tactile/{side}/marker', data=marker)
            stream.create_dataset(f'tactile/{side}/pose', data=np.zeros((4, 7), np.float32))
            stream.create_dataset(f'tactile/{side}/depth', data=np.full((4, 8, 10), 29.0))


class DatasetTests(unittest.TestCase):
    def test_stage2_uses_each_timestep_and_future_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            _write_dataset(Path(directory))
            dataset = VTLADataset(
                directory,
                ['cam_high', 'cam_wrist'],
                ['tac_left', 'tac_right'],
                chunk_size=3,
                state_dim=3,
                verbose=False,
            )
            self.assertEqual(len(dataset), 3)
            first = dataset[0]
            last = dataset[-1]
            self.assertEqual(first['cam_image'].shape, (2, 3, 256, 256))
            self.assertEqual(first['tac_image'].shape, (2, 3, 256, 256))
            self.assertEqual(first['actions'].shape, (3, 3))
            self.assertEqual((~first['is_pad']).sum().item(), 3)
            self.assertEqual((~last['is_pad']).sum().item(), 1)
            self.assertTrue(torch.isfinite(first['qpos']).all())

    def test_stage1_uses_both_sensors_and_marker_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            _write_dataset(Path(directory))
            dataset = TactilePretrainDataset(
                directory, ['tac_left', 'tac_right'], verbose=False
            )
            self.assertEqual(len(dataset), 8)
            sample = dataset[0]
            self.assertEqual(sample['targets']['marker'].shape, (63, 2))
            self.assertEqual(sample['targets']['depth'].shape, (1, 8, 10))


class ModelUnitTests(unittest.TestCase):
    def test_tactile_residual_respects_configured_scale(self):
        head = DualPathActionHead(
            hidden_dim=16,
            tactile_dim=16,
            action_dim=3,
            refine_scale=0.1,
            adaptive_scale=True,
        )
        _, _, components = head(
            torch.randn(2, 5, 16),
            torch.randn(2, 5, 16),
            return_components=True,
        )
        self.assertLessEqual(components['scaled_residual'].abs().max().item(), 0.100001)
        self.assertLessEqual(components['adaptive_scale'].max().item(), 0.100001)

    def test_2d_position_encoding_shape(self):
        encoding = get_2d_sinusoid_encoding(7, 5, 32, torch.device('cpu'), torch.float32)
        self.assertEqual(encoding.shape, (1, 32, 7, 5))
        self.assertTrue(torch.isfinite(encoding).all())


if __name__ == '__main__':
    unittest.main()
