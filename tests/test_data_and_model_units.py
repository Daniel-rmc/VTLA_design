import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from dataloader import (
    TactilePretrainDataset,
    VTLADataset,
    discover_episode_files,
    infer_grasp_classify_label,
    split_episode_files,
    split_episode_files_univtac,
)
from models.action_heads import DualPathActionHead
from models.vtla_policy import get_2d_sinusoid_encoding
from scripts.evaluation.eval_vtla_offline import select_episode_files


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


def _write_official_episode(root: Path, episode_id: int, label: str = 'plain') -> None:
    image = np.full((4, 16, 20, 3), 128, dtype=np.uint8)
    joints = np.arange(36, dtype=np.float32).reshape(4, 9) + episode_id
    with h5py.File(root / f'{episode_id}.hdf5', 'w') as stream:
        stream.create_dataset('embodiment/joint', data=joints)
        stream.create_dataset('observation/head/rgb', data=image)
        stream.create_dataset('observation/wrist/rgb', data=image)
        rough_y, plain_y = ((0.0, -1.0) if label == 'rough' else (1.0, 0.0))
        rough_pose = np.zeros((4, 7), dtype=np.float32)
        plain_pose = np.zeros((4, 7), dtype=np.float32)
        rough_pose[:, 1] = rough_y
        plain_pose[:, 1] = plain_y
        stream.create_dataset('actor/rough_prism', data=rough_pose)
        stream.create_dataset('actor/plain_prism', data=plain_pose)
        for side in ('left_gsmini', 'right_gsmini'):
            stream.create_dataset(f'tactile/{side}/rgb', data=image)


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

    def test_published_layout_projects_raw_nine_dimensional_joints_to_native_eight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_official_episode(root, 0)
            dataset = VTLADataset(
                directory,
                ['cam_high', 'cam_wrist'],
                ['tac_left', 'tac_right'],
                chunk_size=2,
                state_dim=8,
                joint_indices=range(8),
                verbose=False,
            )
            self.assertEqual(dataset.raw_joint_dim, 9)
            self.assertEqual(dataset.get_stats()['joint_indices'], list(range(8)))
            self.assertEqual(dataset[0]['qpos'].shape, (8,))
            self.assertEqual(dataset[0]['actions'].shape, (2, 8))

    def test_episode_split_is_deterministic_and_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for episode_id in range(10):
                _write_official_episode(
                    root, episode_id, label='plain' if episode_id < 6 else 'rough'
                )
            files = discover_episode_files(root)
            labels = {path: infer_grasp_classify_label(path) for path in files}
            train_a, val_a = split_episode_files(files, 0.2, 123, strata=labels)
            train_b, val_b = split_episode_files(files, 0.2, 123, strata=labels)
            self.assertEqual(train_a, train_b)
            self.assertEqual(val_a, val_b)
            self.assertEqual(len(train_a), 8)
            self.assertEqual(len(val_a), 2)
            self.assertFalse(set(train_a) & set(val_a))

    def test_univtac_split_matches_legacy_numpy_permutation(self):
        files = [Path(f'{episode_id}.hdf5') for episode_id in range(50)]
        train, val = split_episode_files_univtac(files, 0.2, 1)
        expected = np.random.RandomState(1).permutation(50)
        self.assertEqual([int(path.stem) for path in train], expected[:40].tolist())
        self.assertEqual([int(path.stem) for path in val], expected[40:].tolist())

    def test_qpos_and_action_stats_use_shifted_pairs_from_all_selected_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_official_episode(root, 0)
            _write_official_episode(root, 10)
            files = discover_episode_files(root)
            dataset = VTLADataset(
                directory,
                ['cam_high'],
                ['tac_left', 'tac_right'],
                chunk_size=2,
                state_dim=8,
                joint_indices=range(8),
                episode_files=files[:1],
                normalization_episode_files=files,
                verbose=False,
            )
            qpos = []
            actions = []
            for path in files:
                with h5py.File(path, 'r') as stream:
                    joints = stream['embodiment/joint'][:, :8]
                    qpos.append(joints[:-1])
                    actions.append(joints[1:])
            stats = dataset.get_stats()
            np.testing.assert_allclose(stats['qpos_mean'], np.concatenate(qpos).mean(0))
            np.testing.assert_allclose(stats['action_mean'], np.concatenate(actions).mean(0))
            self.assertFalse(np.array_equal(stats['qpos_mean'], stats['action_mean']))

    def test_offline_eval_reuses_checkpoint_validation_episode_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for episode_id in range(3):
                _write_official_episode(root, episode_id)
            selected = select_episode_files(
                root,
                'validation',
                {'dataset': {'validation_episodes': ['2.hdf5', '0.hdf5']}},
            )
            self.assertEqual([path.name for path in selected], ['2.hdf5', '0.hdf5'])


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
