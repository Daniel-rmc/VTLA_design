"""
VTLA Training Script - 三阶段训练策略

Stage 1: 触觉编码器预训练（自监督）
Stage 2: 端到端VLA训练（主干网络）
Stage 3: 触觉微调分支训练（接触感知）
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm
import pickle
from collections import defaultdict

from models.shared import TactileEncoderWithRefine
from models.vtla import VTLAPolicy
from training_utils import append_epoch_metrics, build_run_config, write_run_config


class VTLATrainer:
    """VTLA三阶段训练器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.stage = args.stage
        self.dataset_stats = None
        self.run_config = None

        # 创建保存目录
        os.makedirs(args.ckpt_dir, exist_ok=True)

        # 根据训练阶段构建模型
        if args.stage == 'stage1':
            self.model = self._build_stage1_model()
        else:
            self.model = VTLAPolicy(args, stage=args.stage)
            self.model.to(self.device)

        # 优化器
        self.optimizer = self._build_optimizer()

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=args.lr_decay_step,
            gamma=args.lr_decay_gamma
        )

        print(f"=== Training Stage: {args.stage} ===")
        self._print_trainable_params()

    def _build_stage1_model(self):
        """Stage 1: 构建触觉编码器预训练模型"""
        model = TactileEncoderWithRefine(
            backbone=self.args.tactile_backbone,
            latent_dim=self.args.tactile_latent_dim,
            supervise=self.args.tactile_supervise,
            marker_nums=63,
            pretrained=True
        )
        model.to(self.device)
        return model

    def _build_optimizer(self):
        """构建优化器"""
        if self.stage == 'stage1':
            # Stage 1: 预训练触觉编码器
            params = self.model.parameters()
            lr = self.args.lr_stage1
        elif self.stage == 'stage2':
            # Stage 2: 端到端训练，使用不同学习率
            params = [
                {'params': self.model.model.tactile_encoder.parameters(), 'lr': self.args.lr_tactile},
                {'params': self.model.model.vision_backbone.parameters(), 'lr': self.args.lr_backbone},
                {'params': [p for n, p in self.model.named_parameters()
                           if 'tactile_encoder' not in n and 'vision_backbone' not in n],
                 'lr': self.args.lr}
            ]
            lr = self.args.lr
        elif self.stage == 'stage3':
            # Stage 3: 仅训练触觉微调分支
            params = filter(lambda p: p.requires_grad, self.model.parameters())
            lr = self.args.lr_stage3
        else:
            params = self.model.parameters()
            lr = self.args.lr

        optimizer = torch.optim.AdamW(
            params if isinstance(params, list) else [{'params': params, 'lr': lr}],
            lr=lr,
            weight_decay=self.args.weight_decay
        )
        return optimizer

    def _print_trainable_params(self):
        """打印可训练参数统计"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
        print(f"Frozen parameters: {(total_params - trainable_params) / 1e6:.2f}M")

    def train_stage1(self, train_loader, num_epochs):
        """
        Stage 1: 触觉编码器自监督预训练
        """
        print("\n=== Stage 1: Tactile Encoder Pretraining ===")

        for epoch in range(num_epochs):
            self.model.train()
            epoch_losses = defaultdict(list)

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                # 假设batch包含: tactile_images, targets (marker/rgb/depth/pose)
                tactile_images = batch['tactile_image'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}

                # 前向传播
                loss, loss_dict = self.model.compute_loss(
                    tactile_images, targets, weights=self.args.loss_weights
                )

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()

                # 记录损失
                for k, v in loss_dict.items():
                    epoch_losses[k].append(v)

                pbar.set_postfix({k: f"{np.mean(v):.4f}" for k, v in epoch_losses.items()})

            # 保存checkpoint
            append_epoch_metrics(self.args.run_dir, 'stage1', epoch, epoch_losses)
            if (epoch + 1) % self.args.save_freq == 0 or epoch + 1 == num_epochs:
                self.save_checkpoint(epoch, epoch_losses, stage='stage1')

            self.scheduler.step()

        print("Stage 1 training completed!")
        print(f"Tactile encoder saved to: {self.args.ckpt_dir}")

    def train_stage2(self, train_loader, num_epochs):
        """
        Stage 2: 端到端VLA训练
        """
        print("\n=== Stage 2: End-to-End VLA Training ===")

        # 加载预训练的触觉编码器（如果有）
        if self.args.stage1_ckpt:
            self.load_tactile_encoder(self.args.stage1_ckpt)

        for epoch in range(num_epochs):
            self.model.train()
            epoch_losses = defaultdict(list)

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                qpos = batch['qpos'].to(self.device)
                cam_image = batch['cam_image'].to(self.device)
                tac_image = batch['tac_image'].to(self.device)
                actions = batch['actions'].to(self.device)
                is_pad = batch['is_pad'].to(self.device)

                # 前向传播 + 损失计算
                loss_dict = self.model(qpos, cam_image, tac_image, actions, is_pad)

                # 反向传播
                self.optimizer.zero_grad()
                loss_dict['loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()

                # 记录损失
                for k, v in loss_dict.items():
                    epoch_losses[k].append(v.item())

                pbar.set_postfix({k: f"{np.mean(v):.4f}" for k, v in epoch_losses.items()})

            # 保存checkpoint
            append_epoch_metrics(self.args.run_dir, 'stage2', epoch, epoch_losses)
            if (epoch + 1) % self.args.save_freq == 0 or epoch + 1 == num_epochs:
                self.save_checkpoint(epoch, epoch_losses, stage='stage2')

            self.scheduler.step()

        print("Stage 2 training completed!")

    def train_stage3(self, train_loader, num_epochs):
        """
        Stage 3: 触觉微调分支训练
        使用接触丰富的轨迹段
        """
        print("\n=== Stage 3: Tactile Refine Branch Training ===")

        # 加载stage2的checkpoint
        if self.args.stage2_ckpt:
            self.load_checkpoint(self.args.stage2_ckpt)

        for epoch in range(num_epochs):
            self.model.train()
            epoch_losses = defaultdict(list)

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                qpos = batch['qpos'].to(self.device)
                cam_image = batch['cam_image'].to(self.device)
                tac_image = batch['tac_image'].to(self.device)
                actions = batch['actions'].to(self.device)
                is_pad = batch['is_pad'].to(self.device)

                # 前向传播
                loss_dict = self.model(qpos, cam_image, tac_image, actions, is_pad)

                # 反向传播（仅更新触觉微调分支）
                self.optimizer.zero_grad()
                loss_dict['loss'].backward()
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    self.args.grad_clip
                )
                self.optimizer.step()

                # 记录损失
                for k, v in loss_dict.items():
                    epoch_losses[k].append(v.item())

                pbar.set_postfix({k: f"{np.mean(v):.4f}" for k, v in epoch_losses.items()})

            # 保存checkpoint
            append_epoch_metrics(self.args.run_dir, 'stage3', epoch, epoch_losses)
            if (epoch + 1) % self.args.save_freq == 0 or epoch + 1 == num_epochs:
                self.save_checkpoint(epoch, epoch_losses, stage='stage3')

            self.scheduler.step()

        print("Stage 3 training completed!")

    def save_checkpoint(self, epoch, losses, stage):
        """保存checkpoint"""
        ckpt_path = os.path.join(
            self.args.ckpt_dir,
            f"{stage}_epoch_{epoch+1}.ckpt"
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'losses': {k: np.mean(v) for k, v in losses.items()},
            'dataset_stats': self.dataset_stats,
            'run_config': self.run_config,
        }, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

    def load_tactile_encoder(self, ckpt_path):
        """加载预训练的触觉编码器"""
        print(f"Loading tactile encoder from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        # 提取触觉编码器的参数
        tactile_state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('encoder.'):
                new_key = k.replace('encoder.', 'model.tactile_encoder.')
                tactile_state_dict[new_key] = v

        self.model.load_state_dict(tactile_state_dict, strict=False)
        print("Tactile encoder loaded successfully!")

    def load_checkpoint(self, ckpt_path):
        """加载完整checkpoint"""
        print(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Checkpoint loaded successfully!")


def get_args():
    parser = argparse.ArgumentParser()

    # 训练阶段
    parser.add_argument('--stage', type=str, required=True,
                       choices=['stage1', 'stage2', 'stage3'],
                       help='Training stage')

    # 数据相关
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--camera_names', nargs='+', default=['cam_high'])
    parser.add_argument('--tactile_names', nargs='+', default=['tac_left', 'tac_right'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)

    # 模型相关
    parser.add_argument('--state_dim', type=int, default=8)
    parser.add_argument('--joint_indices', nargs='+', type=int, default=None,
                       help='Raw UniVTAC joint columns used by the model; defaults to range(state_dim)')
    parser.add_argument('--chunk_size', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--enc_layers', type=int, default=4)
    parser.add_argument('--dec_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)

    # 触觉编码器
    parser.add_argument('--tactile_backbone', type=str, default='resnet34')
    parser.add_argument('--tactile_latent_dim', type=int, default=512)
    parser.add_argument('--tactile_supervise', nargs='+', default=['marker', 'rgb'])

    # 交叉注意力
    parser.add_argument('--cross_attn_layers', type=int, default=2)
    parser.add_argument('--use_tactile_refine', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--refine_scale', type=float, default=0.1)

    # 损失权重
    parser.add_argument('--kl_weight', type=float, default=10.0)
    parser.add_argument('--refine_weight', type=float, default=0.5)
    parser.add_argument('--contact_weight', type=float, default=0.1)
    parser.add_argument('--pad_weight', type=float, default=1.0)

    # 训练超参数
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_stage1', type=float, default=1e-4)
    parser.add_argument('--lr_stage3', type=float, default=5e-5)
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--lr_tactile', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--lr_decay_step', type=int, default=200)
    parser.add_argument('--lr_decay_gamma', type=float, default=0.5)

    # Checkpoint
    parser.add_argument('--ckpt_dir', type=str, required=True)
    parser.add_argument('--stage1_ckpt', type=str, default=None)
    parser.add_argument('--stage2_ckpt', type=str, default=None)
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--run_dir', type=str, default=None,
                       help='Directory for config.json and metrics.jsonl')

    # 其他
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)

    # UniVTAC兼容参数
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--masks', action='store_true', default=False)
    parser.add_argument('--dilation', action='store_true', default=False)
    parser.add_argument('--position_embedding', type=str, default='sine')
    parser.add_argument('--pre_norm', action='store_true', default=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    args.run_dir = args.run_dir or os.path.dirname(os.path.abspath(args.ckpt_dir))

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 创建训练器
    trainer = VTLATrainer(args)

    # 加载数据
    from dataloader import create_dataloader

    print(f"Loading dataset from: {args.dataset_dir}")
    train_loader = create_dataloader(args, stage=args.stage)
    print(f"Dataset loaded: {len(train_loader)} batches")
    trainer.dataset_stats = (
        train_loader.dataset.get_stats()
        if hasattr(train_loader.dataset, 'get_stats') else None
    )
    trainer.run_config = build_run_config(
        args,
        world_size=1,
        dataset_size=len(train_loader.dataset),
        batches_per_rank=len(train_loader),
        dataset_stats=trainer.dataset_stats,
    )
    trainer.run_config['model'] = {
        'total_parameters': sum(p.numel() for p in trainer.model.parameters()),
        'trainable_parameters': sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        ),
    }
    config_path = write_run_config(args.run_dir, trainer.run_config)
    print(f"Run configuration saved: {config_path}")

    # 训练流程
    if args.stage == 'stage1':
        trainer.train_stage1(train_loader, args.num_epochs)
    elif args.stage == 'stage2':
        trainer.train_stage2(train_loader, args.num_epochs)
    elif args.stage == 'stage3':
        trainer.train_stage3(train_loader, args.num_epochs)

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Checkpoints saved to: {args.ckpt_dir}")
    print(f"{'='*60}")
