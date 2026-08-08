"""
VTLA Multi-GPU Training Script with DDP (Distributed Data Parallel)
支持4卡训练
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import os
import argparse
from tqdm import tqdm
import pickle
import socket
from datetime import timedelta
from collections import defaultdict

from models.tactile_encoder import TactileEncoderWithRefine
from models.vtla_policy import VTLAPolicy
from dataloader import create_dataloader, VTLADataset, TactilePretrainDataset
from training_utils import append_epoch_metrics, build_run_config, write_run_config


def setup_ddp(rank, world_size, args):
    """初始化DDP"""
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=args.ddp_timeout),
    )


def cleanup_ddp():
    """清理DDP"""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


class MultiGPUVTLATrainer:
    """多卡VTLA训练器"""

    def __init__(self, args, rank, world_size):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')
        self.stage = args.stage
        self.is_main = (rank == 0)
        self.dataset_stats = None
        self.run_config = None

        # 只在主进程打印
        if self.is_main:
            os.makedirs(args.ckpt_dir, exist_ok=True)
            print(f"=== Multi-GPU Training: {world_size} GPUs ===")
            print(f"=== Training Stage: {args.stage} ===")

        # 构建模型
        if args.stage == 'stage1':
            self.model = self._build_stage1_model()
        else:
            self.model = VTLAPolicy(args, stage=args.stage)
            self.model.to(self.device)

        # 包装为DDP
        self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=False)

        # 优化器（仅在主进程打印）
        self.optimizer = self._build_optimizer()
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=args.lr_decay_step,
            gamma=args.lr_decay_gamma
        )

        if self.is_main:
            self._print_trainable_params()

    def _build_stage1_model(self):
        """Stage 1: 构建触觉编码器预训练模型"""
        model = TactileEncoderWithRefine(
            backbone=self.args.tactile_backbone,
            latent_dim=self.args.tactile_latent_dim,
            supervise=self.args.tactile_supervise,
            marker_nums=63,
            pretrained=(self.rank == 0)
        )
        model.to(self.device)
        return model

    def _build_optimizer(self):
        """构建优化器"""
        if self.stage == 'stage1':
            params = self.model.parameters()
            lr = self.args.lr_stage1
        elif self.stage == 'stage2':
            params = [
                {'params': self.model.module.model.tactile_encoder.parameters(),
                 'lr': self.args.lr_tactile},
                {'params': self.model.module.model.vision_backbone.parameters(),
                 'lr': self.args.lr_backbone},
                {'params': [p for n, p in self.model.named_parameters()
                           if p.requires_grad and 'tactile_encoder' not in n
                           and 'vision_backbone' not in n],
                 'lr': self.args.lr}
            ]
            lr = self.args.lr
        elif self.stage == 'stage3':
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
        """Stage 1: 触觉编码器自监督预训练"""
        if self.is_main:
            print("\n=== Stage 1: Tactile Encoder Pretraining (Multi-GPU) ===")

        for epoch in range(num_epochs):
            self.model.train()
            train_loader.sampler.set_epoch(epoch)  # 重要：DDP需要设置epoch
            epoch_losses = defaultdict(list)

            if self.is_main:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            else:
                pbar = train_loader

            for batch in pbar:
                tactile_images = batch['tactile_image'].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch['targets'].items()}

                # 前向传播
                loss, loss_dict = self.model(
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

                if self.is_main and isinstance(pbar, tqdm):
                    pbar.set_postfix({k: f"{np.mean(v):.4f}" for k, v in epoch_losses.items()})

            epoch_losses = self.reduce_epoch_losses(epoch_losses)
            peak_memory = self.reduce_peak_memory()
            if self.is_main:
                epoch_losses['peak_gpu_memory_gib'] = [peak_memory]
                print(f"Peak GPU memory: {peak_memory:.2f} GiB")
                append_epoch_metrics(self.args.run_dir, 'stage1', epoch, epoch_losses)
                if (epoch + 1) % self.args.save_freq == 0 or epoch + 1 == num_epochs:
                    self.save_checkpoint(epoch, epoch_losses, stage='stage1')

            self.scheduler.step()

        if self.is_main:
            print("Stage 1 training completed!")

    def train_stage2(self, train_loader, num_epochs):
        """Stage 2: 端到端VLA训练"""
        if self.is_main:
            print("\n=== Stage 2: End-to-End VLA Training (Multi-GPU) ===")

        # 加载预训练的触觉编码器
        if self.args.stage1_ckpt:
            self.load_tactile_encoder(self.args.stage1_ckpt)

        for epoch in range(num_epochs):
            self.model.train()
            train_loader.sampler.set_epoch(epoch)
            epoch_losses = defaultdict(list)

            if self.is_main:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            else:
                pbar = train_loader

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

                if self.is_main and isinstance(pbar, tqdm):
                    pbar.set_postfix({k: f"{np.mean(v):.4f}" for k, v in epoch_losses.items()})

            epoch_losses = self.reduce_epoch_losses(epoch_losses)
            peak_memory = self.reduce_peak_memory()
            if self.is_main:
                epoch_losses['peak_gpu_memory_gib'] = [peak_memory]
                print(f"Peak GPU memory: {peak_memory:.2f} GiB")
                append_epoch_metrics(self.args.run_dir, 'stage2', epoch, epoch_losses)
                if (epoch + 1) % self.args.save_freq == 0 or epoch + 1 == num_epochs:
                    self.save_checkpoint(epoch, epoch_losses, stage='stage2')

            self.scheduler.step()

        if self.is_main:
            print("Stage 2 training completed!")

    def train_stage3(self, train_loader, num_epochs):
        """Stage 3: 触觉微调分支训练"""
        if self.is_main:
            print("\n=== Stage 3: Tactile Refine Branch Training (Multi-GPU) ===")

        # 加载stage2的checkpoint
        if self.args.stage2_ckpt:
            self.load_checkpoint(self.args.stage2_ckpt)

        for epoch in range(num_epochs):
            self.model.train()
            train_loader.sampler.set_epoch(epoch)
            epoch_losses = defaultdict(list)

            if self.is_main:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            else:
                pbar = train_loader

            for batch in pbar:
                qpos = batch['qpos'].to(self.device)
                cam_image = batch['cam_image'].to(self.device)
                tac_image = batch['tac_image'].to(self.device)
                actions = batch['actions'].to(self.device)
                is_pad = batch['is_pad'].to(self.device)

                # 前向传播
                loss_dict = self.model(qpos, cam_image, tac_image, actions, is_pad)

                # 反向传播
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

                if self.is_main and isinstance(pbar, tqdm):
                    pbar.set_postfix({k: f"{np.mean(v):.4f}" for k, v in epoch_losses.items()})

            epoch_losses = self.reduce_epoch_losses(epoch_losses)
            peak_memory = self.reduce_peak_memory()
            if self.is_main:
                epoch_losses['peak_gpu_memory_gib'] = [peak_memory]
                print(f"Peak GPU memory: {peak_memory:.2f} GiB")
                append_epoch_metrics(self.args.run_dir, 'stage3', epoch, epoch_losses)
                if (epoch + 1) % self.args.save_freq == 0 or epoch + 1 == num_epochs:
                    self.save_checkpoint(epoch, epoch_losses, stage='stage3')

            self.scheduler.step()

        if self.is_main:
            print("Stage 3 training completed!")

    def save_checkpoint(self, epoch, losses, stage):
        """保存checkpoint（仅主进程）"""
        if not self.is_main:
            return

        ckpt_path = os.path.join(
            self.args.ckpt_dir,
            f"{stage}_epoch_{epoch+1}.ckpt"
        )
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),  # 注意：DDP需要用module
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'losses': {k: np.mean(v) for k, v in losses.items()},
            'dataset_stats': self.dataset_stats,
            'run_config': self.run_config,
        }, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

    def reduce_epoch_losses(self, losses):
        """Return global, sample-count-weighted epoch means on every rank."""
        reduced = defaultdict(list)
        for key in sorted(losses):
            values = losses[key]
            totals = torch.tensor(
                [float(np.sum(values)), float(len(values))],
                device=self.device,
                dtype=torch.float64,
            )
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            reduced[key].append((totals[0] / totals[1].clamp_min(1)).item())
        return reduced

    def reduce_peak_memory(self):
        peak = torch.tensor(
            torch.cuda.max_memory_allocated(self.device) / 1024 ** 3,
            device=self.device,
            dtype=torch.float64,
        )
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        torch.cuda.reset_peak_memory_stats(self.device)
        return peak.item()

    def load_tactile_encoder(self, ckpt_path):
        """加载预训练的触觉编码器"""
        if self.is_main:
            print(f"Loading tactile encoder from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        tactile_state_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            if k.startswith('encoder.'):
                new_key = k.replace('encoder.', 'model.tactile_encoder.')
                tactile_state_dict[new_key] = v

        self.model.module.load_state_dict(tactile_state_dict, strict=False)
        if self.is_main:
            print("Tactile encoder loaded successfully!")

    def load_checkpoint(self, ckpt_path):
        """加载完整checkpoint"""
        if self.is_main:
            print(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        if self.is_main:
            print("Checkpoint loaded successfully!")


def main_worker(rank, world_size, args):
    """每个GPU进程的主函数"""
    setup_ddp(rank, world_size, args)
    try:
        # Only rank 0 reads pretrained weights; DDP broadcasts them at construction.
        args.pretrained_backbones = (rank == 0)
        torch.manual_seed(args.seed + rank)
        np.random.seed(args.seed + rank)

        trainer = MultiGPUVTLATrainer(args, rank, world_size)

        if args.stage == 'stage1':
            dataset = TactilePretrainDataset(
                args.dataset_dir, args.tactile_names, verbose=(rank == 0)
            )
            dataset_stats = None
        else:
            dataset = VTLADataset(
                args.dataset_dir,
                args.camera_names,
                args.tactile_names,
                args.chunk_size,
                state_dim=args.state_dim,
                verbose=(rank == 0),
            )
            dataset_stats = dataset.get_stats()

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )

        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )
        trainer.dataset_stats = dataset_stats

        if rank == 0:
            print(f"Dataset loaded: {len(train_loader)} batches per GPU")
            print(f"Total samples: {len(dataset)}, Samples per GPU: {sampler.num_samples}")
            trainer.run_config = build_run_config(
                args,
                world_size=world_size,
                dataset_size=len(dataset),
                batches_per_rank=len(train_loader),
                dataset_stats=dataset_stats,
            )
            trainer.run_config['model'] = {
                'total_parameters': sum(p.numel() for p in trainer.model.parameters()),
                'trainable_parameters': sum(
                    p.numel() for p in trainer.model.parameters() if p.requires_grad
                ),
            }
            config_path = write_run_config(args.run_dir, trainer.run_config)
            print(f"Run configuration saved: {config_path}")

        if args.stage == 'stage1':
            trainer.train_stage1(train_loader, args.num_epochs)
        elif args.stage == 'stage2':
            trainer.train_stage2(train_loader, args.num_epochs)
        elif args.stage == 'stage3':
            trainer.train_stage3(train_loader, args.num_epochs)
    finally:
        cleanup_ddp()


def get_args():
    parser = argparse.ArgumentParser()

    # Multi-GPU设置
    parser.add_argument('--num_gpus', type=int, default=4,
                       help='Number of GPUs to use')
    parser.add_argument('--master_addr', type=str, default='127.0.0.1')
    parser.add_argument('--master_port', type=int, default=0,
                       help='DDP TCP port; 0 selects a free local port')
    parser.add_argument('--ddp_timeout', type=int, default=180,
                       help='Collective timeout in seconds')

    # 训练阶段
    parser.add_argument('--stage', type=str, required=True,
                       choices=['stage1', 'stage2', 'stage3'])

    # 数据相关
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--camera_names', nargs='+', default=['cam_high'])
    parser.add_argument('--tactile_names', nargs='+', default=['tac_left', 'tac_right'])
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=4)

    # 模型相关
    parser.add_argument('--state_dim', type=int, default=14)
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
    parser.add_argument('--loss_weights', type=dict, default={'marker': 1.0, 'rgb': 0.5})

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
    parser.add_argument('--seed', type=int, default=42)

    # UniVTAC兼容参数
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--lr_vision_backbone', type=float, default=1e-5)
    parser.add_argument('--masks', action='store_true', default=False)
    parser.add_argument('--dilation', action='store_true', default=False)
    parser.add_argument('--position_embedding', type=str, default='sine')
    parser.add_argument('--pre_norm', action='store_true', default=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    world_size = args.num_gpus
    if world_size < 1 or world_size > torch.cuda.device_count():
        raise ValueError(
            f"Requested {world_size} GPUs, but {torch.cuda.device_count()} are visible"
        )
    args.run_dir = args.run_dir or os.path.dirname(os.path.abspath(args.ckpt_dir))
    os.environ['MASTER_ADDR'] = args.master_addr
    args.master_port = args.master_port or find_free_port()
    os.environ['MASTER_PORT'] = str(args.master_port)

    print(f"{'='*60}")
    print(f"VTLA Multi-GPU Training")
    print(f"{'='*60}")
    print(f"Stage: {args.stage}")
    print(f"Number of GPUs: {world_size}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Effective batch size: {args.batch_size * world_size}")
    print(f"{'='*60}\n")

    # 使用torch.multiprocessing启动多进程
    torch.multiprocessing.spawn(
        main_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Checkpoints saved to: {args.ckpt_dir}")
    print(f"{'='*60}")
