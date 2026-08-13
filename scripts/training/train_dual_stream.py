"""
Dual-Stream VTLA Training Script
双流架构端到端训练脚本
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path

# 尝试导入wandb（可选）
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, will use tensorboard only")

# 添加项目根目录到path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.dual_stream_vtla_policy import DualStreamVTLAPolicy, build_dual_stream_vtla_model
from dataloader import create_dataloader
from training_utils import (
    append_epoch_metrics,
    build_run_config,
    write_run_config,
)


class DualStreamTrainer:
    """双流VTLA训练器"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.run_dir = None

        print(f"\n{'='*60}")
        print(f"Dual-Stream VTLA Training")
        print(f"{'='*60}")

        # 创建运行目录
        self._setup_run_directory()

        # 加载数据
        print("\n[1/5] Loading dataset...")
        self.train_dataloader, self.val_dataloader, self.dataset_stats = self._load_data()

        # 构建模型
        print("\n[2/5] Building model...")
        self.model = self._build_model()

        # 优化器
        print("\n[3/5] Setting up optimizer...")
        self.optimizer = self._build_optimizer()

        # 学习率调度器（可选）
        self.scheduler = None
        if hasattr(args, 'use_scheduler') and args.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=args.lr_decay_step,
                gamma=args.lr_decay_gamma
            )

        # 保存配置
        print("\n[4/5] Saving configuration...")
        self._save_config()

        # 初始化日志记录器
        print("\n[5/6] Setting up logging...")
        self._setup_logging()

        print("\n[6/6] Setup complete!")
        self._print_model_info()

    def _setup_run_directory(self):
        """创建运行目录"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        git_hash = os.popen('git rev-parse --short HEAD 2>/dev/null').read().strip() or 'nogit'

        run_name = f"dual_stream_{self.args.task}_{timestamp}_{git_hash}"
        self.run_dir = Path(self.args.output_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # 创建checkpoints目录
        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)

        print(f"Run directory: {self.run_dir}")

    def _load_data(self):
        """加载数据"""
        # 创建训练dataloader
        train_dataloader = create_dataloader(self.args, stage='stage2')

        # 创建验证dataloader（使用相同的dataset但不shuffle）
        # 简化处理：暂时使用训练集的一部分作为验证
        # TODO: 实现proper train/val split
        val_dataloader = None  # 先不用验证集

        # Dataset stats（从第一个batch推断）
        sample_batch = next(iter(train_dataloader))
        stats = {
            'qpos_mean': 0.0,
            'qpos_std': 1.0,
            'action_mean': 0.0,
            'action_std': 1.0,
        }

        print(f"  Train batches: {len(train_dataloader)}")
        print(f"  Sample batch keys: {list(sample_batch.keys())}")

        return train_dataloader, val_dataloader, stats

    def _build_model(self):
        """构建双流VTLA模型"""
        # 将args转换为dict以兼容build函数
        args_dict = vars(self.args)

        # 添加dataset stats到args
        args_dict['dataset_stats'] = self.dataset_stats

        # 构建策略（包含模型）
        policy = DualStreamVTLAPolicy(args_dict)
        policy.to(self.device)

        return policy

    def _build_optimizer(self):
        """构建优化器 - 支持不同学习率"""
        # 分组参数：vision backbone, tactile encoder, 其他
        vision_backbone_params = []
        tactile_encoder_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if 'vision_backbone' in name:
                vision_backbone_params.append(param)
            elif 'tactile_encoder' in name:
                tactile_encoder_params.append(param)
            else:
                other_params.append(param)

        # 构建参数组
        param_groups = [
            {'params': other_params, 'lr': self.args.lr},
            {'params': vision_backbone_params, 'lr': self.args.lr_backbone},
            {'params': tactile_encoder_params, 'lr': self.args.lr_tactile}
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )

        print(f"  Main LR: {self.args.lr}")
        print(f"  Vision backbone LR: {self.args.lr_backbone}")
        print(f"  Tactile encoder LR: {self.args.lr_tactile}")
        print(f"  Weight decay: {self.args.weight_decay}")

        return optimizer

    def _save_config(self):
        """保存训练配置"""
        config = {
            'task': self.args.task,
            'dataset_dir': self.args.dataset_dir,
            'output_dir': str(self.run_dir),
            'batch_size': self.args.batch_size,
            'num_epochs': self.args.num_epochs,
            'lr': self.args.lr,
            'lr_backbone': self.args.lr_backbone,
            'lr_vision_backbone': self.args.lr_vision_backbone,
            'lr_tactile': self.args.lr_tactile,
            'weight_decay': self.args.weight_decay,
            'chunk_size': self.args.chunk_size,
            'state_dim': self.args.state_dim,
            'hidden_dim': self.args.hidden_dim,
            'model_type': 'dual_stream_vtla',
            'dual_stream_config': {
                'shared_encoder': self.args.shared_encoder,
                'shared_decoder': self.args.shared_decoder,
                'enable_cross_stream': self.args.enable_cross_stream,
                'cross_stream_layers': self.args.cross_stream_layers,
                'fusion_type': self.args.fusion_type,
                'use_contact_routing': self.args.use_contact_routing,
                'use_cvae': self.args.use_cvae,
            },
            'dataset_stats': self.dataset_stats,
        }

        # 保存配置
        config_path = self.run_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)

        print(f"  Config saved to: {config_path}")

    def _setup_logging(self):
        """设置TensorBoard和WandB日志"""
        # TensorBoard
        self.tb_writer = SummaryWriter(log_dir=self.run_dir / 'tensorboard')
        print(f"  TensorBoard log: {self.run_dir / 'tensorboard'}")

        # WandB (可选)
        self.use_wandb = WANDB_AVAILABLE and getattr(self.args, 'use_wandb', False)
        if self.use_wandb:
            wandb.init(
                project=getattr(self.args, 'wandb_project', 'dual-stream-vtla'),
                name=self.run_dir.name,
                config=vars(self.args),
                dir=str(self.run_dir),
            )
            print(f"  WandB initialized: {wandb.run.url}")
        else:
            print(f"  WandB: disabled")

        # CSV日志文件
        self.csv_log_path = self.run_dir / 'training_log.csv'
        with open(self.csv_log_path, 'w') as f:
            f.write('epoch,train_loss,train_l1,train_kl,train_pad,val_loss,val_l1,val_kl,val_pad,lr\n')
        print(f"  CSV log: {self.csv_log_path}")

    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\nModel Information:")
        print(f"  Total parameters: {total_params / 1e6:.2f}M")
        print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
        print(f"  Fusion type: {self.args.fusion_type}")
        print(f"  Shared encoder: {self.args.shared_encoder}")
        print(f"  Shared decoder: {self.args.shared_decoder}")
        print(f"  Enable cross-stream: {self.args.enable_cross_stream}")

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()

        epoch_loss = 0
        epoch_metrics = {
            'l1': 0,
            'kl': 0,
            'pad': 0,
            'aux_vision': 0,
            'aux_tactile': 0,
        }

        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{self.args.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # 将数据移到device
            qpos = batch['qpos'].to(self.device)
            cam_image = batch['cam_image'].to(self.device)
            tac_image = batch['tac_image'].to(self.device)
            actions = batch['actions'].to(self.device)
            is_pad = batch['is_pad'].to(self.device)

            # 前向传播
            loss_dict = self.model(
                qpos=qpos,
                cam_image=cam_image,
                tac_image=tac_image,
                actions=actions,
                is_pad=is_pad
            )

            loss = loss_dict['loss']

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（可选）
            if hasattr(self.args, 'grad_clip') and self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.grad_clip
                )

            self.optimizer.step()

            # 累积指标
            epoch_loss += loss.item()
            for key in epoch_metrics:
                if key in loss_dict:
                    epoch_metrics[key] += loss_dict[key].item()

            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'l1': f"{loss_dict['l1'].item():.4f}",
                'kl': f"{loss_dict['kl'].item():.4f}"
            })

        # 平均指标
        num_batches = len(self.train_dataloader)
        avg_loss = epoch_loss / num_batches
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        return avg_loss, epoch_metrics

    def validate(self):
        """验证"""
        if self.val_dataloader is None:
            # 没有验证集，返回空结果
            return 0.0, {'l1': 0.0, 'kl': 0.0, 'pad': 0.0}

        self.model.eval()

        val_loss = 0
        val_metrics = {
            'l1': 0,
            'kl': 0,
            'pad': 0,
        }

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                qpos = batch['qpos'].to(self.device)
                cam_image = batch['cam_image'].to(self.device)
                tac_image = batch['tac_image'].to(self.device)
                actions = batch['actions'].to(self.device)
                is_pad = batch['is_pad'].to(self.device)

                loss_dict = self.model(
                    qpos=qpos,
                    cam_image=cam_image,
                    tac_image=tac_image,
                    actions=actions,
                    is_pad=is_pad
                )

                val_loss += loss_dict['loss'].item()
                for key in val_metrics:
                    if key in loss_dict:
                        val_metrics[key] += loss_dict[key].item()

        # 平均指标
        num_batches = len(self.val_dataloader)
        avg_val_loss = val_loss / num_batches
        for key in val_metrics:
            val_metrics[key] /= num_batches

        return avg_val_loss, val_metrics

    def save_checkpoint(self, epoch, is_best=False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'args': vars(self.args),
            'dataset_stats': self.dataset_stats,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # 保存last checkpoint
        ckpt_path = self.ckpt_dir / f"dual_stream_last.ckpt"
        torch.save(checkpoint, ckpt_path)

        # 保存best checkpoint
        if is_best:
            best_path = self.ckpt_dir / f"dual_stream_best.ckpt"
            torch.save(checkpoint, best_path)
            print(f"  [BEST] Saved to {best_path}")

        # 保存epoch checkpoint（可选）
        if epoch % self.args.save_every == 0:
            epoch_path = self.ckpt_dir / f"dual_stream_epoch_{epoch}.ckpt"
            torch.save(checkpoint, epoch_path)

    def train(self):
        """完整训练流程"""
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}\n")

        best_val_loss = float('inf')
        metrics_log = self.run_dir / "metrics.jsonl"

        for epoch in range(1, self.args.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.args.num_epochs}")
            print("-" * 60)

            # 训练
            train_loss, train_metrics = self.train_epoch(epoch)

            # 验证
            val_loss, val_metrics = self.validate()

            # 学习率调度
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"  Learning rate: {current_lr:.6f}")

            # 打印指标
            print(f"\n  Train Loss: {train_loss:.4f}")
            print(f"    L1: {train_metrics['l1']:.4f}")
            print(f"    KL: {train_metrics['kl']:.4f}")
            print(f"    Pad: {train_metrics['pad']:.4f}")

            print(f"  Val Loss: {val_loss:.4f}")
            print(f"    L1: {val_metrics['l1']:.4f}")
            print(f"    KL: {val_metrics['kl']:.4f}")
            print(f"    Pad: {val_metrics['pad']:.4f}")

            # 记录到TensorBoard
            self.tb_writer.add_scalar('Loss/train', train_loss, epoch)
            self.tb_writer.add_scalar('Loss/val', val_loss, epoch)
            self.tb_writer.add_scalar('Train/L1', train_metrics['l1'], epoch)
            self.tb_writer.add_scalar('Train/KL', train_metrics['kl'], epoch)
            self.tb_writer.add_scalar('Train/Pad', train_metrics['pad'], epoch)
            self.tb_writer.add_scalar('Val/L1', val_metrics['l1'], epoch)
            self.tb_writer.add_scalar('Val/KL', val_metrics['kl'], epoch)
            self.tb_writer.add_scalar('Val/Pad', val_metrics['pad'], epoch)
            self.tb_writer.add_scalar('LearningRate', current_lr, epoch)

            # 记录到WandB
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'train/l1': train_metrics['l1'],
                    'train/kl': train_metrics['kl'],
                    'train/pad': train_metrics['pad'],
                    'val/loss': val_loss,
                    'val/l1': val_metrics['l1'],
                    'val/kl': val_metrics['kl'],
                    'val/pad': val_metrics['pad'],
                    'lr': current_lr,
                }, step=epoch)

            # 保存到CSV
            with open(self.csv_log_path, 'a') as f:
                f.write(f"{epoch},{train_loss:.6f},{train_metrics['l1']:.6f},"
                       f"{train_metrics['kl']:.6f},{train_metrics['pad']:.6f},"
                       f"{val_loss:.6f},{val_metrics['l1']:.6f},"
                       f"{val_metrics['kl']:.6f},{val_metrics['pad']:.6f},"
                       f"{current_lr:.8f}\n")

            # 保存指标
            metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
            }
            with open(metrics_log, 'a') as f:
                json.dump(metrics, f)
                f.write('\n')

            # 保存checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            self.save_checkpoint(epoch, is_best=is_best)

        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Run directory: {self.run_dir}")
        print(f"TensorBoard: tensorboard --logdir={self.run_dir / 'tensorboard'}")
        print(f"CSV log: {self.csv_log_path}")
        if self.use_wandb:
            print(f"WandB: {wandb.run.url}")
        print(f"{'='*60}\n")

        # 关闭日志记录器
        self.tb_writer.close()
        if self.use_wandb:
            wandb.finish()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Dual-Stream VTLA Training')

    # 基础参数
    parser.add_argument('--task', type=str, required=True, help='Task name')
    parser.add_argument('--dataset-dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--output-dir', type=str, default='runs/dual_stream', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    # 数据参数
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--num-episodes', type=int, default=50, help='Number of episodes')
    parser.add_argument('--train-ratio', type=float, default=0.9, help='Train ratio')
    parser.add_argument('--chunk-size', type=int, default=50, help='Action chunk size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')

    # 相机和触觉传感器
    parser.add_argument('--camera-names', nargs='+', default=['cam_high'], help='Camera names')
    parser.add_argument('--tactile-names', nargs='+', default=['tac_left', 'tac_right'], help='Tactile names')

    # 模型参数
    parser.add_argument('--state-dim', type=int, default=14, help='State dimension')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--nheads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dim-feedforward', type=int, default=2048, help='FFN dimension')
    parser.add_argument('--enc-layers', type=int, default=4, help='Encoder layers')
    parser.add_argument('--dec-layers', type=int, default=6, help='Decoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout')
    parser.add_argument('--pre-norm', action='store_true', help='Pre-normalization')

    # 双流特定参数
    parser.add_argument('--shared-encoder', type=bool, default=True, help='Share encoder weights')
    parser.add_argument('--shared-decoder', type=bool, default=False, help='Share decoder weights')
    parser.add_argument('--enable-cross-stream', type=bool, default=False, help='Enable cross-stream attention')
    parser.add_argument('--cross-stream-layers', nargs='+', type=int, default=[], help='Cross-stream layers')
    parser.add_argument('--fusion-type', type=str, default='gated', choices=['concat', 'gated', 'cross_attn', 'moe'], help='Fusion type')
    parser.add_argument('--use-contact-routing', type=bool, default=False, help='Use contact-aware routing')
    parser.add_argument('--use-cvae', type=bool, default=True, help='Use CVAE')
    parser.add_argument('--latent-dim', type=int, default=32, help='CVAE latent dimension')

    # 触觉编码器参数
    parser.add_argument('--tactile-backbone', type=str, default='resnet34', help='Tactile backbone')
    parser.add_argument('--tactile-latent-dim', type=int, default=512, help='Tactile latent dimension')
    parser.add_argument('--pretrained-backbones', type=bool, default=True, help='Use pretrained backbones')

    # Vision backbone参数
    parser.add_argument('--backbone', type=str, default='resnet18', help='Vision backbone')
    parser.add_argument('--position-embedding', type=str, default='sine', help='Position embedding type')
    parser.add_argument('--masks', type=bool, default=False, help='Use masks')
    parser.add_argument('--dilation', type=bool, default=False, help='Use dilation')

    # 损失权重
    parser.add_argument('--kl-weight', type=float, default=10.0, help='KL divergence weight')
    parser.add_argument('--pad-weight', type=float, default=1.0, help='Padding loss weight')
    parser.add_argument('--l1-reduction', type=str, default='valid_mean', help='L1 reduction method')
    parser.add_argument('--aux-vision-weight', type=float, default=0.0, help='Auxiliary vision loss weight')
    parser.add_argument('--aux-tactile-weight', type=float, default=0.0, help='Auxiliary tactile loss weight')

    # 训练参数
    parser.add_argument('--num-epochs', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr-backbone', type=float, default=1e-5, help='Backbone learning rate')
    parser.add_argument('--lr-vision-backbone', type=float, default=1e-5, help='Vision backbone learning rate')
    parser.add_argument('--lr-tactile', type=float, default=1e-5, help='Tactile encoder learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=0.0, help='Gradient clipping (0 to disable)')

    # 学习率调度（可选）
    parser.add_argument('--use-scheduler', type=bool, default=False, help='Use LR scheduler')
    parser.add_argument('--lr-decay-step', type=int, default=1000, help='LR decay step')
    parser.add_argument('--lr-decay-gamma', type=float, default=0.1, help='LR decay gamma')

    # Checkpoint
    parser.add_argument('--save-every', type=int, default=100, help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    # Logging
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='dual-stream-vtla', help='WandB project name')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 创建训练器
    trainer = DualStreamTrainer(args)

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
