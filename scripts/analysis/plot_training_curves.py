#!/usr/bin/env python3
"""
训练曲线可视化脚本
从CSV日志生成loss曲线图
"""
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys

def plot_training_curves(csv_path, output_dir=None):
    """绘制训练曲线"""
    # 读取CSV
    df = pd.read_csv(csv_path)

    if output_dir is None:
        output_dir = Path(csv_path).parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    # 设置样式
    plt.style.use('seaborn-v0_8-darkgrid')

    # 1. Total Loss曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in df.columns and df['val_loss'].notna().any():
        ax.plot(df['epoch'], df['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Total Loss Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_curve.png', dpi=150)
    print(f"✓ Saved: {output_dir / 'loss_curve.png'}")
    plt.close()

    # 2. L1 Loss曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['epoch'], df['train_l1'], label='Train L1', linewidth=2)
    if 'val_l1' in df.columns and df['val_l1'].notna().any():
        ax.plot(df['epoch'], df['val_l1'], label='Val L1', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('L1 Loss', fontsize=12)
    ax.set_title('L1 Loss Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'l1_loss_curve.png', dpi=150)
    print(f"✓ Saved: {output_dir / 'l1_loss_curve.png'}")
    plt.close()

    # 3. KL Loss曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['epoch'], df['train_kl'], label='Train KL', linewidth=2)
    if 'val_kl' in df.columns and df['val_kl'].notna().any():
        ax.plot(df['epoch'], df['val_kl'], label='Val KL', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('KL Divergence', fontsize=12)
    ax.set_title('KL Divergence Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'kl_loss_curve.png', dpi=150)
    print(f"✓ Saved: {output_dir / 'kl_loss_curve.png'}")
    plt.close()

    # 4. 综合图（2x2）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total Loss
    axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train', linewidth=2)
    if 'val_loss' in df.columns and df['val_loss'].notna().any():
        axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # L1 Loss
    axes[0, 1].plot(df['epoch'], df['train_l1'], label='Train', linewidth=2)
    if 'val_l1' in df.columns and df['val_l1'].notna().any():
        axes[0, 1].plot(df['epoch'], df['val_l1'], label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('L1 Loss')
    axes[0, 1].set_title('L1 Loss', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # KL Loss
    axes[1, 0].plot(df['epoch'], df['train_kl'], label='Train', linewidth=2)
    if 'val_kl' in df.columns and df['val_kl'].notna().any():
        axes[1, 0].plot(df['epoch'], df['val_kl'], label='Val', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_title('KL Divergence', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning Rate
    axes[1, 1].plot(df['epoch'], df['lr'], linewidth=2, color='green')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.suptitle('Training Curves Summary', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_summary.png', dpi=150)
    print(f"✓ Saved: {output_dir / 'training_summary.png'}")
    plt.close()

    # 5. 打印统计信息
    print(f"\n{'='*60}")
    print(f"Training Statistics")
    print(f"{'='*60}")
    print(f"Total epochs: {len(df)}")
    print(f"\nTrain Loss:")
    print(f"  Initial: {df['train_loss'].iloc[0]:.4f}")
    print(f"  Final:   {df['train_loss'].iloc[-1]:.4f}")
    print(f"  Best:    {df['train_loss'].min():.4f} (epoch {df['train_loss'].idxmin() + 1})")
    print(f"\nTrain L1 Loss:")
    print(f"  Initial: {df['train_l1'].iloc[0]:.4f}")
    print(f"  Final:   {df['train_l1'].iloc[-1]:.4f}")
    print(f"  Best:    {df['train_l1'].min():.4f} (epoch {df['train_l1'].idxmin() + 1})")
    print(f"\nTrain KL Loss:")
    print(f"  Initial: {df['train_kl'].iloc[0]:.4f}")
    print(f"  Final:   {df['train_kl'].iloc[-1]:.4f}")
    print(f"  Mean:    {df['train_kl'].mean():.4f}")
    print(f"{'='*60}\n")

    # 检查收敛情况
    last_100_epochs = df.tail(100) if len(df) > 100 else df
    loss_std = last_100_epochs['train_loss'].std()
    loss_mean = last_100_epochs['train_loss'].mean()

    print(f"Convergence Analysis (last {len(last_100_epochs)} epochs):")
    print(f"  Loss mean: {loss_mean:.4f}")
    print(f"  Loss std:  {loss_std:.4f}")

    if loss_std < 0.05:
        print(f"  Status: ✓ Well converged (std < 0.05)")
    elif loss_std < 0.1:
        print(f"  Status: ⚠ Converging (std < 0.1)")
    else:
        print(f"  Status: ⚠ Still training (std > 0.1)")
    print()

def main():
    parser = argparse.ArgumentParser(description='Plot training curves from CSV log')
    parser.add_argument('csv_path', type=str, help='Path to training_log.csv')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for plots (default: same as CSV)')
    args = parser.parse_args()

    csv_path = Path(args.csv_path)

    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist")
        sys.exit(1)

    print(f"Reading training log from: {csv_path}")
    plot_training_curves(csv_path, args.output_dir)
    print("✓ All plots generated successfully!")

if __name__ == '__main__':
    main()
