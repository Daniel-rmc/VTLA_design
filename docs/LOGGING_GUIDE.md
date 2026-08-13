# 训练监控和日志使用指南

**更新时间**: 2026-08-13  
**状态**: ✅ 已启用完整日志系统

---

## 📊 可用的日志系统

训练现在支持三种日志方式：

### 1. CSV日志 (离线分析) ✅
- **位置**: `runs/dual_stream/<run_name>/training_log.csv`
- **内容**: epoch, train_loss, train_l1, train_kl, train_pad, val_loss, val_l1, val_kl, val_pad, lr
- **优势**: 简单、轻量、易于后处理

### 2. TensorBoard ✅
- **位置**: `runs/dual_stream/<run_name>/tensorboard/`
- **启动**: `tensorboard --logdir=<path>`
- **内容**: 实时loss曲线、学习率、各项指标
- **优势**: 交互式、实时可视化、支持平滑

### 3. Weights & Biases (可选)
- **启动方式**: 添加 `--use-wandb` 参数
- **内容**: 云端存储、团队协作、实验对比
- **优势**: 远程监控、自动对比、丰富的可视化

---

## 🚀 快速使用

### 查看训练状态
```bash
# 快速检查（推荐）
./scripts/analysis/check_training_status.sh

# 输出：
# - 已完成的epoch数
# - 最近5个epoch的loss
# - 最佳指标
# - TensorBoard路径
# - 自动生成loss曲线图
```

### 查看实时训练日志
```bash
# 进入tmux会话
tmux attach -t dual_stream_training

# 退出但不停止训练
Ctrl+B, D
```

### 启动TensorBoard
```bash
# 找到run目录
RUN_DIR=$(ls -td runs/dual_stream/dual_stream_* | head -1)

# 启动TensorBoard
tensorboard --logdir=$RUN_DIR/tensorboard --port 6006

# 在浏览器打开: http://localhost:6006
```

### 生成Loss曲线图
```bash
# 手动生成
RUN_DIR=$(ls -td runs/dual_stream/dual_stream_* | head -1)
python scripts/analysis/plot_training_curves.py $RUN_DIR/training_log.csv

# 输出：
# - loss_curve.png (总loss)
# - l1_loss_curve.png (L1 loss)
# - kl_loss_curve.png (KL loss)
# - training_summary.png (4合1图)
# + 收敛性分析统计
```

---

## 📁 日志文件结构

```
runs/dual_stream/dual_stream_insert_HDMI_<timestamp>_<git>/
├── config.json                  # 训练配置
├── training_log.csv             # CSV日志（主要）
├── metrics.jsonl                # JSON Lines格式
├── tensorboard/                 # TensorBoard日志
│   └── events.out.tfevents.*
├── checkpoints/                 # 模型checkpoint
│   ├── dual_stream_epoch_100.ckpt
│   ├── dual_stream_best.ckpt
│   └── dual_stream_last.ckpt
├── loss_curve.png              # 生成的可视化图
├── l1_loss_curve.png
├── kl_loss_curve.png
└── training_summary.png
```

---

## 📈 CSV日志格式

```csv
epoch,train_loss,train_l1,train_kl,train_pad,val_loss,val_l1,val_kl,val_pad,lr
1,2.9148,0.3465,0.2363,0.0000,0.0000,0.0000,0.0000,0.0000,0.00010000
2,0.8626,0.1362,0.0665,0.0000,0.0000,0.0000,0.0000,0.0000,0.00010000
3,0.7234,0.1156,0.0578,0.0000,0.0000,0.0000,0.0000,0.0000,0.00010000
...
```

### 使用pandas分析
```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取
df = pd.read_csv('training_log.csv')

# 查看统计
print(df.describe())

# 绘图
df.plot(x='epoch', y=['train_loss', 'val_loss'])
plt.show()

# 找到最佳epoch
best_epoch = df['train_loss'].idxmin()
print(f"Best epoch: {best_epoch}, loss: {df.loc[best_epoch, 'train_loss']:.4f}")
```

---

## 🔍 TensorBoard使用

### 启动TensorBoard
```bash
RUN_DIR=$(ls -td runs/dual_stream/dual_stream_* | head -1)
tensorboard --logdir=$RUN_DIR/tensorboard --port 6006 --bind_all

# --bind_all: 允许远程访问（如果在服务器上）
```

### 可视化内容
- **Scalars**: 
  - Loss/train, Loss/val
  - Train/L1, Train/KL, Train/Pad
  - Val/L1, Val/KL, Val/Pad
  - LearningRate

### TensorBoard功能
- **平滑曲线**: 调整Smoothing滑块
- **对比实验**: 加载多个run的日志
- **下载数据**: 点击右下角下载按钮

---

## 🌐 Weights & Biases (可选)

### 启用WandB
```bash
# 1. 安装wandb（如果未安装）
pip install wandb

# 2. 登录
wandb login

# 3. 启动训练时添加参数
./scripts/training/start_dual_stream_training.sh insert_HDMI 0 --use-wandb
```

### WandB优势
- ☁️ 云端存储，随时随地查看
- 📊 自动生成丰富的可视化
- 🔄 实验对比和超参数搜索
- 👥 团队协作和分享
- 📱 手机端监控

---

## 📊 监控最佳实践

### 训练初期（0-100 epochs）
**检查频率**: 每30分钟  
**关注指标**:
- Loss是否快速下降（从~3.0降到~1.0）
- 是否出现NaN或爆炸
- GPU利用率是否稳定

**检查命令**:
```bash
./scripts/analysis/check_training_status.sh
```

### 训练中期（100-500 epochs）
**检查频率**: 每2-3小时  
**关注指标**:
- L1 loss收敛速度（应该从~0.3降到~0.15）
- KL loss是否稳定（应该在0.05-0.15之间）
- 学习率调度是否正常

**检查命令**:
```bash
# 查看TensorBoard
tensorboard --logdir=$RUN_DIR/tensorboard

# 或生成曲线图
python scripts/analysis/plot_training_curves.py $RUN_DIR/training_log.csv
```

### 训练后期（500+ epochs）
**检查频率**: 每6-12小时  
**关注指标**:
- Loss是否已经收敛（std < 0.05）
- 是否需要提前停止
- 最佳checkpoint是哪个epoch

**决策**:
- 如果loss已经收敛且不再下降，可以提前停止
- 对比最近100个epochs的std判断收敛性

---

## 🎯 收敛判断标准

### 良好收敛
- ✅ L1 loss < 0.15
- ✅ 最近100 epochs的loss std < 0.05
- ✅ Loss曲线平滑下降，无震荡

### 需要调整
- ⚠️ Loss下降很慢或停滞
- ⚠️ Loss出现周期性震荡
- ⚠️ KL loss过大（>0.3）

### 训练失败
- ❌ Loss为NaN
- ❌ Loss爆炸（突然增大）
- ❌ Loss不下降（1000+ epochs仍>2.0）

---

## 📋 故障排查

### 问题1: CSV文件为空或缺失
**原因**: 训练刚启动，还未完成第一个epoch  
**解决**: 等待第一个epoch完成（~60秒）

### 问题2: TensorBoard无法访问
**原因**: 端口被占用或防火墙阻止  
**解决**:
```bash
# 换个端口
tensorboard --logdir=<path> --port 6007

# 或kill占用6006的进程
lsof -ti:6006 | xargs kill -9
```

### 问题3: 图片生成失败
**原因**: matplotlib未安装或版本不兼容  
**解决**:
```bash
pip install matplotlib seaborn pandas
```

### 问题4: WandB登录失败
**原因**: 网络问题或API key错误  
**解决**:
```bash
# 重新登录
wandb login --relogin

# 或使用离线模式
export WANDB_MODE=offline
```

---

## 💡 高级分析示例

### 对比多个实验
```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取多个run
runs = [
    'runs/dual_stream/run1/training_log.csv',
    'runs/dual_stream/run2/training_log.csv',
]

fig, ax = plt.subplots(figsize=(10, 6))
for run in runs:
    df = pd.read_csv(run)
    ax.plot(df['epoch'], df['train_loss'], label=run.split('/')[-2])

ax.legend()
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss Comparison')
plt.savefig('comparison.png')
```

### 计算移动平均
```python
import pandas as pd

df = pd.read_csv('training_log.csv')

# 10-epoch移动平均
df['loss_ma10'] = df['train_loss'].rolling(window=10).mean()

# 50-epoch移动平均
df['loss_ma50'] = df['train_loss'].rolling(window=50).mean()

# 绘图
df.plot(x='epoch', y=['train_loss', 'loss_ma10', 'loss_ma50'])
```

### 寻找最佳学习率区间
```python
import pandas as pd

df = pd.read_csv('training_log.csv')

# 找到loss下降最快的epoch
df['loss_change'] = df['train_loss'].diff()
fastest_descent = df.loc[df['loss_change'].idxmin()]

print(f"Fastest descent at epoch {fastest_descent['epoch']}")
print(f"LR: {fastest_descent['lr']}")
```

---

## 📞 快速参考

| 任务 | 命令 |
|------|------|
| 检查状态 | `./scripts/analysis/check_training_status.sh` |
| 查看训练 | `tmux attach -t dual_stream_training` |
| 启动TensorBoard | `tensorboard --logdir=<run_dir>/tensorboard` |
| 生成图表 | `python scripts/analysis/plot_training_curves.py <csv_path>` |
| 分析CSV | `pandas` + `matplotlib` |

---

**当前训练状态**: ✅ 正在运行，完整日志已启用  
**推荐检查频率**: 初期30分钟，中期2-3小时，后期6-12小时  
**主要日志**: CSV + TensorBoard

🎉 **祝监控顺利！**
