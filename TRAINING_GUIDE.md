# VTLA训练启动指南

## 🖥️ 服务器配置

- **GPU**: 4x NVIDIA L40S (46GB 每张)
- **CPU**: Intel Xeon Gold 6330 (112 cores)
- **内存**: 251GB
- **环境**: Conda (UniVTAC)

## 🚀 快速启动

### 单卡训练（快速测试）
```bash
cd /home/rmc/workspace/VTLA_design

# Stage 2 单卡训练
./start_training.sh stage2
```

### **多卡训练（推荐 - 4卡并行）**
```bash
cd /home/rmc/workspace/VTLA_design

# Stage 1: 触觉编码器预训练 (4卡)
./start_training_multigpu.sh stage1 4

# Stage 2: 端到端VLA训练 (4卡)
./start_training_multigpu.sh stage2 4

# Stage 3: 触觉微调分支训练 (4卡)
./start_training_multigpu.sh stage3 4
```

## 📊 训练配置

### Stage 1 (触觉编码器预训练)
- **数据**: 触觉图像 + 自监督标签
- **Batch Size**: 8 per GPU (4卡总batch=32)
- **Epochs**: 100
- **预计时间**: 2-3小时
- **显存占用**: ~6GB per GPU

### Stage 2 (端到端VLA训练)
- **数据**: 完整演示轨迹
- **Batch Size**: 4 per GPU (4卡总batch=16)
- **Epochs**: 500
- **预计时间**: 12-18小时
- **显存占用**: ~12GB per GPU

### Stage 3 (触觉微调分支)
- **数据**: 接触丰富的轨迹段
- **Batch Size**: 4 per GPU (4卡总batch=16)
- **Epochs**: 200
- **预计时间**: 4-6小时
- **显存占用**: ~10GB per GPU

## 📁 目录结构

```
/home/rmc/workspace/VTLA_design/
├── logs/                          # 训练日志
│   ├── stage1/
│   │   └── train_4gpu_20240809_*.log
│   ├── stage2/
│   └── stage3/
├── checkpoints/                   # 模型检查点
│   ├── stage1/
│   │   ├── stage1_epoch_50.ckpt
│   │   └── stage1_epoch_100.ckpt
│   ├── stage2/
│   └── stage3/
├── models/                        # 模型代码
├── start_training_multigpu.sh    # 多卡启动脚本 ⭐
├── start_training.sh             # 单卡启动脚本
├── check_training.sh             # 状态检查脚本
└── train_vtla_multigpu.py        # 多卡训练代码
```

## 🎮 训练控制

### 启动训练
```bash
# 方式1: 交互式（会提示是否attach）
./start_training_multigpu.sh stage2 4

# 方式2: 后台运行（不attach）
./start_training_multigpu.sh stage2 4 <<< "n"
```

### 查看训练进度
```bash
# 方法1: attach到tmux会话
tmux attach -t vtla_stage2_4gpu

# 方法2: 实时查看日志
tail -f logs/stage2/train_4gpu_*.log

# 方法3: 使用状态检查脚本
./check_training.sh
```

### 监控GPU使用
```bash
# 实时监控
watch -n 1 nvidia-smi

# 或者使用gpustat (如果安装了)
watch -n 1 gpustat -cp
```

### tmux操作
```bash
# Detach (在tmux内部): Ctrl+B, 然后按 D
# 列出所有会话
tmux ls

# Attach到指定会话
tmux attach -t vtla_stage2_4gpu

# 杀掉会话（停止训练）
tmux kill-session -t vtla_stage2_4gpu
```

## 📈 训练监控

### 关键指标

**Stage 1:**
- `marker_loss`: Marker重建MSE (目标: <2.0)
- `rgb_loss`: RGB重建MSE (目标: <0.01)
- `total_loss`: 总损失 (应持续下降)

**Stage 2:**
- `l1`: 动作L1损失 (目标: <0.5)
- `kl`: KL散度 (健康范围: 5-20)
- `loss`: 总损失 (应持续下降)

**Stage 3:**
- `l1`: 主动作L1
- `main_l1`: 主路动作L1
- `contact`: 接触检测BCE损失
- `loss`: 总损失

### 日志示例

```
Epoch 1/500: 100%|████████| 10/10 [00:15<00:00, 1.5s/it, l1=1.234, kl=12.5, loss=13.7]
Checkpoint saved: checkpoints/stage2/stage2_epoch_50.ckpt
```

## 🔧 故障排查

### 问题1: OOM (显存不足)
```bash
# 解决方案: 降低batch size
# 编辑 start_training_multigpu.sh
BATCH_SIZE=2  # 从4降到2
```

### 问题2: 训练速度慢
```bash
# 检查数据加载瓶颈
# 增加num_workers (在脚本中修改)
--num_workers 8  # 从4增加到8
```

### 问题3: 多卡不同步
```bash
# 检查NCCL环境
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 如果InfiniBand有问题
```

### 问题4: Checkpoint加载失败
```bash
# 检查checkpoint路径
ls -lh checkpoints/stage1/*.ckpt
ls -lh checkpoints/stage2/*.ckpt

# 手动指定checkpoint
# 编辑start_training_multigpu.sh，添加:
--stage1_ckpt /path/to/specific/checkpoint.ckpt
```

## 💡 训练技巧

### 1. 渐进式训练
```bash
# 先用小数据集快速验证
# 修改NUM_EPOCHS测试流程
NUM_EPOCHS=10  # 在脚本中

# 验证通过后再进行完整训练
```

### 2. 继续训练
```bash
# 如果训练中断，从最新checkpoint继续
# Stage 2会自动加载最新的stage1 checkpoint
# 如需从中断处继续，手动指定checkpoint
```

### 3. 学习率调优
```bash
# 在train_vtla_multigpu.py中修改学习率
--lr 5e-5              # 降低主学习率
--lr_backbone 5e-6     # 降低backbone学习率
```

### 4. 多阶段checkpoint复用
```bash
# Stage 1完成后自动被Stage 2使用
# Stage 2完成后自动被Stage 3使用
# 脚本会自动查找最新的checkpoint
```

## 🎯 性能优化

### 多卡效率
- **理论加速**: 4x
- **实际加速**: 3.5-3.8x (考虑通信开销)
- **通信开销**: ~5-10% (DDP同步)

### Batch Size选择
- **Stage 1**: 8 per GPU (总32) - 内存友好
- **Stage 2**: 4 per GPU (总16) - 显存占用较大
- **Stage 3**: 4 per GPU (总16) - 显存占用中等

### 数据加载优化
- **num_workers**: 4-8 per GPU
- **pin_memory**: True (加速GPU传输)
- **persistent_workers**: True (避免重启开销)

## 📞 常用命令速查

```bash
# 启动4卡训练
./start_training_multigpu.sh stage2 4

# 查看训练状态
./check_training.sh

# 实时监控GPU
watch -n 1 nvidia-smi

# 查看实时日志
tail -f logs/stage2/train_4gpu_*.log

# Attach到训练会话
tmux attach -t vtla_stage2_4gpu

# 停止训练
tmux kill-session -t vtla_stage2_4gpu

# 列出所有checkpoints
ls -lht checkpoints/stage2/ | head -10
```

## ⏱️ 预计训练时间表

| Stage | 配置 | 预计时间 | 累计时间 |
|-------|------|---------|---------|
| Stage 1 | 4卡, 100 epochs | 2-3小时 | 2-3小时 |
| Stage 2 | 4卡, 500 epochs | 12-18小时 | 14-21小时 |
| Stage 3 | 4卡, 200 epochs | 4-6小时 | 18-27小时 |
| **总计** | | | **~1天** |

## 🎉 开始训练！

```bash
cd /home/rmc/workspace/VTLA_design

# 一键启动Stage 2 (4卡训练)
./start_training_multigpu.sh stage2 4
```

祝训练顺利！🚀
