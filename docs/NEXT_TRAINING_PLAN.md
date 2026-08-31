# 下一轮训练改进计划

> 历史快照：本文记录 2026-08-13 dual-stream 分支当时的计划和未实施草案，不代表当前训练配置或运行状态。

**创建时间**: 2026-08-13
**状态**: 📋 计划阶段（等待当前训练完成）

---

## 🚀 当前训练状态

### 继续运行的原因
- ✅ 已经开始运行（Epoch 2/2000）
- ✅ 可以观察loss收敛行为
- ✅ 了解当前配置的实际效果
- ✅ 避免浪费已投入的计算

### 当前训练的价值
1. **收敛分析**: 观察loss下降速度和模式
2. **超参数实验**: 了解lr=1e-4的效果
3. **基准数据**: 为下一轮提供参考
4. **经验积累**: 理解双流架构的训练行为

### 何时停止当前训练？

#### 选项A: 早期停止（推荐）
**时机**: Epoch 11-22 (约4,000-8,000 steps)
- Epoch 11: 相当于ACT的4,000 steps
- Epoch 22: 充分训练（8,000 steps）

**判断标准**:
```bash
# 检查当前epoch
tmux capture-pane -t dual_stream_training -p | grep "Epoch"

# Epoch 11时评估
# 如果loss已经很低（<0.3），可以停止并评测
```

**停止命令**:
```bash
tmux send-keys -t dual_stream_training C-c
```

#### 选项B: 观察收敛
**时机**: Loss收敛后（可能在Epoch 50-100）
- 观察训练曲线
- 当loss不再下降时停止

**判断标准**:
```bash
# 每天检查一次
./scripts/analysis/check_training_status.sh

# 如果最近50 epochs的loss std < 0.02，说明已收敛
```

#### 选项C: 完整运行
**时机**: 完整2,000 epochs
- 了解极度过拟合的情况
- 作为对照实验

**注意**: 需要33小时，可能过拟合严重

---

## 🎯 下一轮训练配置

### 配置文件: `configs/dual_stream_act_aligned.json`

```json
{
  "description": "Dual-Stream VTLA aligned with ACT+UniVTAC baseline",

  "training": {
    "num_steps": 4000,
    "batch_size": 64,
    "lr": 1e-5,
    "lr_backbone": 1e-5,
    "lr_vision_backbone": 1e-5,
    "lr_tactile": 1e-5,
    "weight_decay": 1e-4,

    "save_freq": 500,
    "val_freq": 500,
    "print_freq": 100
  },

  "data": {
    "train_ratio": 0.9,
    "num_episodes": 50,
    "chunk_size": 50,
    "num_workers": 4
  },

  "model": {
    "state_dim": 9,
    "hidden_dim": 512,
    "enc_layers": 4,
    "dec_layers": 7,
    "nheads": 8,
    "dim_feedforward": 3200,
    "kl_weight": 10.0,

    "dual_stream_config": {
      "shared_encoder": false,
      "shared_decoder": false,
      "fusion_type": "gated",
      "enable_cross_stream": false
    }
  }
}
```

### 关键改动点

| 配置项 | 当前值 | 新值 | 原因 |
|--------|--------|------|------|
| num_epochs | 2000 | 移除 | 改用steps |
| num_steps | N/A | 4000 | 与ACT对齐 |
| batch_size | 32 | 64 | 与ACT对齐 |
| lr | 1e-4 | 1e-5 | 与ACT对齐 |
| dec_layers | 6 | 7 | 与ACT对齐 |
| train_ratio | 1.0 | 0.9 | 添加验证集 |
| val_freq | N/A | 500 | 监控过拟合 |

---

## 📝 需要修改的代码

### 1. 训练脚本: `scripts/training/train_dual_stream_v2.py`

#### 改动1: 使用steps而不是epochs
```python
# OLD
for epoch in range(1, args.num_epochs + 1):
    for batch in train_dataloader:
        # training code

# NEW
step = 0
max_steps = args.num_steps
epoch = 0

pbar = tqdm(range(max_steps), total=max_steps)
while step < max_steps:
    epoch += 1
    for batch in train_dataloader:
        if step >= max_steps:
            break

        # training code
        step += 1
        pbar.update(1)

        # Validation
        if step % args.val_freq == 0:
            validate()

        # Save checkpoint
        if step % args.save_freq == 0:
            save_checkpoint(step)
```

#### 改动2: 添加验证集split
```python
# 在_load_data()中添加
def _load_data(self):
    # 创建train dataloader (90%)
    train_dataloader = create_dataloader(
        self.args,
        stage='stage2',
        split='train',
        train_ratio=0.9
    )

    # 创建val dataloader (10%)
    val_dataloader = create_dataloader(
        self.args,
        stage='stage2',
        split='val',
        train_ratio=0.9
    )

    return train_dataloader, val_dataloader, stats
```

#### 改动3: 修改dataloader支持split
```python
# 在dataloader.py中
def create_dataloader(args, stage='stage2', split='train', train_ratio=0.9):
    dataset = VTLADataset(...)

    if split == 'train':
        indices = range(int(len(dataset) * train_ratio))
    else:  # val
        indices = range(int(len(dataset) * train_ratio), len(dataset))

    subset = Subset(dataset, indices)

    return DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=(split == 'train'),
        ...
    )
```

### 2. 启动脚本: `scripts/training/start_dual_stream_training_v2.sh`

```bash
#!/bin/bash

# ACT-aligned configuration
python scripts/training/train_dual_stream_v2.py \
    --task $TASK \
    --dataset-dir $DATASET_DIR \
    --output-dir ${PROJECT_ROOT}/runs/dual_stream_v2 \
    --device cuda \
    \
    --num-steps 4000 \
    --batch-size 64 \
    --train-ratio 0.9 \
    --val-freq 500 \
    --save-freq 500 \
    \
    --lr 1e-5 \
    --lr-backbone 1e-5 \
    --lr-vision-backbone 1e-5 \
    --lr-tactile 1e-5 \
    --weight-decay 1e-4 \
    \
    --state-dim 9 \
    --hidden-dim 512 \
    --enc-layers 4 \
    --dec-layers 7 \
    --chunk-size 50 \
    \
    --camera-names cam_high \
    --tactile-names tac_left tac_right \
    \
    --use-wandb
```

---

## 📊 预期训练时间

### 计算
```
数据: 11,700 samples
Train: 10,530 samples (90%)
Batch size: 64
Batches per epoch: 10,530 / 64 ≈ 165 batches

4000 steps / 165 batches = 24 epochs

假设速度: 6 it/s
4000 steps / 6 = 667秒 ≈ 11分钟

实际估计（考虑validation）: ~15-20分钟
```

### 对比
| 配置 | 训练时长 | Steps | 对比 |
|------|---------|-------|------|
| 当前 (全部) | 33小时 | 732,000 | 基准 |
| 当前 (Epoch 11) | 11分钟 | 4,000 | 1/180 |
| 新配置 (4000 steps) | 15-20分钟 | 4,000 | 快速 |
| 新配置 (8000 steps) | 30-40分钟 | 8,000 | 充分 |

---

## 🎯 实验计划

### 阶段1: 当前训练（进行中）
**目标**: 观察loss收敛行为，了解当前配置效果

**监控**:
```bash
# 每3小时检查一次
./scripts/analysis/check_training_status.sh
```

**决策点**: Epoch 11 (约4000 steps)
- 检查loss是否已收敛
- 如果收敛，可以停止
- 如果未收敛，继续到Epoch 22

**输出**:
- 训练曲线
- 收敛分析
- 经验总结

---

### 阶段2: ACT-aligned训练（下一轮）
**时机**: 当前训练停止后

**步骤**:
1. 修改代码（预计30分钟）
2. 测试新配置（smoke test）
3. 启动训练（15-20分钟）
4. 评测（8-10小时）
5. 对比结果

**配置**: 严格对齐ACT
- 4000 steps
- batch_size=64
- lr=1e-5
- 90/10 split
- validation every 500 steps

**目标**: 公平对比 vs ACT+UniVTAC (14%)

---

### 阶段3: 延长训练（可选）
**时机**: 如果阶段2成功（≥18%）

**步骤**:
1. 从4000 steps checkpoint继续
2. 训练到8000 steps
3. 降低学习率到5e-6
4. 再次评测

**目标**: 探索架构潜力上限

---

### 阶段4: 扩展到3任务
**时机**: 如果insert_HDMI成功

**任务**:
- put_bottle_in_shelf (8%)
- insert_HDMI (14%)
- lift_can (29%)

**目标**: 验证架构泛化性

---

## 📋 待办任务清单

### 代码修改（准备好，等待部署）
- [ ] 创建 `train_dual_stream_v2.py`（基于steps）
- [ ] 修改 dataloader 支持 train/val split
- [ ] 创建 `start_dual_stream_training_v2.sh`
- [ ] 创建 `configs/dual_stream_act_aligned.json`
- [ ] 更新 smoke test

### 文档
- [x] 配置对比分析 (TRAINING_CONFIG_ANALYSIS.md)
- [x] 下一轮改进计划 (NEXT_TRAINING_PLAN.md)
- [ ] 代码修改指南
- [ ] 实验对比报告模板

### 当前训练监控
- [ ] 每3小时检查loss
- [ ] Epoch 11时评估是否停止
- [ ] 记录收敛行为
- [ ] 生成训练曲线报告

---

## 🤔 关于当前训练的决策建议

### 建议1: Epoch 11停止（推荐）

**理由**:
- 相当于ACT的4000 steps
- 可以快速评测并获得初步结果
- 如果效果好，证明配置可行
- 如果效果差，也能快速迭代

**操作**:
```bash
# 当到达Epoch 11时
tmux attach -t dual_stream_training

# 手动停止（等当前epoch完成）
# 或在脚本中监控epoch数自动停止
```

### 建议2: Epoch 22停止

**理由**:
- 相当于8000 steps，更充分训练
- 能看到架构的真实潜力
- 但比完整2000 epochs快得多

### 建议3: 观察收敛后停止

**理由**:
- 数据驱动的决策
- 但可能需要较长时间

**判断**:
```python
# 如果最近100 batches的loss std < 0.02
# 说明已收敛，可以停止
```

---

## 💡 我的推荐

### 短期（今天-明天）
1. ✅ 让当前训练继续运行
2. ✅ 每3小时检查一次loss
3. ✅ 准备好v2版本的代码（明天完成）
4. ⚠️ **关键决策点**: Epoch 11 (约3小时后)
   - 如果loss已经<0.3，停止并评测
   - 否则继续到Epoch 22

### 中期（明天-后天）
1. 停止当前训练（在Epoch 11或22）
2. 部署v2配置（ACT-aligned）
3. 快速训练（15-20分钟）+ 评测（8-10小时）
4. 对比结果并决策下一步

### 长期（下周）
1. 如果成功，扩展到3任务
2. 如果失败，分析问题并调整
3. 准备论文材料

---

## 📞 监控和沟通

### 当前训练监控
```bash
# 快速检查
./scripts/analysis/check_training_status.sh

# 查看当前epoch
tmux capture-pane -t dual_stream_training -p | grep "Epoch" | tail -1

# 计算等效steps
# Epoch N → steps = N × 366
```

### 关键时间点通知
- **3小时后** (Epoch ~11): 检查是否停止
- **6小时后** (Epoch ~22): 第二次检查点
- **每天**: 查看训练报告

---

**当前状态**: 📋 计划完成，等待部署
**下一步**: 继续监控当前训练，准备v2代码
**目标**: 获得公平、高质量的对比结果
