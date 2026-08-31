# 训练配置对比分析与修正方案

> 历史快照：本文针对 2026-08-13 的 dual-stream 配置，不适用于后续 LeRobot 训练主线。

**日期**: 2026-08-13
**问题**: 当前训练配置与ACT+UniVTAC baseline不匹配

---

## 🚨 严重问题发现

### 当前配置的问题

#### 1. **训练长度严重过长** ⚠️⚠️⚠️
- **ACT baseline**: 4,000 steps
- **当前配置**: 2,000 epochs × 366 batches = **732,000 steps** (183倍!)
- **问题**: 严重过拟合风险，训练时间浪费

#### 2. **主网络学习率过高** ⚠️
- **ACT baseline**: lr = 1e-5
- **当前配置**: lr = 1e-4 (高10倍)
- **问题**: 可能导致训练不稳定

#### 3. **Batch size不匹配** ⚠️
- **ACT baseline**: 64
- **当前配置**: 32
- **问题**: 每个step看到的样本数不同

#### 4. **缺少验证集** ⚠️
- **ACT baseline**: 有train/val split
- **当前配置**: 只有train，没有validation
- **问题**: 无法监控过拟合，无法选择最佳checkpoint

#### 5. **Decoder层数不同**
- **ACT baseline**: 7 layers
- **当前配置**: 6 layers
- **影响**: 模型容量略小

---

## 📊 完整配置对比表

| 配置项 | ACT+UniVTAC | 当前双流VTLA | 问题严重性 |
|--------|-------------|--------------|-----------|
| **训练长度** | 4,000 steps | 732,000 steps | 🔴 严重 |
| **Batch size** | 64 | 32 | 🟡 中等 |
| **主LR** | 1e-5 | 1e-4 | 🟡 中等 |
| **Backbone LR** | 1e-5 | 1e-5 | ✅ 一致 |
| **Tactile LR** | 1e-5 | 1e-5 | ✅ 一致 |
| **Chunk size** | 50 | 50 | ✅ 一致 |
| **Hidden dim** | 512 | 512 | ✅ 一致 |
| **Enc layers** | 4 | 4 | ✅ 一致 |
| **Dec layers** | 7 | 6 | 🟡 中等 |
| **KL weight** | 10.0 | 10.0 | ✅ 一致 |
| **验证集** | 有 | 无 | 🔴 严重 |
| **Save freq** | 1000 steps | 100 epochs | 🟡 不同单位 |

---

## 💡 修正方案

### 方案A: 严格对齐ACT配置（推荐用于公平对比）

**目标**: 与baseline完全一致，确保公平对比

```python
# 修改训练脚本，使用steps而不是epochs
num_steps: 4000              # 与ACT一致
batch_size: 64               # 与ACT一致
lr: 1e-5                     # 与ACT一致
lr_backbone: 1e-5
lr_vision_backbone: 1e-5
lr_tactile: 1e-5
save_freq: 1000              # 每1000 steps保存

# Dataloader配置
train_ratio: 0.9             # 90% train, 10% val
num_episodes: 50             # 与ACT一致

# 模型配置
dec_layers: 7                # 改为7层，与ACT一致
chunk_size: 50
hidden_dim: 512
enc_layers: 4
kl_weight: 10.0
```

**预计训练时间**:
- 每个step: ~1.5秒 (batch_size=64)
- 4000 steps: ~1.7小时
- 比当前快20倍！

**优势**:
- ✅ 完全公平的对比
- ✅ 快速得到结果
- ✅ 避免过拟合

**劣势**:
- ⚠️ 可能没有充分发挥双流架构的潜力

---

### 方案B: 适度延长训练（推荐用于探索架构潜力）

**目标**: 在公平对比基础上，额外探索架构的极限能力

**第一阶段**: 严格对齐baseline
```python
num_steps: 4000
batch_size: 64
lr: 1e-5
# ... 其他与方案A相同
```

**第二阶段**: 延长训练（可选）
```python
num_steps: 8000              # 2倍训练
lr: 5e-6                     # 降低学习率
# 从4000 steps的checkpoint继续训练
```

**优势**:
- ✅ 既有公平对比（4000 steps）
- ✅ 又能看到充分训练的效果（8000 steps）
- ✅ 两组结果都可以报告

---

### 方案C: 当前配置分析（不推荐）

如果继续当前配置：

```
当前进度: Epoch 2/2000 (约732 steps)
相当于ACT的: 732/4000 = 18.3%

建议停止点:
- 公平对比: Epoch 11 (约4000 steps)
- 充分训练: Epoch 22 (约8000 steps)
- 极限训练: Epoch 55 (约20000 steps)
```

**问题**:
- ⚠️ Batch size和LR不一致
- ⚠️ 没有验证集
- ⚠️ 难以与baseline公平对比

---

## 🎯 推荐行动方案

### 立即行动（推荐方案A）

**1. 停止当前训练**
```bash
tmux send-keys -t dual_stream_training C-c
```

**2. 修改配置为steps-based训练**

需要修改训练脚本：
- 将`num_epochs`改为`num_steps`
- 添加train/val split
- 调整batch size到64
- 降低主LR到1e-5
- 改decoder为7层

**3. 重新启动训练**
```bash
# 预计1.7小时完成
./scripts/training/start_dual_stream_training.sh insert_HDMI 0
```

**4. 快速评测**
- 4000 steps后立即评测
- 对比ACT baseline (14%)

**5. 决策**
- 如果≥20%: 成功，可以写论文
- 如果15-20%: 继续训练到8000 steps
- 如果<15%: 分析问题

---

## 📋 关于验证集

### ACT的验证策略

根据ACT代码，验证集的作用：
1. **监控过拟合**: 每N steps在val set上评估
2. **Early stopping**: val loss不再下降时停止
3. **选择最佳checkpoint**: 保存val loss最低的模型

### 建议的验证配置

```python
# Dataloader
train_ratio: 0.9             # 90% train (45 episodes)
                             # 10% val (5 episodes)

# 验证频率
val_freq: 500                # 每500 steps验证一次
```

### 验证集的重要性

| 没有验证集 | 有验证集 |
|-----------|---------|
| ❌ 不知道何时停止 | ✅ 可以early stopping |
| ❌ 可能过拟合 | ✅ 监控泛化性能 |
| ❌ 只能用最后的ckpt | ✅ 选择最佳ckpt |
| ❌ 无法与ACT公平对比 | ✅ 使用相同的评估方式 |

---

## 🔍 数据量计算

### 50 episodes的数据分布

```
总样本数: 11,700 timesteps

方案A (batch_size=64, 90/10 split):
- Train: 10,530 samples → 165 batches/epoch
- Val: 1,170 samples → 19 batches
- 4000 steps ≈ 24 epochs

当前配置 (batch_size=32, 100% train):
- Train: 11,700 samples → 366 batches/epoch
- Val: 0
- 2000 epochs = 732,000 steps (过度训练!)
```

---

## 💭 训练停止时机

### ACT的停止策略

ACT使用固定的4000 steps，原因：
1. **经验值**: 足够收敛但不过拟合
2. **一致性**: 所有任务使用相同的训练长度
3. **效率**: 快速迭代实验

### 我们的停止策略建议

#### 公平对比实验
- **固定**: 4000 steps
- **原因**: 与baseline完全一致

#### 架构潜力探索
- **基准**: 4000 steps (公平对比)
- **延长**: 8000 steps (充分训练)
- **监控**: 使用val loss判断收敛

#### 判断收敛的指标
```python
# 最近100 steps的val loss统计
if val_loss_std < 0.01 and val_loss_trend >= 0:
    # 已收敛，可以停止
    early_stop = True
```

---

## 🎓 论文报告建议

### 主要结果（必须）
- **配置**: 严格对齐ACT (4000 steps, batch=64, lr=1e-5)
- **对比**: Dual-Stream vs ACT+UniVTAC

### 补充分析（可选）
- **消融实验**: 不同训练长度的影响
- **收敛分析**: 8000 steps vs 4000 steps
- **架构潜力**: 充分训练后的性能上限

---

## 📝 总结

### 当前问题
1. 🔴 训练太长（732k steps vs 4k steps）
2. 🟡 配置不匹配（batch size, LR, decoder layers）
3. 🔴 缺少验证集

### 推荐方案
✅ **停止当前训练**
✅ **采用方案A: 严格对齐ACT配置**
✅ **添加train/val split (90/10)**
✅ **4000 steps后评测**
✅ **可选：继续训练到8000 steps**

### 预期结果
- **训练时间**: 1.7小时 (vs 当前的33小时)
- **公平对比**: ✅
- **避免过拟合**: ✅
- **快速迭代**: ✅

---

**建议**: 立即停止当前训练，按方案A重新配置后启动新训练。

**原因**:
1. 当前只训练了732 steps (2 epochs)，相当于ACT的18%，还很早期
2. 配置不匹配会导致结果不公平
3. 重新训练只需1.7小时，比继续错误配置更高效

**下一步**: 是否需要我立即停止训练并重新配置？
