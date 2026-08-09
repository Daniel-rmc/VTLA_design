# VTLA训练策略详解

## 📋 目录
1. [三阶段训练概览](#三阶段训练概览)
2. [Stage 1: 触觉编码器预训练](#stage-1-触觉编码器预训练)
3. [Stage 2: 端到端VLA训练](#stage-2-端到端vla训练)
4. [Stage 3: 触觉微调分支训练](#stage-3-触觉微调分支训练)
5. [损失函数设计](#损失函数设计)
6. [超参数调优指南](#超参数调优指南)
7. [常见问题排查](#常见问题排查)

## 三阶段训练概览

```
┌─────────────────────────────────────────────────────────┐
│  Stage 1: 触觉编码器自监督预训练                          │
│  目标: 学习触觉特征表示                                   │
│  数据: 触觉图像 + marker/rgb/pose标签                    │
│  时长: ~200 epochs (约4-6小时, 单GPU)                    │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Stage 2: 端到端VLA训练                                  │
│  目标: 学习视触觉融合和动作生成                           │
│  数据: 完整演示轨迹 (qpos, images, actions)              │
│  时长: ~1000 epochs (约1-2天, 单GPU)                     │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  Stage 3: 触觉微调分支训练                               │
│  目标: 学习接触后的精细动作微调                           │
│  数据: 接触丰富的轨迹段                                   │
│  时长: ~200 epochs (约4-6小时, 单GPU)                    │
└─────────────────────────────────────────────────────────┘
```

## Stage 1: 触觉编码器预训练

### 目标
学习触觉传感器的良好特征表示，为后续VLA训练提供强初始化。

### 数据要求

```python
# 每个训练样本包含:
{
    'tactile_image': torch.Tensor,  # [C, H, W] 触觉图像 (通常256x256)
    'targets': {
        'marker': torch.Tensor,     # [63, 2] Marker位置坐标
        'rgb': torch.Tensor,        # [3, H, W] 原始RGB图像
        'pose': torch.Tensor,       # [7] 6D pose + contact flag
        # 可选:
        'depth': torch.Tensor,      # [1, H, W] 深度图
    }
}
```

### 自监督任务

**1. Marker重建 (最重要)**
- 任务：从触觉图像预测63个marker的2D坐标
- 损失：MSE(marker_pred, marker_gt)
- 权重：1.0
- 作用：学习marker变形模式，这是触觉最核心的信息

**2. RGB重建**
- 任务：重建触觉传感器的RGB图像
- 损失：MSE(rgb_recon, rgb_gt)
- 权重：0.5
- 作用：学习纹理和外观特征

**3. Pose回归 (可选)**
- 任务：回归接触物体的6D pose
- 损失：MSE(pose_pred, pose_gt)
- 权重：0.3
- 作用：学习几何信息

### 训练命令

```bash
python train_vtla.py \
    --stage stage1 \
    --dataset_dir /path/to/tactile_data \
    --tactile_names tac_left tac_right \
    --tactile_backbone resnet34 \
    --tactile_latent_dim 512 \
    --tactile_supervise marker rgb pose \
    --batch_size 32 \
    --num_epochs 200 \
    --lr_stage1 1e-4 \
    --weight_decay 1e-4 \
    --save_freq 50 \
    --ckpt_dir ./ckpt/stage1 \
    --device cuda:0
```

### 评估标准

**收敛标准**：
- Marker MSE < 2.0 (像素)
- RGB MSE < 0.01
- Pose MSE < 0.1

**可视化检查**：
```python
# 每50个epoch可视化重建结果
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0, 0].imshow(tactile_image)
axes[0, 1].imshow(rgb_gt)
axes[0, 2].imshow(rgb_recon)
axes[1, 0].scatter(marker_gt[:, 0], marker_gt[:, 1], c='r', label='GT')
axes[1, 1].scatter(marker_pred[:, 0], marker_pred[:, 1], c='b', label='Pred')
plt.savefig(f'stage1_epoch_{epoch}.png')
```

### 常见问题

**Q1: Marker重建不收敛？**
- 检查marker标注质量（可能有标注错误）
- 降低学习率至5e-5
- 增加batch size至64

**Q2: RGB重建模糊？**
- 正常现象，VAE类重建通常较模糊
- 关注marker重建质量更重要

**Q3: 需要预训练多久？**
- 最少100 epochs
- 推荐200 epochs
- 如果损失不再下降，可以提前停止

## Stage 2: 端到端VLA训练

### 目标
学习从视觉和触觉观察到机器人动作的端到端映射。

### 数据要求

```python
# 每个训练样本包含:
{
    'qpos': torch.Tensor,           # [state_dim] 机器人状态
    'cam_image': torch.Tensor,      # [N_cam, 3, H, W] 多相机图像
    'tac_image': torch.Tensor,      # [N_tac, 3, H, W] 多触觉图像
    'actions': torch.Tensor,        # [chunk_size, action_dim] 动作序列
    'is_pad': torch.BoolTensor,     # [chunk_size] padding mask
}

# 数据增强 (推荐):
- 相机图像: 随机裁剪、颜色抖动、随机旋转(±5°)
- 触觉图像: 轻微噪声、亮度调整
- 动作序列: 添加高斯噪声(std=0.01)
```

### 损失函数

```python
L_total = L_action + w_kl * L_kl

# 主动作L1损失
L_action = L1(a_pred, a_gt) [仅非padding位置]

# CVAE KL散度 (行为多样性)
L_kl = KL(q(z|τ) || p(z))
```

### 训练命令

```bash
python train_vtla.py \
    --stage stage2 \
    --dataset_dir /path/to/robot_data \
    --camera_names cam_high cam_left cam_right \
    --tactile_names tac_left tac_right \
    --state_dim 8 \
    --joint_indices 0 1 2 3 4 5 6 7 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --nheads 8 \
    --enc_layers 4 \
    --dec_layers 6 \
    --cross_attn_layers 2 \
    --batch_size 8 \
    --num_epochs 1000 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --lr_tactile 5e-5 \
    --kl_weight 10.0 \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    --stage1_ckpt ./ckpt/stage1/stage1_epoch_200.ckpt \
    --ckpt_dir ./ckpt/stage2 \
    --device cuda:0
```

### 学习率策略

**分层学习率**：
- **Vision Backbone**: 1e-5 (ImageNet预训练，微调即可)
- **Tactile Encoder**: 5e-5 (Stage 1预训练，轻微微调)
- **其他组件**: 1e-4 (从头训练，需要较大学习率)

**学习率衰减**：
```python
# 每200个epoch衰减0.5倍
scheduler = StepLR(optimizer, step_size=200, gamma=0.5)

# 或使用余弦退火
scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
```

### 评估标准

**训练监控**：
```python
# 每个epoch记录
metrics = {
    'l1_loss': float,           # 动作L1损失
    'kl_loss': float,           # KL散度
    'total_loss': float,        # 总损失
    'grad_norm': float,         # 梯度范数
}

# 每50个epoch评估
eval_metrics = {
    'success_rate': float,      # 任务成功率
    'action_error': float,      # 动作平均误差
    'contact_accuracy': float,  # 接触检测准确率
}
```

**收敛标准**：
- Action L1 < 0.5
- KL Loss稳定在 5-20之间
- Success Rate > 70% (任务相关)

### 训练技巧

**1. Warmup策略**
```python
# 前50个epoch使用较小的KL权重
if epoch < 50:
    kl_weight = 1.0
else:
    kl_weight = 10.0
```

**2. 梯度裁剪**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**3. 触觉编码器冻结 (可选)**
```python
# 如果触觉编码器已经充分预训练，可以冻结以稳定训练
for param in model.tactile_encoder.parameters():
    param.requires_grad = False
```

### 常见问题

**Q1: 训练不稳定？**
- 降低整体学习率至5e-5
- 增大梯度裁剪至0.5
- 使用warmup策略

**Q2: KL损失爆炸？**
- 降低kl_weight至5.0
- 检查动作标准化是否正确
- 使用KL annealing (逐渐增加权重)

**Q3: 动作L1不下降？**
- 检查数据标准化 (qpos, actions)
- 增加chunk_size至150
- 检查is_pad mask是否正确

## Stage 3: 触觉微调分支训练

### 目标
在接触阶段，利用触觉信息对动作进行精细微调。

### 数据要求

**数据筛选**：仅使用接触丰富的轨迹段
```python
def filter_contact_rich_segments(episode):
    """筛选接触丰富的轨迹段"""
    # 方法1: 基于触觉信号强度
    tactile_intensity = compute_tactile_intensity(episode['tac_image'])
    contact_mask = tactile_intensity > threshold
    
    # 方法2: 基于力传感器
    force = episode.get('force', None)
    if force is not None:
        contact_mask = force > force_threshold
    
    # 提取接触段
    contact_segments = extract_segments(episode, contact_mask)
    return contact_segments

# 建议阈值
threshold = 0.3  # 触觉信号强度归一化到[0,1]
force_threshold = 1.0  # 牛顿
```

### 损失函数

```python
L_total = L_main + α * L_refine + β * L_contact

# 主动作损失 (监督主路不要偏离太多)
L_main = L1(a_main, a_gt)

# 微调后动作损失 (最终动作应该更准确)
L_refine = L1(a_main + Δa_tactile, a_gt)

# 接触检测损失
L_contact = BCE(contact_pred, contact_gt)

# 权重推荐
α = 0.5   # refine_weight
β = 0.1   # contact_weight
```

### 训练命令

```bash
python train_vtla.py \
    --stage stage3 \
    --dataset_dir /path/to/contact_rich_data \
    --camera_names cam_high cam_left cam_right \
    --tactile_names tac_left tac_right \
    --batch_size 16 \
    --num_epochs 200 \
    --lr_stage3 5e-5 \
    --refine_weight 0.5 \
    --contact_weight 0.1 \
    --refine_scale 0.1 \
    --stage2_ckpt ./ckpt/stage2/stage2_epoch_1000.ckpt \
    --ckpt_dir ./ckpt/stage3 \
    --device cuda:0
```

### 评估标准

**微调效果评估**：
```python
# 对比主动作和微调后动作的误差
main_error = L1(a_main, a_gt)
refined_error = L1(a_main + Δa_tactile, a_gt)

# 微调应该有改进
improvement = (main_error - refined_error) / main_error * 100
print(f"Tactile refinement improves action by {improvement:.2f}%")

# 期望: improvement > 5%
```

**接触检测评估**：
```python
# 二分类指标
accuracy = (contact_pred == contact_gt).float().mean()
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)

# 期望: accuracy > 80%, F1 > 0.75
```

### 训练技巧

**1. 接触标签生成**
```python
# 如果没有GT接触标签，可以用启发式方法
def generate_contact_labels(episode):
    # 方法1: 触觉信号强度
    tactile_intensity = compute_intensity(episode['tac_image'])
    contact_gt = (tactile_intensity > 0.3).float()
    
    # 方法2: 动作变化率
    action_change = torch.diff(episode['actions'], dim=0)
    contact_gt = (action_change.norm(dim=-1) < threshold).float()
    
    # 方法3: 时间启发式 (后半段更可能接触)
    T = len(episode['actions'])
    contact_gt = torch.zeros(T)
    contact_gt[T//2:] = 1.0
    
    return contact_gt
```

**2. 残差范围控制**
```python
# Tanh限制残差在[-1, 1]，再乘以scale
Δa = torch.tanh(residual_net(tactile_features)) * refine_scale

# 推荐scale范围: [0.05, 0.2]
# - 0.05: 保守，微小调整
# - 0.1: 默认，适中
# - 0.2: 激进，较大调整
```

**3. 自适应权重调试**
```python
# 观察contact_prob和gate_weight的分布
print(f"Contact prob mean: {contact_prob.mean():.3f}")
print(f"Contact prob std: {contact_prob.std():.3f}")
print(f"Gate weight mean: {gate_weight.mean():.3f}")

# 健康范围:
# - contact_prob: 0.3-0.7 (不应该全是0或1)
# - gate_weight: 0.2-0.8 (有一定区分度)
```

### 常见问题

**Q1: 微调没有改进？**
- 增大refine_scale至0.15或0.2
- 检查接触标签质量
- 确保使用了接触丰富的数据

**Q2: 微调反而变差？**
- 降低refine_scale至0.05
- 增大refine_weight至1.0 (更强监督)
- 检查主干网络是否正确冻结

**Q3: 接触检测不准确？**
- 改进接触标签生成方法
- 增加contact_weight至0.2
- 使用focal loss处理类别不平衡

## 损失函数设计

### Stage 1损失

```python
L_stage1 = w_marker * L_marker + w_rgb * L_rgb + w_pose * L_pose

# 推荐权重
w_marker = 1.0   # Marker最重要
w_rgb = 0.5      # RGB次之
w_pose = 0.3     # Pose可选
```

### Stage 2损失

```python
L_stage2 = L_action + w_kl * L_kl

# L_action: 主动作L1损失（仅非padding）
L_action = (L1(a_pred, a_gt) * ~is_pad).sum() / (~is_pad).sum()

# L_kl: CVAE KL散度
L_kl = -0.5 * sum(1 + log(σ²) - μ² - σ²)

# 推荐权重
w_kl = 10.0  # 标准设置，保证行为多样性
```

### Stage 3损失

```python
L_stage3 = L_main + w_refine * L_refine + w_contact * L_contact

# L_main: 主动作应该接近GT
L_main = L1(a_main, a_gt)

# L_refine: 微调后动作应该更准确
L_refine = L1(a_final, a_gt)
where a_final = a_main + scale * gate * Δa_tactile

# L_contact: 接触检测BCE损失
L_contact = BCE(contact_pred, contact_gt)

# 推荐权重
w_refine = 0.5   # 微调监督
w_contact = 0.1  # 辅助任务
```

## 超参数调优指南

### 优先级排序

**高优先级**（对性能影响大）：
1. `kl_weight`: [5.0, 10.0, 20.0]
2. `lr`: [5e-5, 1e-4, 2e-4]
3. `refine_scale`: [0.05, 0.1, 0.15, 0.2]
4. `chunk_size`: [50, 100, 150]

**中优先级**：
5. `cross_attn_layers`: [1, 2, 4]
6. `refine_weight`: [0.3, 0.5, 0.7]
7. `lr_backbone`: [1e-5, 5e-5]

**低优先级**（通常不需要调整）：
8. `hidden_dim`: 512 (固定)
9. `nheads`: 8 (固定)
10. `dropout`: 0.1 (固定)

### 调优策略

**1. 网格搜索（资源充足）**
```bash
for kl in 5.0 10.0 20.0; do
  for lr in 5e-5 1e-4 2e-4; do
    for scale in 0.05 0.1 0.15; do
      python train_vtla.py \
        --kl_weight $kl \
        --lr $lr \
        --refine_scale $scale \
        --ckpt_dir ./ckpt/grid_${kl}_${lr}_${scale}
    done
  done
done
```

**2. 随机搜索（推荐）**
```python
import random

for trial in range(20):
    kl_weight = random.choice([5.0, 10.0, 20.0])
    lr = random.choice([5e-5, 1e-4, 2e-4])
    refine_scale = random.uniform(0.05, 0.2)
    # ... train with these hyperparams
```

**3. 贝叶斯优化（高级）**
```python
from ax import optimize

best_params, values, _, _ = optimize(
    parameters=[
        {"name": "kl_weight", "type": "range", "bounds": [5.0, 20.0]},
        {"name": "lr", "type": "range", "bounds": [1e-5, 5e-4], "log_scale": True},
        {"name": "refine_scale", "type": "range", "bounds": [0.05, 0.2]},
    ],
    evaluation_function=train_and_evaluate,
    total_trials=30,
)
```

## 常见问题排查

### 训练问题

| 问题 | 可能原因 | 解决方案 |
|-----|---------|---------|
| Loss爆炸 | 学习率过大 | 降低lr至5e-5，增加梯度裁剪 |
| Loss不下降 | 学习率过小 | 增大lr至2e-4，检查数据标准化 |
| 过拟合 | 数据量不足 | 增加数据增强，增大weight_decay |
| 欠拟合 | 模型容量不足 | 增大hidden_dim，增加cross_attn_layers |
| KL崩溃 | KL权重过小 | 增大kl_weight至15-20 |

### 评估问题

| 问题 | 可能原因 | 解决方案 |
|-----|---------|---------|
| 训练好推理差 | 过拟合 | Early stopping，数据增强 |
| 接触阶段失败 | 触觉微调不足 | 增大refine_scale，重新训练Stage 3 |
| 视觉失效时崩溃 | 过度依赖视觉 | 增加触觉dropout，强化触觉分支 |

### 调试技巧

**1. 可视化注意力权重**
```python
# 在cross_modal_fusion中
attn_weights = cross_attention.get_attention_weights()
plt.imshow(attn_weights[0].cpu().detach())
plt.title('Vision-Tactile Cross-Attention')
plt.colorbar()
plt.savefig('attention_vis.png')
```

**2. 监控触觉贡献**
```python
# 计算触觉残差的贡献
tactile_contribution = (scaled_residual.norm(dim=-1) / actions.norm(dim=-1)).mean()
print(f"Tactile contribution: {tactile_contribution:.2%}")

# 健康范围: 5%-20%
```

**3. 单元测试**
```bash
# 测试各个模块
python models/tactile_encoder.py
python models/cross_modal_fusion.py
python models/action_heads.py
python models/vtla_policy.py
```

## 总结

**训练时间估算**（单个V100 GPU）：
- Stage 1: ~4-6小时 (200 epochs, batch_size=32)
- Stage 2: ~24-48小时 (1000 epochs, batch_size=8)
- Stage 3: ~4-6小时 (200 epochs, batch_size=16)
- **总计**: ~2-3天

**内存占用估算**：
- Stage 1: ~6GB
- Stage 2: ~12GB
- Stage 3: ~10GB

**推荐训练流程**：
1. 充分预训练触觉编码器（Stage 1）
2. 端到端训练VLA主干（Stage 2）
3. 在接触丰富数据上微调触觉分支（Stage 3）
4. 消融实验验证各组件贡献
5. 超参数调优提升性能

祝训练顺利！🚀
