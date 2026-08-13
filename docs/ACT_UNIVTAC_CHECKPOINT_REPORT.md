# ACT+UniVTAC Checkpoint查找报告

**日期**: 2026-08-13  
**目的**: 确认UniVTAC官方checkpoint是否包含触觉编码器

---

## 发现总结

✅ **找到了ACT+UniVTAC的官方checkpoint**

所有8个任务都有完整的训练checkpoint：
```
/home/rmc/workspace/UniVTAC/policy/ACT/act_ckpt/act-<task>/demo-50/train_config/policy_last.ckpt
```

---

## Checkpoint结构分析

### 模型类型：ACT + UniVTAC Encoder（双backbone架构）

通过检查`act-insert_HDMI`的checkpoint，确认了以下结构：

#### 1. **两个独立的Backbone**

```python
model.backbones.0.0.body.*        # Backbone 0 - 视觉ResNet
model.backbones.1.backbone.*      # Backbone 1 - 触觉ResNet (UniVTAC encoder)
model.backbones.1.position_embedding.weight  # 触觉位置编码
```

#### 2. **两个Input Projection层**

```python
model.vision_input_proj.weight    # 视觉特征投影
model.tactile_input_proj.weight   # 触觉特征投影
```

#### 3. **Transformer主干**

- Encoder: 4层
- Decoder: 7层
- 符合论文配置

---

## 官方评测结果（Shipped Logs）

| 任务 | 成功率 | Checkpoint路径 |
|------|--------|----------------|
| **put_bottle_in_shelf** | 8% | ✅ 存在 |
| **insert_HDMI** | 14% | ✅ 存在 |
| **lift_can** | 29% | ✅ 存在 |
| insert_hole | 39% | ✅ 存在 |
| lift_bottle | 46% | ✅ 存在 |
| pull_out_key | 48% | ✅ 存在 |
| insert_tube | 64% | ✅ 存在 |
| grasp_classify | 100% | ✅ 存在 |

---

## 结论

### ✅ 可以直接使用现有checkpoint作为ACT+UniVTAC baseline

**优势**：
1. 官方训练的checkpoint，配置与论文对齐
2. 已有完整的评测结果（100 seeds）
3. 包含触觉编码器（UniVTAC的预训练encoder）
4. 无需重新训练，节省时间

**使用方式**：
- 在实验对比中，直接引用这些checkpoint的评测结果
- 成功率来源：各任务的`log.log`文件中的"Final Result"

---

## 对实验计划的影响

### 更新决策

**原计划（Q18）**:
- C. 检查是否已有现成的checkpoint → 如果没有，再决定是否重新训练

**实际情况**:
- ✅ **所有8个任务都有ACT+UniVTAC checkpoint**
- ✅ **无需重新训练ACT+UniVTAC**

### 实验对比baseline确定

| Baseline | 数据来源 | 状态 |
|----------|----------|------|
| **ACT (vision-only)** | 需要查找或使用论文数字 | ⚠️ 待确认 |
| **ACT + UniVTAC Encoder** | 官方checkpoint已存在 | ✅ 可用 |
| **Dual-Stream VTLA** | 需要训练 | 🔄 待执行 |

---

## 下一步行动

### 立即执行
1. ✅ ~~查找ACT+UniVTAC checkpoint~~ （已完成）
2. [ ] 查找纯ACT (vision-only) 的checkpoint
3. [ ] 确认3个优先任务的数据manifest是否存在
4. [ ] 验证官方encoder.pth位置

### 实验策略确认

**阶段1**: 单任务pilot（insert_HDMI）
- 训练：Dual-Stream VTLA
- 对比：
  - ACT+UniVTAC: **14%** (官方checkpoint结果)
  - Dual-Stream VTLA: 待测

**阶段2**: 3个困难任务
- 训练：Dual-Stream VTLA on put_bottle_in_shelf, insert_HDMI, lift_can
- 对比：
  - ACT+UniVTAC: **8%, 14%, 29%** (官方结果)
  - Dual-Stream VTLA: 待测

---

## 附录：Checkpoint详细信息

### insert_HDMI checkpoint

**文件**: `/home/rmc/workspace/UniVTAC/policy/ACT/act_ckpt/act-insert_HDMI/demo-50/train_config/policy_last.ckpt`

**评测结果**: `log.log`
```
[2026-03-22 23:44:36] Final Result: 14/100(14.00%) success.
```

**模型参数统计**:
- Vision backbone (backbones.0): ResNet-18
- Tactile backbone (backbones.1): ResNet-18 (UniVTAC encoder)
- Transformer encoder: 4 layers
- Transformer decoder: 7 layers
- CVAE encoder: 4 layers
- 总参数量: ~40M (估算)

**关键组件**:
- `model.vision_input_proj`: 视觉特征投影
- `model.tactile_input_proj`: 触觉特征投影
- `model.input_proj_robot_state`: 本体感觉投影
- `model.action_head`: 动作预测头
- `model.is_pad_head`: Padding预测头
