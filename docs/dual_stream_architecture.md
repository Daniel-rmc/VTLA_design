# 双流架构设计文档 (Dual-Stream Architecture)

## 设计理念

### 核心思想
保持视觉和触觉的**模态独立性**直到最终的动作生成层，让每个模态有独立的信息处理通路。

### 与原架构的关键区别

| 维度 | 原架构 | 双流架构 |
|------|--------|----------|
| 融合时机 | 早期融合（特征提取后立即交叉注意力） | 晚期融合（在action head层面） |
| Encoder | 单一混合encoder | 双独立encoder（可选共享参数） |
| Decoder | 单一decoder | 双独立decoder |
| Memory | 混合的vision+tactile tokens | 分离的vision memory和tactile memory |
| Action Head | 主路+伪独立触觉微调 | 真正独立的双流融合 |

## 架构设计

### 整体信息流

```
输入
├── RGB Images → Vision Backbone → Vision Tokens
├── Tactile Images → Tactile Encoder → Tactile Tokens
└── Robot State (qpos)

独立编码
├── Vision Tokens → Vision Encoder → Vision Memory
└── Tactile Tokens → Tactile Encoder → Tactile Memory

独立解码
├── Vision Memory + Queries → Vision Decoder → Vision Action Features
└── Tactile Memory + Queries → Tactile Decoder → Tactile Action Features

晚期融合
└── Fusion Action Head(Vision Features, Tactile Features) → Final Actions
```

### 关键模块

#### 1. DualStreamTransformer

```python
class DualStreamTransformer(nn.Module):
    """
    双流Transformer：维护两个独立的encoder-decoder路径
    
    参数：
    - shared_encoder: 是否共享encoder参数（节省参数量）
    - shared_decoder: 是否共享decoder参数
    - cross_stream_layers: 在哪些层进行跨流交互（可选）
    """
```

#### 2. FusionActionHead

```python
class FusionActionHead(nn.Module):
    """
    融合动作头：智能地融合视觉和触觉特征
    
    融合策略：
    - Concat + MLP: 简单拼接后投影
    - Gated Fusion: 学习自适应权重
    - Cross-Attention: 让一个模态attend到另一个
    - MoE: 混合专家路由
    """
```

#### 3. ContactAwareRouting (可选)

```python
class ContactAwareRouting(nn.Module):
    """
    接触感知路由：根据接触状态动态调整模态权重
    
    非接触阶段：视觉权重高（粗粒度规划）
    接触阶段：触觉权重高（精细调整）
    """
```

## 实现策略

### 阶段1：基础双流实现（当前）
- 实现独立的vision/tactile encoder和decoder
- 简单的gated fusion action head
- 保持与原有训练脚本兼容

### 阶段2：接触感知增强
- 添加接触检测模块
- 实现动态权重调整
- 添加接触阶段的特殊监督

### 阶段3：跨流交互（可选）
- 在decoder的中间层添加可选的跨流注意力
- 平衡独立性和交互性

## 训练策略

### 推荐：端到端训练
```python
loss = loss_action + w_kl * loss_kl + w_pad * loss_pad
```

- 两个流联合优化
- 自然地学习模态分工
- 无需手工设计分阶段策略

### 可选：分流预训练
如果效果不好，可以尝试：
1. 冻结触觉流，只训练视觉流
2. 冻结视觉流，只训练触觉流
3. 联合微调

### 损失函数设计

```python
# 主损失：最终动作与GT的L1
loss_action = L1(final_actions, actions_gt)

# 可选：监督中间输出
loss_vision = L1(vision_actions, actions_gt)  # 视觉流单独预测
loss_tactile = L1(tactile_actions, actions_gt)  # 触觉流单独预测

# 总损失
loss_total = loss_action + α * (loss_vision + loss_tactile)
```

## 预期优势

1. **理论清晰**：符合多模态学习的独立-融合范式
2. **可解释性**：可以单独评估每个模态的贡献
3. **鲁棒性**：一个模态失效时另一个仍能工作
4. **消融友好**：容易做模态消融实验

## 参数量分析

假设原始hidden_dim=512：

| 组件 | 原架构 | 双流架构（独立） | 双流架构（共享encoder） |
|------|--------|------------------|------------------------|
| Encoder | 1x | 2x | 1x |
| Decoder | 1x | 2x | 2x |
| Action Head | 1.2x | 1.5x | 1.5x |
| **总增量** | - | ~2x | ~1.5x |

**建议**：先尝试共享encoder参数，如果效果不够好再使用完全独立的encoder。

## 与现有代码的兼容性

### 保留的组件
- Vision Backbone (ResNet18)
- Tactile Encoder (自定义ResNet)
- CVAE模块（加入两个latent，每个流一个）

### 替换的组件
- CrossModalFusion → 删除（或移到decoder中间层）
- 单一Transformer → DualStreamTransformer
- DualPathActionHead → FusionActionHead

### 配置兼容
新增配置项：
```json
{
  "dual_stream": {
    "shared_encoder": true,
    "shared_decoder": false,
    "fusion_type": "gated",  // concat|gated|cross_attn|moe
    "enable_cross_stream": false,
    "cross_stream_layers": []
  }
}
```

## 下一步

1. 实现 `DualStreamTransformer`
2. 实现 `FusionActionHead`
3. 修改 `VTLAModel` 以使用双流架构
4. 更新训练脚本以支持新配置
5. 在 grasp_classify 任务上验证

## 参考文献

- Two-Stream CNNs for Action Recognition (Simonyan & Zisserman, 2014)
- ViLT: Vision-and-Language Transformer Without Convolution (Kim et al., 2021)
- Perceiver: General Perception with Iterative Attention (Jaegle et al., 2021)
