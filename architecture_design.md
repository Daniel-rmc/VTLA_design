# VTLA (Vision-Tactile-Language-Action) 模型架构设计

## 1. 设计动机

在接触丰富的机器人操作任务中，特别是在最后1毫米的关键接触阶段：
- 视觉信息变化不明显，难以提供精确反馈
- 类比人类，触觉和末端力感知在精细操作中起主导作用
- 需要独立的触觉处理通路来捕捉微小的接触变化

## 2. 核心架构组件

```
输入层
├── RGB相机图像 → Vision Backbone (ResNet/ViT)
├── 触觉图像 → Tactile Encoder (专用)
└── 本体感觉 (qpos)

特征提取与融合层
├── Vision Tokens (来自vision backbone)
├── Tactile Tokens (来自tactile encoder)
└── Cross-Modal Fusion (视触觉交叉注意力)

VLA主干 (基于ACT/DETR)
├── Transformer Encoder (融合后的多模态token)
├── CVAE Latent (行为多样性)
└── Transformer Decoder (query-based解码)

动作生成层 (双路输出)
├── 主动作头 (融合视触觉信息)
└── 触觉微调分支 (纯触觉感知，用于接触后微调)
    └── 动作残差输出 (Δaction)
```

## 3. 关键设计特性

### 3.1 独立触觉编码器
- **目的**：不与视觉共享参数，专门学习触觉特征
- **架构**：ResNet18/34 + 自监督重建任务
- **输出**：触觉token序列 (latent embeddings)

### 3.2 视触觉交叉注意力
- **位置**：在进入VLA transformer之前
- **机制**：
  - Vision tokens作为Query
  - Tactile tokens作为Key和Value
  - 反向：Tactile tokens作为Query，Vision tokens作为Key/Value
  - 双向交叉注意力融合

### 3.3 触觉感知动作微调分支
- **触发条件**：检测到接触（通过触觉信号强度）
- **输入**：纯触觉特征（不含视觉）
- **输出**：动作残差 Δa
- **最终动作**：a_final = a_main + λ * Δa_tactile
  - λ是自适应权重，基于接触强度

## 4. 与UniVTAC的主要区别

| 特性 | UniVTAC | 新VTLA |
|------|---------|--------|
| 触觉编码器 | 共享或简单concat | 独立专用编码器 |
| 多模态融合 | 简单concat特征 | 交叉注意力token融合 |
| 动作生成 | 单一输出头 | 双路：主路+触觉微调 |
| 接触感知 | 隐式 | 显式触觉分支 |

## 5. 训练策略

### 5.1 三阶段训练
1. **Stage 1: 触觉编码器预训练**
   - 自监督重建任务
   - 数据：触觉图像 + marker/pose标签
   - 损失：重建损失 + 辅助任务损失

2. **Stage 2: 端到端VLA训练**
   - 固定或微调触觉编码器
   - 训练视觉backbone、融合模块、VLA主干
   - 损失：行为克隆 L1 + KL散度

3. **Stage 3: 触觉微调分支训练**
   - 固定主干网络
   - 仅训练触觉分支
   - 数据：接触丰富的轨迹段
   - 损失：残差动作L1损失

### 5.2 损失函数设计
```
L_total = L_main + α * L_tactile_refine + β * L_contact_aware

L_main = L1(a_main, a_gt) + w_kl * KL(q(z|τ) || p(z))
L_tactile_refine = L1(a_main + Δa_tac, a_gt) [仅在接触阶段]
L_contact_aware = BCE(contact_pred, contact_gt) [接触检测辅助任务]
```

## 6. 实现细节

### 6.1 超参数
- Tactile encoder: ResNet34, latent_dim=512
- Hidden dimension: 512
- Cross-attention heads: 8
- Tactile refine weight λ: 0.0-1.0 (adaptive based on contact)
- Loss weights: α=0.5, β=0.1

### 6.2 数据增强
- RGB图像：标准增强（随机裁剪、颜色抖动）
- 触觉图像：轻微增强（亮度、对比度，保持marker特征）
- 本体感觉：加噪声

### 6.3 接触检测
- 方法1：触觉图像强度阈值
- 方法2：学习的接触分类器（二分类头）
- 触发阈值：可调节，建议0.3-0.5

## 7. 预期优势

1. **精细操作性能**：触觉分支在接触后提供精确微调
2. **模态特异性**：独立编码器学习触觉特有特征
3. **可解释性**：可视化触觉分支的贡献
4. **鲁棒性**：视觉失效时触觉分支仍可工作

## 8. 后续扩展方向

- 加入力传感器数据融合
- 多指触觉（多个触觉传感器）
- 语言条件生成（加入language tokens）
- 在线适应（接触后实时微调）
