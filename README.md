# 双流VTLA (Dual-Stream Vision-Tactile-Language-Action)

面向接触丰富操作任务的双流视触觉动作策略。

---

## 🎯 核心理念

**保持视觉和触觉的模态独立性直到最终融合层**，让每个模态有独立的信息处理通路。

### 与原始VTLA的关键区别

| 维度 | 原始VTLA | 双流架构 |
|------|----------|----------|
| **融合时机** | 早期融合（特征提取后） | 晚期融合（action head层） |
| **独立性** | 交叉注意力导致模态混淆 | 完全独立到decoder输出 |
| **训练策略** | 三阶段分离训练 | 端到端联合训练 |
| **可解释性** | 难以量化模态贡献 | 可视化fusion权重 |

详细分析见 [架构批判文档](docs/architecture_critique.md)

---

## 📁 项目结构

```
VTLA_design/
├── models/                          # 模型实现
│   ├── dual_stream_transformer.py   # 双流Transformer
│   ├── fusion_action_head.py        # 融合动作头（4种策略）
│   └── dual_stream_vtla_policy.py   # 完整策略封装
│
├── scripts/                         # 训练和评测脚本
│   ├── training/
│   │   ├── train_dual_stream.py     # 训练主脚本
│   │   ├── start_dual_stream_training.sh  # 快速启动脚本
│   │   └── smoke_test_dual_stream.py      # 快速测试
│   └── evaluation/
│       └── run_univtac_eval.py      # UniVTAC评测
│
├── configs/                         # 配置文件
│   └── dual_stream_stage2.json      # 训练配置
│
├── univtac_adapter/                 # UniVTAC适配
│   └── deploy_dual_stream.yml       # 部署配置
│
├── data_manifests/                  # 数据清单
│   ├── put_bottle_in_shelf_official_first50.json
│   ├── insert_HDMI_official_first50.json
│   └── lift_can_official_first50.json
│
└── docs/                            # 文档
    ├── LAUNCH_CHECKLIST.md          # 🚀 启动清单
    ├── EXPERIMENT_PLAN.md           # 完整实验计划
    ├── STATUS_SUMMARY.md            # 快速状态总览
    ├── dual_stream_architecture.md  # 架构设计
    └── architecture_critique.md     # 原架构分析
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 使用UniVTAC环境
conda activate UniVTAC

# 确认GPU可用
nvidia-smi
```

### 2. 运行测试

```bash
cd /home/rmc/workspace/VTLA_design

# 快速smoke test（~2分钟）
python scripts/training/smoke_test_dual_stream.py
```

### 3. 启动训练

```bash
# 单任务pilot：insert_HDMI on GPU 1
./scripts/training/start_dual_stream_training.sh insert_HDMI 1
```

**训练参数**：
- 数据：前50条轨迹，80/20 split
- Batch size: 8
- Epochs: 2000
- 预计时间：8-12小时

### 4. UniVTAC评测

```bash
# 找到run目录
RUN_DIR=$(ls -td runs/dual_stream/dual_stream_insert_HDMI_* | head -1)

# 评测100 seeds
python scripts/evaluation/run_univtac_eval.py \
  --run-dir $RUN_DIR \
  --deploy-config univtac_adapter/deploy_dual_stream.yml \
  --gpu 1 \
  --total-num 100
```

**预计时间**：8-10小时

---

## 📊 实验目标

### 阶段1：单任务验证 (2-3天)

| 任务 | ACT+UniVTAC Baseline | 双流目标 |
|------|---------------------|---------|
| insert_HDMI | 14% | **≥20%** |

### 阶段2：困难任务验证 (5-7天)

| 任务 | Baseline | 目标 | 期望提升 |
|------|----------|------|----------|
| put_bottle_in_shelf | 8% | ≥13% | +5% |
| insert_HDMI | 14% | ≥20% | +6% |
| lift_can | 29% | ≥35% | +6% |

### 阶段3：全任务扩展 (1-2周)

在UniVTAC全部8个任务上验证双流架构的泛化性能。

### 阶段4：深度分析 (3-5天)

- Fusion type消融实验
- 模态权重可视化
- 特征独立性分析
- 接触阶段精度分析

---

## 🏗️ 架构设计

### 信息流

```
视觉输入 → Vision Encoder → Vision Decoder → Vision Features ──┐
                                                               ├→ Fusion Head → Final Actions
触觉输入 → Tactile Encoder → Tactile Decoder → Tactile Features ┘
```

**关键特性**：
- ✅ 完全独立的处理通路
- ✅ 晚期融合保持模态特异性
- ✅ 可学习的fusion权重
- ✅ 端到端训练

### 核心模块

#### 1. DualStreamTransformer
- 独立的vision/tactile encoder-decoder
- 可选的跨流注意力（中间层）
- 支持encoder/decoder参数共享

#### 2. FusionActionHead
4种融合策略：
- **Concat**: 简单拼接+MLP
- **Gated**: 自适应权重（推荐）
- **Cross-Attention**: 模态间注意力
- **MoE**: 混合专家路由

#### 3. ContactAwareRouting (可选)
根据接触状态动态调整模态权重。

---

## 📖 文档导航

### 开始使用
- **[启动清单](docs/LAUNCH_CHECKLIST.md)** - 从这里开始！
- **[快速总览](docs/STATUS_SUMMARY.md)** - 一页纸状态

### 实验规划
- **[实验计划](docs/EXPERIMENT_PLAN.md)** - 完整4阶段计划
- **[Baseline报告](docs/ACT_UNIVTAC_CHECKPOINT_REPORT.md)** - ACT+UniVTAC状态
- **[准备报告](docs/EXPERIMENT_READY_REPORT.md)** - 检查清单

### 架构设计
- **[双流架构](docs/dual_stream_architecture.md)** - 设计理念和实现
- **[架构批判](docs/architecture_critique.md)** - 原VTLA的6个设计缺陷

---

## 🔬 为什么需要双流架构？

### 原始VTLA的问题

```python
# ❌ 早期融合破坏独立性
fused_vision, fused_tactile = CrossModalFusion(vision, tactile)
# fused_tactile已经包含视觉信息！

tactile_refine_head(fused_tactile)  
# 虚假的"纯触觉"分支
```

**结果**：
- 触觉特征被视觉信息"污染"
- 无法独立评估每个模态的贡献
- 三阶段训练缺乏联合优化

### 双流架构的解决方案

```python
# ✅ 保持独立直到最后
vision_output = vision_encoder → vision_decoder
tactile_output = tactile_encoder → tactile_decoder

final_actions = fusion_head(vision_output, tactile_output)
```

**优势**：
- ✅ 真正的模态独立性
- ✅ 可解释的fusion权重
- ✅ 端到端训练
- ✅ 易于消融实验

---

## 📈 预期结果

### 成功标准

| 提升幅度 | 判断 | 后续行动 |
|----------|------|----------|
| >10%绝对提升 | 🎉 显著成功 | 发表论文 |
| 5-10%绝对提升 | ✅ 成功 | 继续优化 |
| 2-5%绝对提升 | ⚠️ 边际改进 | 分析原因 |
| <2%绝对提升 | ❌ 无显著差异 | 重新设计 |

### 假设验证

1. **模态独立性**：vision和tactile特征低相关性（<0.5）
2. **接触感知**：接触阶段触觉权重增加
3. **训练效率**：端到端训练优于分阶段

---

## 💻 技术栈

- **框架**: PyTorch 2.0+
- **环境**: UniVTAC conda环境
- **硬件**: NVIDIA L40S GPU
- **评测**: IsaacSim (UniVTAC官方评测)

---

## 🤝 贡献

本项目是研究原型，用于验证双流架构在接触丰富操作任务中的有效性。

### 相关工作
- **UniVTAC**: [GitHub](https://github.com/univtac/UniVTAC) | [论文](https://arxiv.org/abs/2602.10093)
- **ACT**: Action Chunking with Transformers
- **ViTAL**: Vision-Tactile Agent Learning (UniVTAC变体)

---

## 📄 许可

本项目用于学术研究。

---

## 📞 联系

详细文档请参考 `docs/` 目录。

---

**准备好了吗？开始实验！** 🚀

```bash
# 从这里开始
cat docs/LAUNCH_CHECKLIST.md
```
