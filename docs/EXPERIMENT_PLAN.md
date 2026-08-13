# 双流VTLA实验验证计划

**日期**: 2026-08-13  
**目标**: 验证双流架构相对于现有baselines的性能提升  
**预计周期**: 2-3周

---

## 一、实验目标

### 核心假设
双流架构通过保持视觉和触觉的**模态独立性**直到最终融合层，能够在接触丰富的操作任务中：
1. 提升整体成功率（特别是困难任务）
2. 在接触阶段更好地利用触觉信息
3. 具有更好的可解释性（可视化模态权重）

### 验证方法
在UniVTAC官方任务上对比：
- **Baseline 1**: ACT (vision-only)
- **Baseline 2**: ACT + UniVTAC Encoder (vision + tactile, 原论文方法)
- **新方法**: Dual-Stream VTLA (vision + tactile, 双流架构)

---

## 二、已确定的实验设置

### 2.1 任务范围

| 决策项 | 选择 | 理由 |
|--------|------|------|
| **总任务数** | 全部8个UniVTAC任务 | 完整评估 |
| **数据规模** | 前50条轨迹/任务 | 与论文对齐 |
| **训练/验证切分** | Seed 1, 80/20 split | 与论文对齐 |
| **评测指标** | UniVTAC闭环成功率（100 seeds） | 最终性能指标 |

### 2.2 优先任务选择

**阶段1验证任务**（按成功率从低到高）：

| 任务 | Shipped Log成功率 | 选择原因 |
|------|------------------|----------|
| **put_bottle_in_shelf** | 8% | 最困难任务，最大改进空间 |
| **insert_HDMI** | 14% | 精细插入任务，高触觉依赖 |
| **lift_can** | 29% | 力控任务，需要触觉反馈 |

**理由**: 
- 这3个任务成功率最低，改进空间最大
- 覆盖不同类型操作：放置、插入、力控
- 如果在困难任务上有提升，说服力更强
- grasp_classify (100%) 太简单，跳过

**剩余任务**（阶段2扩展）：
- insert_hole (39%)
- lift_bottle (46%)
- pull_out_key (48%)
- insert_tube (64%)
- grasp_classify (100%) - 可选，用于完整性

### 2.3 Baseline对比

| Baseline | 描述 | 数据来源 | 状态 |
|----------|------|----------|------|
| **ACT** | Vision-only | 论文Table I或需查找checkpoint | ⚠️ 待确认 |
| **ACT + UniVTAC Encoder** | Vision + Tactile (原论文双backbone) | ✅ **官方checkpoint已找到** | ✅ 可用 |
| **ViTAL** | CLIP预训练编码器 | **跳过**（当前不可用，需要CLIP模型） | ❌ 跳过 |
| **Dual-Stream VTLA** | 本次实验新方法 | 需要训练 | 🔄 待训练 |

**ACT+UniVTAC官方结果**（来自shipped logs）：
- put_bottle_in_shelf: **8%**
- insert_HDMI: **14%**
- lift_can: **29%**
- insert_hole: 39%
- lift_bottle: 46%
- pull_out_key: 48%
- insert_tube: 64%
- grasp_classify: 100%

### 2.4 训练配置对齐

**必须一致的项目**（保证公平对比）：
```json
{
  "data": {
    "episodes": "前50条（episode 0-49）",
    "split_seed": 1,
    "train_ratio": 0.8
  },
  "optimization": {
    "max_steps": 4000,
    "batch_size": 64,
    "lr": 1e-5,
    "lr_backbone": 1e-5,
    "weight_decay": 1e-4,
    "optimizer": "AdamW",
    "scheduler": "none"
  },
  "model_shared": {
    "state_dim": 8,
    "chunk_size": 50,
    "hidden_dim": 512,
    "nheads": 8,
    "dim_feedforward": 3200,
    "enc_layers": 4,
    "dec_layers": 7,
    "kl_weight": 10,
    "latent_dim": 32,
    "backbone": "resnet18",
    "tactile_encoder": "使用官方encoder.pth预训练权重"
  },
  "evaluation": {
    "test_seeds": "100个固定seeds",
    "seed_range": "1000000-1000099"
  }
}
```

**架构特定配置**（Dual-Stream VTLA独有）：
```json
{
  "dual_stream": {
    "shared_encoder": true,
    "shared_decoder": false,
    "enable_cross_stream": false,
    "cross_stream_layers": [],
    "fusion_type": "gated",
    "use_contact_routing": false,
    "use_cvae": true
  }
}
```

### 2.5 不控制的差异

| 项目 | 说明 |
|------|------|
| **参数量** | 架构差异导致的参数量变化是设计的一部分，会记录但不强制匹配 |
| **Transformer结构** | 双流使用独立encoder/decoder，ACT使用单流 |
| **Fusion方式** | 双流使用gated fusion，ACT+UniVTAC可能使用concat |

### 2.6 随机性控制

| 阶段 | 策略 |
|------|------|
| **训练** | 固定seed（保证可复现） |
| **评测** | 100个不同seeds（评估鲁棒性） |
| **多次训练** | 资源受限，不做多seed训练 |

---

## 三、实验执行计划

### 阶段0：准备工作（1天）

**任务清单**：
- [x] 完成双流架构实现
- [x] 完成配置文件
- [x] **查找ACT+UniVTAC的现有checkpoint** ✅
  - ✅ 找到所有8个任务的官方checkpoint
  - ✅ 确认包含双backbone（vision + tactile）
  - ✅ 验证有完整评测结果（shipped logs）
  - 📄 详细报告：[ACT_UNIVTAC_CHECKPOINT_REPORT.md](ACT_UNIVTAC_CHECKPOINT_REPORT.md)
- [ ] 准备数据manifest（3个优先任务）
- [ ] 验证官方encoder.pth可用
- [ ] 准备训练启动脚本

**产出**：
- ✅ ACT+UniVTAC checkpoint状态报告（已完成）
- [ ] 3个任务的数据manifest
- [ ] 训练脚本就绪

**关键发现**：
- ✅ **所有8个任务都有ACT+UniVTAC的官方checkpoint**
- ✅ **无需重新训练ACT+UniVTAC baseline**
- ✅ **可以直接使用官方评测结果作为对比**

---

### 阶段1：单任务快速验证（2-3天）

**目标**: 在1个代表性任务上验证双流架构的有效性

**选择任务**: `insert_HDMI` (14%基线，精细插入任务)

**步骤**：

1. **训练Dual-Stream VTLA**
   ```bash
   CUDA_VISIBLE_DEVICES=1 python scripts/training/train_vtla.py \
     --config configs/dual_stream_stage2.json \
     --task insert_HDMI \
     --dataset-dir /home/rmc/workspace/UniVTAC/data/official/insert_HDMI/clean \
     --manifest data_manifests/insert_HDMI_official_first50.json \
     --output runs/stage2/dual_stream_insert_HDMI_pilot
   ```
   - 预计时间：8-12小时（4000 steps）
   - GPU：1张L40S

2. **评测100 seeds**
   ```bash
   python scripts/evaluation/run_univtac_eval.py \
     --run-dir runs/stage2/dual_stream_insert_HDMI_pilot \
     --deploy-config univtac_adapter/deploy_dual_stream.yml \
     --gpu 1 --total-num 100
   ```
   - 预计时间：8-10小时
   - GPU：1张L40S

3. **对比baseline**
   - ACT: 15%（论文Table I）
   - ACT+UniVTAC: 28%（论文Table I）或从checkpoint评测
   - Dual-Stream VTLA: 待测

4. **决策点**
   - ✅ **如果 >= 28%**: 继续阶段2
   - ⚠️ **如果 20-28%**: 分析原因，考虑调整fusion_type
   - ❌ **如果 < 20%**: 回到架构设计，检查实现

**产出**：
- 单任务成功率对比
- 初步可行性判断
- 进入阶段2的决策

---

### 阶段2：优先任务集验证（5-7天）

**目标**: 在3个最困难任务上验证双流架构

**任务并行训练**（GPU 1-3并行）：
```bash
# GPU 1: put_bottle_in_shelf
CUDA_VISIBLE_DEVICES=1 python scripts/training/train_vtla.py \
  --config configs/dual_stream_stage2.json \
  --task put_bottle_in_shelf \
  ... &

# GPU 2: insert_HDMI (如果阶段1已完成，跳过)
CUDA_VISIBLE_DEVICES=2 python scripts/training/train_vtla.py \
  --config configs/dual_stream_stage2.json \
  --task insert_HDMI \
  ... &

# GPU 3: lift_can
CUDA_VISIBLE_DEVICES=3 python scripts/training/train_vtla.py \
  --config configs/dual_stream_stage2.json \
  --task lift_can \
  ... &
```

**时间估算**：
- 训练：8-12小时/任务（并行）
- 评测：8-10小时/任务（串行或并行）
- 分析：1-2天

**产出**：
- 3个任务的完整对比结果
- 成功率提升分析
- 进入阶段3的决策

---

### 阶段3：全任务扩展（1-2周）

**条件**: 阶段2显示显著提升（>5%绝对提升或>20%相对提升）

**任务**: 剩余5个任务（或跳过grasp_classify，只做4个）

**执行方式**: 与阶段2相同，3GPU并行分批训练

**产出**：
- 8个任务（或7个）的完整对比
- 平均成功率提升
- 不同难度任务的改进幅度分析

---

### 阶段4：消融和分析实验（3-5天）

**目标**: 深入理解双流架构的工作机制

#### 4.1 Fusion Type消融

在insert_HDMI任务上对比不同fusion策略：
- [x] Baseline: gated (阶段1已完成)
- [ ] concat
- [ ] cross_attn
- [ ] moe

**问题**: 哪种融合策略最适合接触操作任务？

#### 4.2 模态权重可视化

**分析内容**：
- 提取每个timestep的fusion权重
- 绘制时序图：vision_weight vs tactile_weight
- 标注接触发生的时刻

**预期观察**：
- 非接触阶段：vision权重高（粗粒度运动规划）
- 接触阶段：tactile权重高（精细力控）

**代码**：
```python
# 从训练好的模型提取权重
components = model(qpos, cam_img, tac_img, return_components=True)
weights = components['fusion_weights']  # [B, T, 2]
```

#### 4.3 特征独立性分析

**问题**: 双流架构是否真正保持了模态独立性？

**分析方法**：
```python
# 计算vision和tactile decoder输出的相关性
vision_feat = components['vision_decoder_output']  # [B, T, D]
tactile_feat = components['tactile_decoder_output']  # [B, T, D]

correlation = cosine_similarity(vision_feat, tactile_feat)
```

**预期结果**：
- 双流架构：低相关性（<0.5）
- 原始VTLA（early fusion）：高相关性（>0.8）

#### 4.4 接触阶段动作精度分析

**问题**: 触觉信息是否真的在接触阶段发挥了作用？

**分析方法**：
- 标注轨迹的接触前/接触后阶段
- 分别计算两个阶段的动作L1 loss
- 对比双流 vs ACT vs ACT+UniVTAC

**预期结果**：
- 双流在接触后阶段L1显著降低

#### 4.5 模态消融

**实验**：在推理时手动调整fusion权重
```python
# 纯视觉模式
fusion_weights = torch.tensor([[1.0, 0.0]])  # [vision=1, tactile=0]

# 纯触觉模式
fusion_weights = torch.tensor([[0.0, 1.0]])  # [vision=0, tactile=1]
```

**问题**: 每个模态单独能达到什么性能？

**产出**：
- Fusion type消融结果表
- 模态权重时序可视化图
- 特征相关性分析报告
- 接触阶段精度分析
- 模态消融结果

---

## 四、预期结果与判断标准

### 4.1 成功标准

**主要指标**：UniVTAC闭环成功率

| 提升幅度 | 判断 | 后续行动 |
|----------|------|----------|
| **>10%绝对提升** | 🎉 显著成功 | 发表论文，全任务验证 |
| **5-10%绝对提升** | ✅ 成功 | 继续优化，撰写论文 |
| **2-5%绝对提升** | ⚠️ 边际改进 | 分析是否有其他优势（可解释性等） |
| **<2%绝对提升** | ❌ 无显著差异 | 重新审视架构设计 |

**次要指标**：
- 模态权重与接触状态的相关性（可解释性）
- 特征独立性（设计理念验证）
- 接触阶段动作精度提升

### 4.2 不同任务的预期

| 任务 | 当前最佳 | 触觉依赖程度 | 预期提升 |
|------|----------|--------------|----------|
| put_bottle_in_shelf | 8% (ACT+UniVTAC) | 高（放置需要力反馈） | **大** (>10%) |
| insert_HDMI | 28% (ACT+UniVTAC) | 高（精密插入） | **大** (>5%) |
| lift_can | 29% (ACT+UniVTAC) | 中（力控） | **中** (3-8%) |
| insert_hole | 39% | 中 | 中 (3-8%) |
| lift_bottle | 46% | 中 | 中 (2-5%) |
| pull_out_key | 48% | 低 | 小 (<5%) |
| insert_tube | 64% (ACT+UniVTAC) | 中 | 中 (2-5%) |
| grasp_classify | 100% | 低（已饱和） | 无 |

**假设**: 双流架构在高触觉依赖任务上提升更明显

---

## 五、资源需求

### 5.1 计算资源

**GPU**: 4 × NVIDIA L40S
- GPU 0: 保留/其他任务
- GPU 1-3: 实验任务

**时间估算**（单任务）：
- 训练：8-12小时（4000 steps）
- 评测：8-10小时（100 seeds）
- **总计**: ~20小时/任务

**阶段时间**：
- 阶段0（准备）：1天
- 阶段1（单任务pilot）：2-3天
- 阶段2（3任务）：5-7天（并行）
- 阶段3（5任务）：1-2周（并行）
- 阶段4（消融分析）：3-5天
- **总计**: 2-3周

### 5.2 存储空间

**每个任务**：
- Checkpoints: ~5GB（4个checkpoint + best + last）
- 训练日志: ~100MB
- 评测结果: ~1GB（100 episodes的视频和数据）
- **小计**: ~6GB/任务

**总存储**（8任务 + 消融）：~80-100GB

### 5.3 人力投入

- 实验执行和监控：每天2-3小时
- 结果分析和可视化：3-5天
- 论文撰写（如果成功）：1-2周

---

## 六、实验记录规范

### 6.1 每次训练记录

每个训练run保存在：
```
runs/stage2/dual_stream_<task>_<timestamp>_<git>/
├── config.json              # 完整配置
├── protocol.json            # 训练协议
├── dataset_manifest.json    # 数据清单
├── launch_command.sh        # 启动命令
├── train.log               # 训练日志
├── metrics.jsonl           # 逐epoch指标
├── checkpoints/            # 模型checkpoint
│   ├── stage2_step_1000.ckpt
│   ├── stage2_step_2000.ckpt
│   ├── stage2_step_3000.ckpt
│   ├── stage2_step_4000.ckpt
│   ├── stage2_best.ckpt
│   └── stage2_last.ckpt
└── exit_code               # 退出状态
```

### 6.2 每次评测记录

```
runs/stage2/dual_stream_<task>_<timestamp>_<git>/eval/univtac/
├── aggregate_result.json    # 聚合结果
├── deploy.yml              # 部署配置副本
├── task_config.yml         # 任务配置副本
├── 20260813_140000/        # 单次评测
│   ├── request.json
│   ├── result.json
│   ├── stdout.log
│   └── exit_code
└── per_seed_results/       # 详细结果
    ├── seed_1000000.json
    ├── seed_1000001.json
    └── ...
```

### 6.3 分析结果记录

```
runs/stage2/dual_stream_<task>_<timestamp>_<git>/analysis/
├── fusion_weights_visualization.png
├── feature_correlation_analysis.json
├── contact_phase_accuracy.json
├── modality_ablation_results.json
└── analysis_report.md
```

---

## 七、风险与应对

### 7.1 训练风险

| 风险 | 可能性 | 影响 | 应对措施 |
|------|--------|------|----------|
| 训练不收敛 | 中 | 高 | 检查梯度、学习率、初始化 |
| GPU OOM | 低 | 中 | 减小batch size或hidden_dim |
| 训练中断 | 中 | 低 | 使用checkpoint恢复 |

### 7.2 性能风险

| 风险 | 可能性 | 影响 | 应对措施 |
|------|--------|------|----------|
| 无显著提升 | 中 | 高 | 尝试cross-stream attention，调整fusion type |
| 某些任务退化 | 中 | 中 | 分析原因，可能需要task-specific配置 |
| 评测不稳定 | 低 | 中 | 增加test seeds数量 |

### 7.3 时间风险

| 风险 | 可能性 | 影响 | 应对措施 |
|------|--------|------|----------|
| 训练比预期慢 | 中 | 中 | 优化数据加载，减少logging |
| 评测比预期慢 | 中 | 中 | 并行评测多个seeds |
| 分析耗时 | 高 | 低 | 优先核心分析，次要分析可选 |

---

## 八、检查点与决策节点

### Checkpoint 1: 准备完成（Day 1）
- [ ] ACT+UniVTAC checkpoint查找完成
- [ ] 数据manifest准备完成
- [ ] 训练脚本测试通过
- **决策**: 是否开始阶段1训练

### Checkpoint 2: 单任务验证（Day 3-4）
- [ ] insert_HDMI训练完成
- [ ] 100 seeds评测完成
- [ ] 成功率与baseline对比完成
- **决策**: 
  - 如果成功率 >= 28%：进入阶段2
  - 如果成功率 20-28%：分析并调整后再进入阶段2
  - 如果成功率 < 20%：重新审视架构

### Checkpoint 3: 3任务验证（Week 2）
- [ ] 3个困难任务训练完成
- [ ] 评测完成
- [ ] 初步分析完成
- **决策**: 是否扩展到全任务

### Checkpoint 4: 全任务完成（Week 3）
- [ ] 8任务（或7任务）训练评测完成
- [ ] 主要分析完成
- **决策**: 是否进入阶段4深度分析

### Checkpoint 5: 实验完成（Week 4）
- [ ] 所有消融实验完成
- [ ] 所有分析完成
- [ ] 实验报告撰写完成
- **决策**: 发表论文或继续改进

---

## 九、待办事项清单

### ✅ 已完成
- [x] 查找ACT+UniVTAC的现有checkpoint
  - ✅ 找到所有8个任务的官方checkpoint
  - 📄 报告：[ACT_UNIVTAC_CHECKPOINT_REPORT.md](ACT_UNIVTAC_CHECKPOINT_REPORT.md)
- [x] 检查3个优先任务的数据manifest是否存在
  - ✅ `data_manifests/put_bottle_in_shelf_official_first50.json`
  - ✅ `data_manifests/insert_HDMI_official_first50.json`
  - ✅ `data_manifests/lift_can_official_first50.json`
- [x] 验证官方encoder.pth位置和可用性
  - ✅ `/home/rmc/workspace/UniVTAC/policy/ACT/encoder/checkpoints/resnet18/official/encoder.pth`
- [x] 双流架构实现和测试
- [x] 配置文件创建
- [x] 实验计划文档完成

### 立即执行（今天）
- [ ] 适配训练脚本以支持双流架构
  - [ ] 修改`scripts/training/train_vtla.py`或创建`train_dual_stream.py`
  - [ ] 添加双流模型的加载逻辑
  - [ ] 测试数据加载和模型初始化
- [ ] 创建deploy配置文件
  - [ ] `univtac_adapter/deploy_dual_stream.yml`
  - [ ] 配置双流模型的推理接口
- [ ] 快速smoke test
  - [ ] 运行几个training steps验证流程
  - [ ] 确认loss计算正确
  - [ ] 验证checkpoint保存

### 明天（2026-08-14）
- [ ] 启动阶段1训练（insert_HDMI，单GPU）
  - [ ] GPU: 1张L40S
  - [ ] 预计时间：8-12小时
  - [ ] 监控loss曲线和指标
- [ ] 设置实验监控
  - [ ] 定期检查train.log
  - [ ] 监控GPU使用率
  - [ ] 保存中间checkpoint

### 本周（阶段1完成）
- [ ] 完成insert_HDMI评测（100 seeds）
  - [ ] 预计时间：8-10小时
  - [ ] 生成aggregate_result.json
- [ ] 对比分析
  - [ ] 双流 vs ACT+UniVTAC (14%)
  - [ ] 生成对比报告
- [ ] **决策点**：是否进入阶段2
  - [ ] 如果≥20%：✅ 进入阶段2
  - [ ] 如果15-20%：⚠️ 分析后决定
  - [ ] 如果<15%：❌ 调试架构

---

## 十、文档引用

- [双流架构设计](docs/dual_stream_architecture.md)
- [架构深度剖析](docs/architecture_critique.md)
- [UniVTAC论文对齐训练协议](docs/UNIVTAC_PAPER_ALIGNED_TRAINING.md)
- [双流配置文件](configs/dual_stream_stage2.json)

---

## 附录A：论文中的Baseline数据

**来源**: UniVTAC论文Table I vs Shipped Checkpoint Logs

| 任务 | ACT (论文) | ViTAL (论文) | ACT+UniVTAC (论文) | Shipped Log |
|------|-----------|-------------|-------------------|-------------|
| grasp_classify | 50 | **100** | 99 | 100 |
| insert_tube | 45 | 34 | **56** | 64 |
| pull_out_key | - | - | - | 48 |
| lift_can | - | - | - | 29 |
| insert_hole | 19 | 25 | **24** | 39 |
| insert_HDMI | 15 | 6 | **28** | 14 |
| lift_bottle | - | - | - | 46 |
| put_bottle_in_shelf | - | - | - | 8 |

**注**: Shipped Log是官方checkpoint的实际评测结果，与论文Table I有差异。

---

## 附录B：双流架构核心改进点

| 原始VTLA问题 | 双流架构解决方案 |
|-------------|-----------------|
| 早期CrossModalFusion破坏独立性 | 保持独立到decoder输出 |
| 虚假的"触觉微调分支"（输入是混合特征） | 真正独立的vision/tactile decoder |
| 三阶段训练必要性存疑 | 端到端训练，无需分阶段 |
| 无法解释模态贡献 | 可视化fusion权重，量化模态作用 |
| 无法单独评估模态 | 推理时可调整模态权重做消融 |

---

**审批**: 待用户确认  
**下一步**: 执行待办事项清单，开始阶段0准备工作
