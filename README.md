# VTLA: Vision-Tactile-Language-Action Model

一个用于接触丰富机器人操作任务的新型视触觉融合模型，具有独立触觉编码器和触觉感知动作微调机制。

> 当前实现尚未接入语言编码器，因此严格来说是 Vision-Tactile-Action 策略。UniVTAC 原始轨迹没有独立 action 字段，训练加载器遵循官方 ACT 预处理约定，以“下一时刻关节位置”构造 action。详见 [CODE_REVIEW.md](CODE_REVIEW.md)。

正式训练使用 ModelScope 发布的 `grasp_classify/clean` 100 条轨迹。原始记录为 9D（7 个机械臂关节 + 两个重复手指位置），模型和 UniVTAC 控制接口统一为原生 8D（7 个机械臂关节 + 1 个夹爪），选取原始列 `0..7`。训练采用按 rough/plain 分层的固定 90/10 episode 级训练/验证划分。

本地训练会为每次运行创建独立目录 `runs/<stage>/<run-name>/`，保存完整配置、Git 版本、GPU 映射、数据归一化统计、日志、逐 epoch 指标和 checkpoint。当前三卡命令为：

```bash
./start_training_multigpu.sh stage2 3 1,2,3
```

默认配置为每卡 batch 64、全局 batch 192、BF16、150 epochs，每 5 epochs 在留出的 10 条轨迹上验证，并保存 `stage2_best.ckpt`。每个 run 的 `config.json` 会记录数据 manifest、episode 清单、8D 列选择、Git 版本、GPU 映射和完整启动命令。

当前主机的 NCCL 2.21.5 在 L40S 间启用 P2P 时 collective 会超时；启动器默认设置 `NCCL_P2P_DISABLE=1`，使用已通过三卡 all-reduce 健康检查的共享内存通信路径，并把该设置记录到 run 配置。

## UniVTAC 评测

最终 checkpoint 可通过仓库内的适配器接入 UniVTAC 统一评测。首次运行 Isaac Sim 时，必须由用户本人阅读并接受 NVIDIA Omniverse EULA：

```bash
/home/rmc/miniconda/envs/UniVTAC/bin/python -c "import isaacsim"
```

接受后，在物理 GPU 3 上执行 100 个 `grasp_classify/demo` episode：

```bash
/home/rmc/miniconda/envs/UniVTAC/bin/python run_univtac_eval.py \
    --run-dir runs/stage2/20260809_155942_1073ae9_gpu123 \
    --deploy-config univtac_adapter/deploy_official8d_epoch130.yml \
    --gpu 3 --total-num 100
```

适配器位于 `univtac_adapter/`，使用训练时相同的双相机、原始触觉 RGB、ImageNet 图像归一化和 checkpoint 关节统计。UniVTAC 原始视频/metadata 保存在其 `eval_result/`，本次请求、配置、完整日志和结构化结果同时归档到训练 run 的 `eval/univtac/` 目录。

如果评测按多个 GPU/seed 区间并行运行，可按 seed 去重合并为一个结果：

```bash
/home/rmc/miniconda/envs/UniVTAC/bin/python summarize_univtac_eval.py \
    --result-root /home/rmc/workspace/UniVTAC/eval_result/VTLA/grasp_classify/deploy \
    --start-seed 1000000 --end-seed 1000099 \
    --output runs/stage2/20260809_014410_d04cb96_gpu123/eval/univtac/aggregate_result.json
```

在启动耗时较长的模拟器前，可先做确定性离线部署检查：

```bash
CUDA_VISIBLE_DEVICES=3 /home/rmc/miniconda/envs/UniVTAC/bin/python eval_vtla_offline.py \
    --checkpoint runs/stage2/20260809_155942_1073ae9_gpu123/checkpoints/stage2_epoch_130.ckpt \
    --dataset-dir /home/rmc/workspace/UniVTAC/data/official/grasp_classify/clean \
    --split validation \
    --output runs/stage2/20260809_155942_1073ae9_gpu123/eval/offline_validation_epoch130.json
```

## 🎯 核心创新

1. **独立触觉编码器**：不与视觉共享参数，专门处理触觉传感器数据
2. **视触觉交叉注意力融合**：Token级别的双向交叉注意力，深度融合视觉和触觉信息
3. **触觉感知动作微调分支**：在动作生成阶段，基于纯触觉信息进行接触后的精细调整
4. **自适应门控机制**：根据接触强度动态调整触觉残差的贡献权重

## 📐 架构概览

```
                    RGB Images              Tactile Images
                        ↓                         ↓
                  Vision Backbone          Tactile Encoder
                   (ResNet18)               (ResNet34)
                        ↓                         ↓
                  Vision Tokens            Tactile Tokens
                        ↘                       ↙
                    Cross-Modal Fusion Layer
                  (Bidirectional Cross-Attention)
                              ↓
                      Fused Tokens + qpos
                              ↓
                    Transformer Encoder-Decoder
                         (CVAE-based)
                              ↓
                   ┌──────────┴──────────┐
                   ↓                     ↓
            Main Action Head     Tactile Refine Head
          (Vision+Tactile)        (Tactile Only)
                   ↓                     ↓
              Main Action           Action Residual
                   └──────────┬──────────┘
                              ↓
                      Final Action
            a_final = a_main + λ * Δa_tactile
```

## 🚀 快速开始

### 环境配置

```bash
# 克隆仓库
cd /home/rmc/workspace/VTLA_design

# 安装依赖
pip install torch torchvision numpy tqdm

# 确保UniVTAC在Python路径中
export PYTHONPATH=$PYTHONPATH:/home/rmc/workspace/UniVTAC
```

### 三阶段训练策略

#### Stage 1: 触觉编码器预训练（自监督）

使用自监督任务预训练触觉编码器：
- Marker位置重建
- RGB图像重建
- Pose回归

```bash
python train_vtla.py \
    --stage stage1 \
    --dataset_dir /path/to/tactile_dataset \
    --tactile_names tac_left tac_right \
    --tactile_supervise marker rgb pose \
    --batch_size 32 \
    --num_epochs 200 \
    --lr_stage1 1e-4 \
    --ckpt_dir ./checkpoints/stage1 \
    --device cuda:0
```

**预期输出**：
- 预训练的触觉编码器权重
- 重建损失曲线（marker MSE, RGB MSE, pose MSE）

#### Stage 2: 端到端VLA训练

加载预训练的触觉编码器，训练完整的VLA模型：

```bash
python train_vtla.py \
    --stage stage2 \
    --dataset_dir /path/to/robot_dataset \
    --camera_names cam_high cam_left \
    --tactile_names tac_left tac_right \
    --state_dim 8 \
    --joint_indices 0 1 2 3 4 5 6 7 \
    --chunk_size 100 \
    --batch_size 8 \
    --num_epochs 1000 \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --lr_tactile 5e-5 \
    --kl_weight 10.0 \
    --stage1_ckpt ./checkpoints/stage1/stage1_epoch_200.ckpt \
    --ckpt_dir ./checkpoints/stage2 \
    --device cuda:0
```

**训练的组件**：
- ✅ Vision Backbone (ResNet18, 低学习率微调)
- ✅ Tactile Encoder (预训练权重，低学习率微调)
- ✅ Cross-Modal Fusion (从头训练)
- ✅ Transformer Encoder-Decoder (从头训练)
- ✅ Main Action Head (从头训练)
- ❌ Tactile Refine Head (冻结，不参与训练)

#### Stage 3: 触觉微调分支训练

冻结主干网络，仅训练触觉微调分支：

```bash
python train_vtla.py \
    --stage stage3 \
    --dataset_dir /path/to/contact_rich_dataset \
    --camera_names cam_high cam_left \
    --tactile_names tac_left tac_right \
    --batch_size 16 \
    --num_epochs 200 \
    --lr_stage3 5e-5 \
    --refine_weight 0.5 \
    --contact_weight 0.1 \
    --stage2_ckpt ./checkpoints/stage2/stage2_epoch_1000.ckpt \
    --ckpt_dir ./checkpoints/stage3 \
    --device cuda:0
```

**训练的组件**：
- ❌ 主干网络（完全冻结）
- ✅ Tactile Refine Head (从头训练)
- ✅ Contact Detector (从头训练)
- ✅ Adaptive Scale Predictor (从头训练)

**建议数据**：使用接触丰富的任务轨迹（如插入、装配、抓取等）

## 📊 模型配置

### 默认超参数

```python
# 模型结构
hidden_dim: 512              # Transformer隐藏层维度
nheads: 8                    # 注意力头数
enc_layers: 4                # Encoder层数
dec_layers: 6                # Decoder层数
cross_attn_layers: 2         # 交叉注意力层数

# 触觉编码器
tactile_backbone: 'resnet34' # ResNet34
tactile_latent_dim: 512      # 触觉特征维度

# 损失权重
kl_weight: 10.0              # CVAE KL散度权重
refine_weight: 0.5           # 触觉微调损失权重
contact_weight: 0.1          # 接触检测损失权重
refine_scale: 0.1            # 触觉残差缩放系数

# 训练
lr: 1e-4                     # 主学习率
lr_backbone: 1e-5            # Vision backbone学习率
lr_tactile: 5e-5             # Tactile encoder学习率
batch_size: 8                # Stage 2/3批大小
chunk_size: 100              # 动作序列长度
```

### 与UniVTAC的区别

| 特性 | UniVTAC | VTLA (本模型) |
|------|---------|--------------|
| 触觉编码 | 共享vision backbone或简单concat | 独立ResNet34编码器 |
| 多模态融合 | 直接拼接特征 | 双向交叉注意力 (2层) |
| 动作生成 | 单一action head | 双路：主路 + 触觉微调 |
| 接触感知 | 隐式学习 | 显式接触检测 + 自适应门控 |
| 预训练 | 无 | 触觉编码器自监督预训练 |

## 🧪 模型测试

测试各个模块：

```bash
# 测试触觉编码器
cd models
python tactile_encoder.py

# 测试交叉注意力融合
python cross_modal_fusion.py

# 测试动作头
python action_heads.py

# 测试完整模型
python vtla_policy.py
```

## 📈 评估指标

### Stage 1 (触觉编码器预训练)
- **Marker MSE**: Marker位置重建误差
- **RGB MSE**: RGB图像重建误差
- **Pose MSE**: Pose回归误差

### Stage 2 (端到端VLA)
- **Action L1**: 动作预测L1损失
- **KL Divergence**: CVAE潜在空间KL散度
- **Success Rate**: 任务成功率（评估时）

### Stage 3 (触觉微调分支)
- **Main L1**: 主动作L1损失
- **Refine L1**: 微调后动作L1损失
- **Contact Accuracy**: 接触检测准确率
- **Tactile Contribution**: 触觉残差的平均贡献 (λ * ||Δa||)

## 💡 设计动机

### 为什么需要独立的触觉编码器？

1. **模态特异性**：触觉信号（marker变形、力分布）与RGB视觉在特征空间中差异巨大
2. **自监督预训练**：独立编码器可以利用触觉专有的自监督任务（marker重建）
3. **专门优化**：不与视觉竞争梯度，可以更好地学习触觉特有的细微特征

### 为什么需要交叉注意力融合？

简单的拼接（concat）无法捕捉视觉和触觉之间的关联：
- **视觉→触觉**：视觉提供上下文，触觉关注哪里在接触
- **触觉→视觉**：触觉感知接触点，视觉提供周围物体信息
- **双向交互**：两个模态互相增强

### 为什么需要触觉微调分支？

在接触后的最后1mm关键阶段：
- 视觉变化不明显（视差小、分辨率限制）
- 触觉信号变化显著（压力、变形）
- 类比人类：闭眼也能通过触觉完成精细操作

## 🔧 定制化

### 更换触觉编码器backbone

```python
# 使用更大的ResNet50
--tactile_backbone resnet50 --tactile_latent_dim 1024

# 使用ResNet18（更快）
--tactile_backbone resnet18 --tactile_latent_dim 512
```

### 调整交叉注意力层数

```python
# 更深的融合（更强，但更慢）
--cross_attn_layers 4

# 更浅的融合（更快）
--cross_attn_layers 1
```

### 禁用触觉微调分支

```python
# 仅使用主动作头（类似UniVTAC）
--use_tactile_refine False
```

### 调整触觉残差权重

```python
# 更大的触觉贡献
--refine_scale 0.2

# 更小的触觉贡献
--refine_scale 0.05

# 自适应（推荐）
# 权重由contact detector自动调整
```

## 📁 代码结构

```
VTLA_design/
├── architecture_design.md          # 架构设计文档
├── models/
│   ├── __init__.py
│   ├── tactile_encoder.py          # 独立触觉编码器
│   ├── cross_modal_fusion.py       # 视触觉交叉注意力
│   ├── action_heads.py              # 双路动作生成头
│   └── vtla_policy.py               # 完整VTLA模型
├── train_vtla.py                    # 三阶段训练脚本
└── README.md                        # 本文档
```

## 🎓 训练建议

### 数据准备

1. **Stage 1数据**：
   - 触觉图像 + marker标注
   - 可以使用UniVTAC的触觉数据集
   - 或自己收集触觉图像（接触不同物体）

2. **Stage 2数据**：
   - 完整的演示轨迹（qpos, images, actions）
   - 需要相机图像 + 触觉图像同步采集
   - 使用UniVTAC的数据格式

3. **Stage 3数据**：
   - 筛选接触丰富的轨迹段
   - 标准：触觉信号强度 > 阈值
   - 可以是插入、装配、抓取等任务

### 训练技巧

1. **Stage 1预训练很重要**：
   - 充分预训练（200+ epochs）
   - 监控重建质量（可视化）
   - 保存最佳checkpoint

2. **Stage 2学习率调整**：
   - Vision backbone: 1e-5 (微调)
   - Tactile encoder: 5e-5 (微调)
   - 其他组件: 1e-4 (从头训练)

3. **Stage 3数据增强**：
   - 仅使用接触丰富的轨迹段
   - 可以对触觉图像增加更多噪声
   - 测试不同的refine_scale

4. **超参数搜索**：
   - `kl_weight`: 尝试 [5.0, 10.0, 20.0]
   - `refine_scale`: 尝试 [0.05, 0.1, 0.2]
   - `cross_attn_layers`: 尝试 [1, 2, 4]

## 📖 引用

如果您使用本模型，请引用：

```bibtex
@misc{vtla2024,
  title={VTLA: Vision-Tactile-Language-Action Model with Tactile-Aware Refinement},
  author={Your Name},
  year={2024}
}
```

## 🙏 致谢

本项目基于以下工作：
- **UniVTAC**: 视触觉动作克隆框架
- **ACT**: Action Chunking with Transformers
- **DETR**: End-to-End Object Detection with Transformers

## 📧 联系

如有问题或建议，请提Issue或联系作者。
