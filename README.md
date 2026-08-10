# VTLA Design

面向 UniVTAC 接触操作任务的视觉—触觉动作策略研究仓库。项目包含模型实现、三阶段训练、官方数据校验、离线验证、UniVTAC 闭环评测和完整的实验归档工具。

> 当前实现尚未接入语言编码器，因此严格来说是 Vision-Tactile-Action（VTA）策略。官方 HDF5 没有独立的 commanded-action 字段，训练按 ACT 数据约定使用下一时刻关节位置作为动作代理。

## 当前基线

- 数据：ModelScope `byml2024/UniVTAC` 的完整 `grasp_classify/clean` 子集，共 100 条轨迹、5,265 个时序样本。
- 控制接口：原始 9D 关节观测取列 `0..7`，模型与 UniVTAC 控制端统一为 8D（7 个机械臂关节 + 1 个夹爪命令）。
- 训练：物理 GPU `1,2,3`，BF16，batch 64/GPU，全局 batch 192，150 epochs，90/10 分层训练/验证切分。
- 最佳部署候选：epoch 130。
- UniVTAC：固定 100 个 seed 上成功 80/100（80.0%）。

完整状态和复盘见 [训练状态](docs/TRAINING_STATUS.md) 与本地运行报告：
`runs/stage2/20260809_155942_1073ae9_gpu123/eval/EVALUATION_SUMMARY.md`。

## 目录结构

```text
VTLA_design/
├── README.md                  # 项目总览和常用命令
├── dataloader.py              # HDF5 数据发现、切分、归一化和样本构造
├── training_utils.py          # 配置、Git/GPU 元数据和指标记录
├── models/                    # 模型主体、编码器、融合层和动作头
├── scripts/
│   ├── training/              # 单卡/多卡训练入口、启动器和状态检查
│   ├── evaluation/            # 离线评测、UniVTAC 评测和 seed 聚合
│   └── diagnostics/           # NCCL 与官方数据完整性检查
├── univtac_adapter/           # UniVTAC Policy 适配器和冻结部署配置
├── data_manifests/            # 官方数据清单与 SHA-256 指纹
├── docs/                      # 架构、训练策略、代码审查和运行指南
├── tests/                     # 单元与集成回归测试
├── runs/                      # 每次训练/评测的自包含产物（Git 忽略）
├── logs/                      # 历史日志（Git 忽略）
└── checkpoints/               # 历史 checkpoint（Git 忽略）
```

脚本和文档的详细索引分别见 [scripts/README.md](scripts/README.md) 与 [docs/README.md](docs/README.md)。

## 环境约定

当前验证环境：

- 项目：`/home/rmc/workspace/VTLA_design`
- UniVTAC：`/home/rmc/workspace/UniVTAC`
- Python：`/home/rmc/miniconda/envs/UniVTAC/bin/python`
- GPU：4 × NVIDIA L40S

以下命令均假设从项目根目录执行：

```bash
cd /home/rmc/workspace/VTLA_design
```

## 数据校验

重新验证官方任务子集并生成 manifest：

```bash
/home/rmc/miniconda/envs/UniVTAC/bin/python \
  -m scripts.diagnostics.validate_univtac_dataset \
  --dataset-dir /home/rmc/workspace/UniVTAC/data/official/grasp_classify/clean \
  --output data_manifests/grasp_classify_clean_modelscope.json
```

校验内容包括 episode 数量、metadata、HDF5 必需字段、帧数一致性、9D 原始关节布局、类别以及每个文件的 SHA-256。

## 训练

### UniVTAC 论文对齐的四任务训练

`grasp_classify`、`insert_HDMI`、`insert_hole`、`insert_tube` 使用官方 episode 0–49、官方 seed-1 80/20 切分、单任务全局 batch 64 和精确 4000 optimizer steps：

```bash
scripts/training/start_official_tasks.sh
```

任务在物理 GPU 1/2/3 上并行调度；每次运行独立保存完整命令、配置、数据指纹、训练指标、曝光量和 checkpoint。参数依据、逐任务样本核算以及与论文实现的差异见 [UniVTAC 论文对齐训练协议](docs/UNIVTAC_PAPER_ALIGNED_TRAINING.md)。

### 既有通用训练入口

推荐的三卡 Stage 2 训练：

```bash
scripts/training/start_training_multigpu.sh stage2 3 1,2,3
```

单卡启动和状态检查：

```bash
scripts/training/start_training.sh stage2 1
scripts/training/check_training.sh
```

启动器会自动创建 `runs/<stage>/<timestamp>_<git>_gpu<ids>/`，保存：

- `config.json`：完整参数、命令、Git 状态、GPU/CUDA/PyTorch 信息、数据切分和归一化统计；
- `launch_command.sh`：可复查的原始启动命令；
- `train.log` 与 `metrics.jsonl`：完整日志和逐 epoch 指标；
- `checkpoints/`：模型、优化器、调度器、数据统计和嵌入式运行配置；
- `exit_code`：后台任务最终退出码。

本机 NCCL 2.21.5 的 GPU P2P collective 会挂起，因此多卡启动器默认记录并设置 `NCCL_P2P_DISABLE=1`。

更多参数和故障排查见 [训练指南](docs/TRAINING_GUIDE.md)。

## 评测

### 留出集离线评测

```bash
CUDA_VISIBLE_DEVICES=3 /home/rmc/miniconda/envs/UniVTAC/bin/python \
  -m scripts.evaluation.eval_vtla_offline \
  --checkpoint runs/stage2/20260809_155942_1073ae9_gpu123/checkpoints/stage2_epoch_130.ckpt \
  --dataset-dir /home/rmc/workspace/UniVTAC/data/official/grasp_classify/clean \
  --split validation \
  --output runs/stage2/20260809_155942_1073ae9_gpu123/eval/offline_validation_epoch130.json
```

`--split validation` 会严格复用 checkpoint 中记录的验证 episode 清单，避免把训练轨迹混入模型选择。

### UniVTAC 闭环评测

首次启动 Isaac Sim 前，用户需要亲自阅读并接受 NVIDIA Omniverse EULA：

```bash
/home/rmc/miniconda/envs/UniVTAC/bin/python -c "import isaacsim"
```

随后执行：

```bash
/home/rmc/miniconda/envs/UniVTAC/bin/python \
  -m scripts.evaluation.run_univtac_eval \
  --run-dir runs/stage2/20260809_155942_1073ae9_gpu123 \
  --deploy-config univtac_adapter/deploy_official8d_epoch130.yml \
  --gpu 3 --total-num 100
```

合并分片结果：

```bash
/home/rmc/miniconda/envs/UniVTAC/bin/python \
  -m scripts.evaluation.summarize_univtac_eval \
  --result-root /home/rmc/workspace/UniVTAC/eval_result/VTLA/grasp_classify/deploy_official8d_epoch130 \
  --start-seed 1000000 --end-seed 1000099 \
  --output runs/stage2/20260809_155942_1073ae9_gpu123/eval/univtac/aggregate_result.json
```

评测请求、deploy/task 配置副本、stdout、退出码和结构化结果都会归档到对应 run；UniVTAC 原始场景数据、metadata 和视频保留在其 `eval_result/` 目录。

## 测试与诊断

```bash
/home/rmc/miniconda/envs/UniVTAC/bin/python -m pytest -q
```

三卡 NCCL 健康检查：

```bash
CUDA_VISIBLE_DEVICES=1,2,3 NCCL_P2P_DISABLE=1 \
  /home/rmc/miniconda/envs/UniVTAC/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=3 -m scripts.diagnostics.nccl_smoke
```

## 关键设计限制

- 尚无语言输入、tokenizer 或语言编码器；
- 动作为“下一时刻 joint position”代理，而非机器人控制器真实 commanded action；
- Stage 3 缺少可信的 contact 标签，无法直接监督接触检测；
- 当前结果只覆盖 `grasp_classify`，不能外推为 UniVTAC 多任务能力。

详细设计与问题清单见 [架构设计](docs/architecture_design.md)、[训练策略](docs/training_strategy.md) 和 [代码审查](docs/CODE_REVIEW.md)。
