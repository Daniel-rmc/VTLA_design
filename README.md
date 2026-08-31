# VTLA Design

面向接触操作任务的视觉—触觉动作策略仓库。当前推荐主线是作为独立插件接入固定版本的 Hugging Face LeRobot，用 LeRobot v3 真机数据完成训练、checkpoint 保存和推理；原有 UniVTAC/HDF5 实验链路保留在仓库中作为历史基线。

> 当前实现尚未接入语言编码器，因此严格来说是 Vision-Tactile-Action（VTA）策略。推荐主线的 manipulationNet v3 数据使用 metadata 定义的 16D 双臂末端位姿/夹爪 action；历史 UniVTAC HDF5 没有独立的 commanded-action 字段，旧链路才按 ACT 数据约定使用下一时刻关节位置作为动作代理。

## 推荐主线：LeRobot v3 + manipulationNet

当前已验证的组合：

- LeRobot `v0.6.1`，submodule commit `7e241bd630a3719a56157a497ce5d08f244784f1`；
- 专用镜像 `vtla-train:lerobot-0.6.1`，Python 3.12、PyTorch 2.11.0+cu128；
- 数据集 `/home/rmc/workspace/datasets/manipulationNet/peg_in_hole/15holes_v3`；
- 1,508 episodes、196,667 frames、15 tasks、10 Hz；
- 28D state、16D action、3 路视觉相机、4 路触觉相机；
- 单卡完整模型和双卡 DDP 训练、checkpoint 保存、严格回载和真实样本推理均已通过；
- 首轮双卡 30K 正式训练已完成，最终 checkpoint 位于 `outputs/train/2026-08-27_vtla_manipulationnet_30k_gpu03_v2/checkpoints/030000`。

首次使用：

```bash
cd /home/rmc/workspace/VTLA_design
git submodule update --init --recursive
./docker/build_train_image.sh
./docker/run_train_container.sh
docker exec -it vtla_train bash
```

进入容器后先做环境和数据校验，再启动训练：

```bash
python docker/verify_environment.py
python scripts/data/validate_manipulationnet.py \
  --source-root /workspace/datasets/manipulationNet/peg_in_hole/15holes_v3 \
  --decode-samples 3 --quiet
VTLA_GPUS=0,3 ./scripts/training/start_lerobot_vtla_tmux.sh
```

容器固定可见全部 GPU、内置 tmux、使用 120G SHM，并以 `unless-stopped` 策略常驻；具体训练卡必须通过 `VTLA_GPUS` 显式选择。正式配置是每 GPU batch 32、8 workers、30,000 steps，每 5,000 steps 保存；LR 在前 1,000 steps warmup，保持到 27,000，然后 cosine decay 到 `1e-6`。W&B 默认使用在线项目 `vtla-manipulationnet-30k`，API key 从 Git 忽略的 `.secrets/wandb.env` 读取。配置位于 [configs/lerobot/vtla_manipulationnet.yaml](configs/lerobot/vtla_manipulationnet.yaml)，完整的数据重建、tmux/W&B、多卡训练、恢复训练和 checkpoint 推理命令见 [LeRobot 训练与容器指南](docs/LEROBOT_TRAINING.md)。

## 历史 UniVTAC 基线

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
├── pyproject.toml             # lerobot_policy_vtla 外部插件包
├── third_party/lerobot/       # 固定为 v0.6.1 的 Git submodule
├── src/lerobot_policy_vtla/   # LeRobot policy/config/processor 与模型主体
├── configs/lerobot/           # 正式训练和 smoke 配置
├── docker/                    # 固定版本训练镜像与环境/DDP 健康检查
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

## 历史 UniVTAC 环境约定

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

### UniVTAC 论文对齐的八任务训练

全部八个官方任务使用各自的 episode 0–49、官方 seed-1 80/20 切分、单任务全局 batch 64 和精确 4000 optimizer steps。首轮四任务启动命令为：

```bash
scripts/training/start_official_tasks.sh
```

剩余四任务可续接到同一个 run group：

```bash
scripts/training/continue_official_tasks.sh \
  runs/stage2/official_<timestamp>_<git>_gpu123
```

任务在物理 GPU 1/2/3 上做任务级并行；每次运行独立保存完整命令、配置、数据指纹、训练指标、曝光量和 checkpoint。参数依据、逐任务样本核算以及与论文实现的差异见 [UniVTAC 论文对齐训练协议](docs/UNIVTAC_PAPER_ALIGNED_TRAINING.md)。

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
