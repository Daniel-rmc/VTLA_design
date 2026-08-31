# VTLA × LeRobot v3 训练与容器指南

## 结论

推荐使用本仓库新建的专用镜像 `vtla-train:lerobot-0.6.1`，不要把旧的 `kdc_icra` 容器继续作为 VTLA 正式训练环境。

旧容器仍可用于复查 `kuavo_data_challenge` 的历史实验，但不适合作为新基线：其中同时存在 LeRobot 0.4.2/0.5.1、不同 Python 环境和旧版 `draccus`。实际 smoke 已证明，它能完成 VTLA 的一个 optimizer step，但在保存 LeRobot 0.6.1 checkpoint 时会因 `draccus` API 不兼容失败。专用镜像完全由本仓库的固定 submodule、锁文件和 Dockerfile 构建，已经完成 checkpoint 保存与回载验证。

## 稳定边界

| 组件 | 固定值 |
| --- | --- |
| LeRobot | `v0.6.1` |
| LeRobot commit | `7e241bd630a3719a56157a497ce5d08f244784f1` |
| Python | 3.12 |
| PyTorch | 2.11.0+cu128，由 LeRobot `uv.lock` 固定 |
| torchvision | 0.26.0+cu128 |
| draccus | 0.11.6 |
| CUDA 基础镜像 | `nvidia/cuda:12.8.1-base-ubuntu24.04`，Dockerfile 中固定 digest |
| uv 构建工具 | 0.8.13，Dockerfile 中固定 digest |
| VTLA 插件 | `lerobot_policy_vtla==0.1.0` |
| 训练镜像 | `vtla-train:lerobot-0.6.1` |

本次已构建镜像的 content ID、文件摘要和精确大小记录在 `docker/image-lock.json`。重新构建后镜像 ID 可能因 Ubuntu apt 仓库快照变化而改变，应重新运行验证并更新该记录；Python 依赖仍由 LeRobot `uv.lock` 固定。

依赖关系如下：

```text
VTLA_design
├── third_party/lerobot           固定的上游源码和 uv.lock
├── src/lerobot_policy_vtla       仓库外 policy 插件，不修改 LeRobot 内部代码
├── configs/lerobot               数据与训练配置
├── scripts/data                  数据校验与聚合
├── scripts/training              单/多卡训练与 checkpoint 校验
└── docker                        构建、启动及 CUDA/NCCL 健康检查
```

这一边界允许未来单独升级 LeRobot：先移动 submodule 和依赖锁，再在分支上运行测试；VTLA 不需要继续向 LeRobot 源码目录植入私有文件。

## 模型与数据接口

LeRobot 从数据集 metadata 自动填入 policy features。本数据当前接口为：

- `observation.state`：28D；
- `action`：16D；
- 普通视觉：front、left hand、right hand，共 3 路；
- 触觉视觉：左右手各 2 路，共 4 路；
- 原始图像：`3 × 480 × 640`，checkpointed processor resize 到 `224 × 224`；
- fps：10；
- action chunk：50。

新模型不再假定 state dimension 等于 action dimension。模型主体包含独立视觉/触觉 ResNet、双向 cross attention、CVAE Transformer action chunk，以及 contact-gated tactile residual action head。模型代码只依赖 PyTorch/torchvision 和 LeRobot 的公共 policy/processor 接口，不再引用 UniVTAC 私有模块。

这个 gate 只通过动作重建损失端到端学习，并没有真实 contact label 监督，因此 `contact_probability` 不是经过校准的物理接触概率；`tactile_residual_l1` 也是 gate 和 scale 作用前的原始残差，并不等于触觉对最终动作的实际改变量。两者只能用于诊断网络内部行为，不能替代接触标签、力监督或闭环触觉消融实验。下一次显式修改训练日志接口时，建议将前者重命名为 `tactile_gate_mean`，并新增缩放后的 `tactile_correction_l1`；本次为保持源码、固定镜像和首轮 30K run 一致，不改变已有指标键。

## 初始化仓库与构建镜像

在宿主机执行：

```bash
cd /home/rmc/workspace/VTLA_design
git submodule update --init --recursive
git -C third_party/lerobot rev-parse HEAD
./docker/build_train_image.sh
```

第二条命令必须输出：

```text
7e241bd630a3719a56157a497ce5d08f244784f1
```

检查镜像：

```bash
docker image inspect vtla-train:lerobot-0.6.1 \
  --format '{{.Id}} {{.Size}} {{json .Config.Labels}}'
```

创建并启动持久训练容器：

```bash
./docker/run_train_container.sh
docker exec -it vtla_train bash
```

`vtla_train` 使用 `--gpus all`、120G SHM 和 `restart=unless-stopped`，内部常驻 `sleep infinity`。脚本可重复执行：已有且配置正确的容器只会被重新启动，不会删除。它将整个 `/home/rmc/workspace` 挂载为 `/workspace`，并将 `${VTLA_DATASETS_ROOT:-/home/rmc/workspace/datasets}` 明确挂载为 `/workspace/datasets`，因此代码、数据和 outputs 都保留在宿主机；Hugging Face、Torch 和 Triton 缓存保存在 Docker volume `vtla-cache`。120G 是按需使用的上限，不会在容器启动时立即占满主机内存。

镜像内固定安装 tmux，可在 SSH 断开后继续训练。`restart=unless-stopped` 只负责恢复容器常驻进程；主机或容器重启后 tmux 任务不会自动恢复，应从最新 5K checkpoint 显式 resume。

## 启动前健康检查

以下命令在训练容器内执行：

```bash
cd /workspace/VTLA_design
python docker/verify_environment.py
```

预期至少看到：

```text
lerobot=0.6.1
lerobot_policy_vtla=0.1.0
cuda_available=True
gpu_count=4
```

本机 L40S 在 Docker 内使用 NCCL P2P/CUDA-memory transport 会挂起。启动器默认使用已经验证过的共享内存 transport：

```text
NCCL_CUMEM_ENABLE=0
NCCL_P2P_DISABLE=1
```

这会牺牲一部分通信性能，但优先保证训练稳定。如果宿主驱动、拓扑或容器运行时以后发生变化，应先运行健康检查，再决定是否覆盖默认值：

```bash
CUDA_VISIBLE_DEVICES=0,3 \
NCCL_CUMEM_ENABLE=0 NCCL_P2P_DISABLE=1 \
accelerate launch --multi_gpu --num_processes=2 \
  --num_machines=1 --mixed_precision=no --dynamo_backend=no \
  docker/verify_distributed.py
```

两张卡应分别输出 `phase=ok all_reduce=3.0`。

## 数据集状态

原始目录：

```text
/workspace/datasets/manipulationNet/peg_in_hole/15holes_merged
```

包含 `hole_A1` 到 `hole_E3` 共 15 个 LeRobot v3 数据集。合计：

- 1,508 episodes；
- 196,667 frames；
- 15 tasks；
- 7 路视频；
- 28D state；
- 16D action。

原始数据中 `hole_D1/D2/D3/E2/E3` 的部分 episode metadata 仍引用合并前已经不存在的 data/meta parquet file index。实际 frame parquet、全局 offset 和视频引用都存在且连续。聚合脚本只在临时 metadata 副本中把陈旧引用映射到唯一存在的 parquet，然后调用 LeRobot 官方 `aggregate_datasets`；不会修改原始 15 个目录。

正式训练目录：

```text
/workspace/datasets/manipulationNet/peg_in_hole/15holes_v3
```

验证命令：

```bash
python scripts/data/validate_manipulationnet.py \
  --source-root /workspace/datasets/manipulationNet/peg_in_hole/15holes_v3 \
  --decode-samples 3 \
  --output data_manifests/manipulationnet_v3.json \
  --quiet
```

预期：

```text
valid=True datasets=1 episodes=1508 frames=196667 errors=0
```

`--decode-samples 3` 会解码首、中、末三个 episode 的全部 7 路视频，而不仅检查文件名。原始和聚合后的审计结果分别保存在：

- `data_manifests/manipulationnet_sources_v3.json`；
- `data_manifests/manipulationnet_v3.json`。

如需从原始数据重新生成，必须使用一个不存在的输出目录；脚本拒绝覆盖已有数据：

```bash
python scripts/data/aggregate_manipulationnet.py \
  --source-root /workspace/datasets/manipulationNet/peg_in_hole/15holes_merged \
  --output-root /workspace/datasets/manipulationNet/peg_in_hole/15holes_v3_rebuilt
```

验证完成后再通过显式路径修改训练配置。不要直接覆盖当前已经验证的 `15holes_v3`。

## 一步 smoke

正式长训前运行一个小模型、一个 batch、一个 step，并保存 checkpoint：

```bash
VTLA_CONFIG=/workspace/VTLA_design/configs/lerobot/vtla_smoke.yaml \
VTLA_GPUS=0 \
VTLA_OUTPUT_DIR=/workspace/VTLA_design/outputs/smoke_$(date +%Y%m%d_%H%M%S) \
./scripts/training/train_lerobot_vtla.sh
```

完成后验证 checkpoint 可用于真实样本推理：

```bash
python scripts/training/validate_checkpoint.py \
  outputs/<smoke_run>/checkpoints/last/pretrained_model \
  --dataset-root /workspace/datasets/manipulationNet/peg_in_hole/15holes_v3 \
  --sample-index 98333
```

它会严格加载 `model.safetensors`、policy config、preprocessor 和 postprocessor，显式把真实 `uint8` 相机 tensor 转成 `[0,1]` float，再检查 resize/normalize 后仍有动态范围，最后确认 28D state 推理得到有限的 `(1, 16)` action tensor。

## 正式训练

正式配置：

```text
configs/lerobot/vtla_manipulationnet.yaml
```

主要默认值：BF16、每进程/每 GPU batch 32、每进程 8 个训练 workers、prefetch 2、30,000 steps、每 5,000 steps 保存、10% episode-level held-out eval split，并每 1,000 steps 对最多 1,024 个 held-out samples 计算 loss。held-out eval 使用 `eval_num_workers: 0`，直接在各训练 rank 中读取样本，避免评估时再复制一组高内存的常驻 worker；训练 worker 数保持不变。该字段由 `docker/patches/lerobot-0.6.1-eval-workers.patch` 在镜像构建时叠加，固定 submodule 本身保持干净。4 卡 DDP 的有效 batch 是 `32 × 4 = 128`，双卡为 64，单卡为 32；不使用梯度累积。

优化器使用适合 VTLA/ACT 的 AdamW 设置：base/backbone peak LR 均为 `1e-5`、betas `(0.9, 0.999)`、eps `1e-8`、weight decay `1e-4`、gradient clip `10`。插件内的 `warmup_stable_cosine_decay` scheduler 不修改 `third_party/lerobot`：

- step 0–1,000：线性 warmup；
- step 1,000–27,000：保持 `1e-5`；
- step 27,000–30,000：cosine decay 到 `1e-6`。

短程 smoke 会按 30K reference 自动缩放 phase boundary，scheduler 状态随 checkpoint 保存和恢复。

### W&B 配置

正式 YAML 默认启用在线 W&B，project 为 `vtla-manipulationnet-30k`。自建服务凭据放在以下本地文件中：

```text
/workspace/VTLA_design/.secrets/wandb.env
```

该目录被 `.gitignore` 和 `.dockerignore` 排除，文件权限应保持 `600`。启动器会在创建训练进程前验证凭据，不会把 API key 打印到日志。以下字段可由启动脚本环境变量覆盖：

- `VTLA_WANDB_PROJECT`：project 名；
- `VTLA_WANDB_RUN_NAME`：run 名，默认使用 output 目录名；
- `VTLA_WANDB_ENTITY`：可选 entity；
- `VTLA_WANDB_MODE`：默认 `online`；
- `VTLA_WANDB_DISABLE_ARTIFACT`：默认 `true`，只同步 metrics/config，避免上传大型 checkpoint；
- `VTLA_WANDB_ENABLE=false`：仅在 smoke 或离线诊断时关闭。

### tmux 正式启动

进入 `vtla_train` 后运行：

```bash
cd /workspace/VTLA_design
nvidia-smi
VTLA_GPUS=0,3 \
VTLA_TMUX_SESSION=vtla_30k \
VTLA_WANDB_PROJECT=vtla-manipulationnet-30k \
VTLA_OUTPUT_DIR=/workspace/VTLA_design/outputs/train/<run_name> \
./scripts/training/start_lerobot_vtla_tmux.sh
```

监控和重连：

```bash
tmux attach -t vtla_30k
tmux capture-pane -p -t vtla_30k -S -40
tail -f /workspace/VTLA_design/outputs/tmux/<log_file>.log
```

单卡：

```bash
VTLA_GPUS=0 \
VTLA_OUTPUT_DIR=/workspace/VTLA_design/outputs/train/single_$(date +%Y%m%d_%H%M%S) \
./scripts/training/train_lerobot_vtla.sh
```

四卡：

```bash
nvidia-smi
VTLA_GPUS=0,1,2,3 \
VTLA_OUTPUT_DIR=/workspace/VTLA_design/outputs/train/ddp4_$(date +%Y%m%d_%H%M%S) \
./scripts/training/train_lerobot_vtla.sh
```

`VTLA_GPUS` 必须显式提供，避免误占共享服务器资源；未设置时启动器会直接失败。`VTLA_NUM_PROCESSES` 默认从 GPU 列表推导，也可以显式传入，但必须与列表长度一致。只有在四张卡都空闲时才运行四卡命令。

临时覆盖参数时直接追加 LeRobot CLI 参数，例如：

```bash
VTLA_GPUS=0 \
./scripts/training/train_lerobot_vtla.sh \
  --batch_size=4 \
  --steps=1000 \
  --save_freq=500
```

## checkpoint 和恢复训练

每个保存点包含：

```text
checkpoints/<step>/
├── pretrained_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── policy_preprocessor.json
│   ├── policy_postprocessor.json
│   ├── *_processor.safetensors
│   └── train_config.json
└── training_state/
    ├── optimizer_state.safetensors
    ├── optimizer_param_groups.json
    ├── rng_state.safetensors
    ├── scheduler_state.json
    └── training_step.json
```

`checkpoints/last` 是指向最新 step 的 symlink。恢复到原 run 目录：

```bash
run_dir=/workspace/VTLA_design/outputs/train/<run_name>
VTLA_CONFIG=${run_dir}/checkpoints/last/pretrained_model/train_config.json \
VTLA_OUTPUT_DIR=${run_dir} \
VTLA_GPUS=0,1,2,3 \
./scripts/training/train_lerobot_vtla.sh \
  --resume=true \
  --steps=30000
```

`steps` 是目标总 step，不是额外增加的 step。为了精确恢复数据顺序，batch size 和进程数应与原 run 保持一致；LeRobot 会从 optimizer、RNG 和 training step 状态恢复。

## 部署接口与边界

训练 checkpoint 已验证可以在同一固定镜像内独立回载。正式机器人控制侧应保存并加载整个 `pretrained_model/`，不能只复制 `model.safetensors`，因为 resize 和 state/action normalization 参数位于 processor 文件中。

LeRobot 的标准推理顺序为：

```python
from lerobot.common.control_utils import predict_action
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot_policy_vtla import VTLAPolicy

checkpoint = "/path/to/checkpoints/last/pretrained_model"
config = PreTrainedConfig.from_pretrained(checkpoint)
policy = VTLAPolicy.from_pretrained(checkpoint, config=config, strict=True)
preprocessor, postprocessor = make_pre_post_processors(
    config, pretrained_path=checkpoint
)

# observation 必须提供 checkpoint config 中记录的 28D state 和 7 个相机 key。
# action = predict_action(observation, policy, torch.device(config.device),
#                         preprocessor, postprocessor, config.use_amp)
```

当前仓库已经保证模型和 processor 的可加载性，但没有替用户猜测 RealMan/控制器 SDK、相机采集线程、控制频率、安全限位或 16D action 到底层命令的映射。这些属于下一步 robot adapter；在接口定义明确前，不应把训练容器直接当成可上真机的安全控制镜像。建议以当前 Dockerfile 为基础再派生一个较小的 runtime image，并显式加入机器人 SDK 与安全层。

## 已完成验证记录

交付前已执行：

- 最终聚合数据 metadata/parquet/维度检查，首中末真实视频解码：通过；
- Python compile：通过；
- VTLA 28D state / 16D action、scheduler 边界/缩放/恢复和 uint8 转换单元测试：5 passed；
- 持久容器 CUDA、4 × L40S、LeRobot/插件、120G SHM 和 restart policy：通过；
- scheduler 在 step 0/1K/27K/28.5K/30K 的 LR 边界及 JSON 状态保存/恢复：通过；
- 新镜像小模型 1 step、checkpoint 全量保存，并从 step 1 恢复到 step 2：通过；
- checkpoint 严格回载 + 聚合数据真实 uint8 样本处理和 28D → 16D 推理：通过；
- 完整 108M 模型，单卡 batch 32、8 workers、prefetch 2：通过，记录峰值约 7.26 GB；
- GPU 0/3 NCCL 双 rank all-reduce：通过；
- 完整 108M 模型双卡 DDP，每 GPU batch 32/8 workers：通过，记录峰值约 7.66 GB/GPU；
- 双卡压力测试期间 `/dev/shm` 观测使用约 103 MB，主机仍有约 158 GiB 可用内存；
- tmux 包装脚本 1-step 训练与退出状态记录：通过；
- 完整 108M 模型 1-step train + held-out eval 回归：通过，`eval_num_workers=0`，`eval_loss=0.8003`；
- W&B 自建端点登录、在线 run 创建及 step 100 metrics 同步：通过；
- 本次新增代码 Ruff 检查：通过。

镜像 ID 为 `sha256:ea3dea88978da70878d8ad256a722f37d04295afa67b9546d899250f943f474d`。首轮 30K 正式训练日志确认在 5K、10K、15K、20K、25K 和 30K 均触发了 checkpoint 保存。

首轮正式训练已于 2026-08-27 在 tmux `vtla_30k` 中正常完成：物理 GPU 0/3、有效 batch 64、30,000 steps、最终进程状态 0。输出目录为：

```text
/workspace/VTLA_design/outputs/train/2026-08-27_vtla_manipulationnet_30k_gpu03_v2
```

W&B run 为 `vtla-manipulationnet-30k/1kovnw8u`，最终训练 loss 为 `0.162`，held-out `eval_loss` 为 `0.1870`，末端 LR 为 `1.0e-6`。当前宿主机保留 `020000`、`030000` 两个实体 checkpoint，`last` 指向 `030000`；不要仅根据保存频率假定所有中间 checkpoint 仍在磁盘。先前的 `1yp8p36n` 是发现 eval worker 内存风险后在首个 checkpoint 前主动中止的预运行，不用于模型选择或恢复。

## 常见问题

### `Output directory already exists`

非恢复训练必须使用新目录。只有带 `--resume=true` 且配置指向 checkpoint 时才复用原目录。

### bind mount 下 `Permission denied`

镜像构建时会按宿主用户 UID/GID 创建 `vtla` 用户。请用 `./docker/build_train_image.sh` 构建，不要用旧容器以 root 创建 outputs。若目录已经由 root 创建，应只修复该精确目录的所有权，不要递归修改整个 workspace。

### 多卡停在 `Creating dataset`

先运行 `docker/verify_distributed.py`。本机必须保留 `NCCL_CUMEM_ENABLE=0`、`NCCL_P2P_DISABLE=1`；当前启动器已默认设置。

### 第一次正式训练下载 ResNet18 权重

这是 torchvision ImageNet 初始化权重，约 45 MB，保存在 `vtla-cache` volume。后续训练复用缓存。若要求完全离线，应在断网前先完成一次正式模型 smoke，并备份该 volume 或将权重纳入受控 artifact 存储。

### 聚合脚本拒绝已有输出目录

这是保护机制。为重建选择一个全新目录，完成校验后再显式切换配置；不要删除或覆盖唯一的已验证数据集。
