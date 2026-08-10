# VTLA 训练指南

## 环境

- 项目根目录：`/home/rmc/workspace/VTLA_design`
- UniVTAC：`/home/rmc/workspace/UniVTAC`
- Python：`/home/rmc/miniconda/envs/UniVTAC/bin/python`
- 官方数据：`/home/rmc/workspace/UniVTAC/data/official/grasp_classify/clean`

命令均从项目根目录运行：

```bash
cd /home/rmc/workspace/VTLA_design
```

## 推荐启动方式

物理 GPU 1/2/3 上运行 Stage 2：

```bash
scripts/training/start_training_multigpu.sh stage2 3 1,2,3
```

单 GPU 调试：

```bash
scripts/training/start_training.sh stage2 1
```

启动器首先检查 GPU 剩余显存和官方数据完整性，然后在独立 tmux session 中启动训练。它不会占用不满足 `MIN_FREE_MEMORY_MB` 的设备。

## 当前默认配置

| Stage | 目标 | batch/GPU | epochs | 备注 |
| --- | --- | ---: | ---: | --- |
| Stage 1 | 触觉编码器预训练 | 16 | 100 | RGB + marker 自监督 |
| Stage 2 | 视觉—触觉端到端动作训练 | 64 | 150 | BF16、原生 8D、90/10 验证切分 |
| Stage 3 | 触觉残差分支训练 | 32 | 200 | 自动加载最新 Stage 2 checkpoint |

Stage 2/3 默认使用双相机、双触觉传感器、chunk size 50 和原始 joint 列 `0..7`。多卡有效 batch 为 `batch/GPU × GPU 数量`。

## 覆盖默认参数

启动器支持环境变量覆盖：

```bash
BATCH_SIZE=32 NUM_EPOCHS=20 NUM_WORKERS=4 \
  scripts/training/start_training_multigpu.sh stage2 3 1,2,3
```

常用变量：

- `DATASET_DIR`：数据根目录；
- `DATASET_MANIFEST`：已验证 manifest；
- `GPU_IDS`：未提供第三个位置参数时使用的 GPU 列表；
- `MIN_FREE_MEMORY_MB`：启动前的单卡最低空闲显存；
- `NCCL_P2P_DISABLE`：当前主机默认 `1`；
- `BATCH_SIZE`、`NUM_EPOCHS`、`NUM_WORKERS`、`SAVE_FREQ`。

需要完全自定义模型参数时，可直接使用模块入口：

```bash
PYTHONPATH=/home/rmc/workspace/VTLA_design:/home/rmc/workspace/UniVTAC \
  /home/rmc/miniconda/envs/UniVTAC/bin/python \
  -m scripts.training.train_vtla --help
```

## Run 产物

每次启动创建：

```text
runs/<stage>/<timestamp>_<git>_gpu<ids>/
├── config.json
├── launch_command.sh
├── train.log
├── metrics.jsonl
├── checkpoints/
└── exit_code
```

`config.json` 是配置事实来源，包含完整 CLI、Git commit/dirty 状态、GPU 映射、软件版本、数据清单、episode 切分、归一化统计和参数量。checkpoint 内也嵌入同一运行配置和数据统计。

## 监控

```bash
scripts/training/check_training.sh
tmux ls
tail -f runs/<stage>/<run-name>/train.log
nvidia-smi
```

训练完成的可靠判据是 `exit_code` 为 `0`、目标 epoch 的 metrics 存在且 checkpoint 写入完成。

## Checkpoint 选择

不要只按训练 loss 选择。Stage 2 会每 5 epochs 在固定的 episode 级验证集上评测，并把总验证 loss 最优模型写入 `stage2_best.ckpt`。部署前还应使用：

```bash
/home/rmc/miniconda/envs/UniVTAC/bin/python \
  -m scripts.evaluation.eval_vtla_offline --help
```

当前部署每次只执行预测 chunk 的第 0 步，因此 first-step raw-joint MAE 是重要候选指标，最终仍以 UniVTAC 闭环成功率为准。

## 故障排查

### GPU 被占用或 OOM

启动器会拒绝低于显存阈值的设备。降低 `BATCH_SIZE` 或显式选择其他物理 GPU，不要绕过检查占用 GPU 0。

### NCCL 初始化或 collective 挂起

本机 NCCL 2.21.5 的 P2P 路径已确认会挂起。保持 `NCCL_P2P_DISABLE=1`，并先执行根 README 中的三卡 NCCL smoke check。

### 数据不完整

Stage 2/3 默认要求 100 个官方 `clean` HDF5 和 manifest。先运行 `scripts.diagnostics.validate_univtac_dataset`，不要以少量本地 demo 冒充正式数据。

### 训练与验证维度不一致

官方观测是 9D，当前模型必须记录 `state_dim=8` 与 `joint_indices=[0,1,2,3,4,5,6,7]`。不要通过 padding 或复制夹爪维度静默兼容。

### 后台任务是否仍在运行

检查 tmux、`exit_code` 和训练进程。日志暂时没有新行不等于失败，只有进程退出且非零退出码才判定运行失败。
