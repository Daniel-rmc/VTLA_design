# Scripts

## LeRobot 推荐入口

- `data/aggregate_manipulationnet.py`：将 15 个本地 LeRobot v3 子数据集合并为一个多任务数据集；
- `data/validate_manipulationnet.py`：校验元数据、parquet、视频引用、维度和真实视频解码；
- `data/upload_to_modelscope.sh`：参数化上传一个或多个文件/目录到 ModelScope dataset/model 仓库，支持安全 token 输入与 dry-run；默认清除本机 HTTP(S)/ALL 代理并直连，只有显式传入 `--use-proxy` 才继承代理；
- `training/train_lerobot_vtla.sh`：单卡或 Accelerate/DDP 正式训练入口；
- `training/start_lerobot_vtla_tmux.sh`：在容器内创建可重连的 tmux 正式训练 session；
- `training/validate_checkpoint.py`：严格回载 checkpoint 与处理器，并用真实数据样本推理。

完整命令见 [LeRobot 训练与容器指南](../docs/LEROBOT_TRAINING.md)。以下内容为原有 UniVTAC/HDF5 工具。

所有命令默认从项目根目录 `/home/rmc/workspace/VTLA_design` 执行。Python 入口统一使用模块方式（`python -m ...`），这样移动脚本后仍能稳定导入根目录的 `models`、`dataloader` 和 `training_utils`。

## training

| 文件 | 用途 |
| --- | --- |
| `training/start_training_multigpu.sh` | 选择/校验物理 GPU，在 tmux 中启动可复现 DDP 训练 |
| `training/start_official_tasks.sh` | 在 GPU 1/2/3 上按 UniVTAC 论文预算并行训练四个官方任务 |
| `training/start_training.sh` | 单 GPU 后台训练 |
| `training/check_training.sh` | 显示 VTLA tmux、最近 run、日志尾部和 GPU 状态 |
| `training/train_vtla_multigpu.py` | DDP 三阶段训练实现 |
| `training/train_vtla.py` | 单 GPU 三阶段训练实现 |

推荐入口：

```bash
scripts/training/start_training_multigpu.sh stage2 3 1,2,3
```

论文对齐的四任务训练：

```bash
scripts/training/start_official_tasks.sh
```

## evaluation

| 模块 | 用途 |
| --- | --- |
| `scripts.evaluation.eval_vtla_offline` | 在 checkpoint 记录的 train/validation episode 上运行确定性离线评测 |
| `scripts.evaluation.run_univtac_eval` | 调用 UniVTAC、固定 GPU，并归档请求/配置/日志/结果 |
| `scripts.evaluation.summarize_univtac_eval` | 按 seed 合并分片，检查缺失、重复、错误、视频和 metadata |
| `evaluation/start_official_suite_eval.sh` | 复用已通过的冒烟检查，在 GPU 0 依次评测尚未覆盖的 7 个官方任务 |
| `scripts.evaluation.summarize_univtac_suite` | 合并指定任务的逐任务结果，计算 suite macro/micro 平均 |

示例：

```bash
/home/rmc/miniconda/envs/UniVTAC/bin/python \
  -m scripts.evaluation.eval_vtla_offline --help
```

## diagnostics

| 模块 | 用途 |
| --- | --- |
| `scripts.diagnostics.validate_univtac_dataset` | 校验官方 HDF5 子集并生成带 SHA-256 的 manifest |
| `scripts.diagnostics.nccl_smoke` | 最小 NCCL all-reduce 健康检查 |

完整命令见项目根目录 [README](../README.md)。
