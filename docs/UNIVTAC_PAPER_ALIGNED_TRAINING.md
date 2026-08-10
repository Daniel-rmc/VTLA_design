# UniVTAC 论文对齐训练协议

本协议用于在 UniVTAC 官方数据上训练 VTLA，并使数据选择、优化步数和样本曝光量尽可能与 UniVTAC 论文中的 ACT 实验一致。机器可读配置见 [`configs/univtac_paper_aligned_stage2.json`](../configs/univtac_paper_aligned_stage2.json)。

## 一手来源与结论

- [UniVTAC 原论文](https://arxiv.org/abs/2602.10093) Appendix B：每任务 50 条合成轨迹、batch size 64、4000 个优化步；视觉/触觉编码器学习率均为 `1e-5`，权重衰减 `1e-4`；Transformer 4 层 encoder、7 层 decoder；预测未来 50 步并做 temporal aggregation。
- [官方训练配置](https://github.com/univtac/UniVTAC/blob/main/policy/ACT/train_config.yml)：确认 `state_dim=8`、`kl_weight=10`、ResNet-18、hidden size 512、FFN 3200、8 heads、dropout 0.1，以及上述优化配置。
- [官方数据加载代码](https://github.com/univtac/UniVTAC/blob/main/policy/ACT/utils.py)：固定 seed 1 打乱 50 个 episode，前 80% 进入训练、后 20% 用于验证；DataLoader 默认不丢弃最后一个不足 64 的 batch。
- 本地官方预处理 `policy/ACT/process_data.py`：按数值顺序选取原始 episode 0–49，把原始 9D joint 的列 0–7 作为 8D qpos/action，并以 `x[t] -> x[t+1]` 构造动作对。

论文正文还说明每个任务用 100 次闭环 rollout 测试。最终横向比较必须使用同样的 100 个测试 seed，不能用离线 loss 代替成功率。

## 本次训练任务

| 任务 | 相机 | 前 50 条数据 | 训练/验证时序样本 | 4000 步实际样本曝光 | 等效训练集遍历 |
| --- | --- | ---: | ---: | ---: | ---: |
| `grasp_classify` | head | 2,653 | 2,122 / 531 | 249,682 | 117.66× |
| `insert_HDMI` | head | 5,850 | 4,680 / 1,170 | 252,976 | 54.05× |
| `insert_hole` | head | 8,528 | 6,877 / 1,651 | 254,705 | 37.04× |
| `insert_tube` | head + wrist | 9,377 | 7,515 / 1,862 | 254,779 | 33.90× |
| `lift_bottle` | head + wrist | 15,284 | 12,234 / 3,050 | 254,920 | 20.84× |
| `lift_can` | head | 9,122 | 7,392 / 1,730 | 254,912 | 34.48× |
| `pull_out_key` | head | 10,114 | 8,092 / 2,022 | 254,884 | 31.50× |
| `put_bottle_in_shelf` | head | 14,615 | 11,627 / 2,988 | 255,559 | 21.98× |

名义曝光量均为 `4000 × 64 = 256,000`。表中实际值略小，是因为官方 DataLoader 不设置 `drop_last=True`，每轮最后一个 batch 会短于 64；VTLA 保留了这一行为。不同任务轨迹长度不同，因此等效遍历次数不同，这是官方逐任务训练协议本身的结果。对同一任务比较不同模型时，episode、split、batch 顺序规则和优化步数保持一致。

发布数据的 `clean/metadata.json` 中，前 50 条里 `insert_HDMI` 有 3 条、`insert_tube` 有 2 条、`lift_bottle`/`pull_out_key`/`put_bottle_in_shelf` 各有 1 条记录为 `fail`；`lift_can` 有 18 条未填写 result。官方 `process_data.py` 不读取或过滤 metadata，而是直接取前 50 个 HDF5；为了复现实验输入，本协议同样保留这些轨迹，并在 manifest 中记录结果计数。

## VTLA 对齐设置

- 四个任务均为原生 8D：原始 9D joint 固定选列 `0..7`；
- 分别统计 `qpos=x[t]` 与 `action=x[t+1]` 的均值/方差，统计范围为全部 50 条选中轨迹；
- 一张 GPU 运行一个任务，单任务全局 batch 64；后三张物理 GPU 做任务级并行；
- `max_steps=4000` 精确停止，checkpoint 在 1000/2000/3000/4000 步及结束时保存；
- AdamW、三组学习率均为 `1e-5`，weight decay `1e-4`，不使用 scheduler 和梯度裁剪；
- chunk 50、KL 10、4/7 层 Transformer、FFN 3200、ResNet-18、learned tactile position embedding；
- 加载官方 `encoder.pth` 的 120 个 ResNet 卷积主干张量，固定其 BatchNorm；VTLA 自己的空间投影和融合层仍从头训练；
- 触觉图像保持 `[0,1]`，不做 ImageNet normalization；视觉图像继续使用 ImageNet normalization；
- L1 reduction 与官方 ACT 实现一致，padding loss 权重为 0；
- 使用 BF16。论文没有报告数值精度，所以该项属于明确记录的实现差异。

官方训练循环使用 `step_count > num_steps`，按当前开源代码会多执行一步。这里以论文写明的总预算为准，精确执行 4000 步，并在 `config.json`/checkpoint 中记录实际 `optimizer_step` 与 `training_examples_seen`。

## 启动与产物

先生成/复核各任务 manifest，再启动：

```bash
scripts/training/start_official_tasks.sh
```

首轮调度为：GPU 1 依次运行 `grasp_classify -> insert_tube`，GPU 2 运行 `insert_HDMI`，GPU 3 运行 `insert_hole`。完整官方数据下载并验证后，可在同一 run group 中续跑：

```bash
scripts/training/continue_official_tasks.sh runs/stage2/official_<timestamp>_<git>_gpu123
```

续跑调度为：空闲的 GPU 3 依次运行 `lift_bottle -> lift_can`；GPU 2 在确认 `insert_HDMI/exit_code=0` 后依次运行 `pull_out_key -> put_bottle_in_shelf`。论文 Appendix B 指定 `lift_bottle` 与 `insert_tube` 使用 head+wrist，其余任务只使用 head。GPU 1 原有队列不变。每个任务目录包含：

```text
runs/stage2/official_<timestamp>_<git>_gpu123/<task>/
├── config.json
├── protocol.json
├── dataset_manifest.json
├── launch_command.sh
├── train.log
├── metrics.jsonl
├── checkpoints/
└── exit_code
```

`config.json` 是单次实验的事实来源；它记录精确 episode 清单、split、归一化统计、模型参数、软件/GPU 环境、Git 状态、目标与实际曝光量。训练完成后再用相同 100 个 UniVTAC test seeds 做闭环评测。

## 论文中可直接比较的成功率

| 方法 | Insert Hole | Insert HDMI | Insert Tube | Grasp Classify |
| --- | ---: | ---: | ---: | ---: |
| ACT（vision only） | 19 | 15 | 45 | 50 |
| VITaL | 25 | 6 | 34 | 100 |
| ACT + UniVTAC Encoder | 24 | 28 | 56 | 99 |
| Scratch ablation | 3 | 16 | 45 | 45 |

这些数字是 100 次闭环 rollout 的成功率百分比。VTLA 保留自己的双向视觉—触觉融合与动作头，因此这是训练预算和数据协议对齐后的模型比较，不是把 VTLA 改造成 ACT。
