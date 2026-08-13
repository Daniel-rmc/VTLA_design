# 双流VTLA实验启动清单

**日期**: 2026-08-13  
**状态**: ✅ 所有准备工作完成

---

## ✅ 完成的准备工作

### 1. 架构实现 ✅
- [x] `models/dual_stream_transformer.py` - 双流Transformer
- [x] `models/fusion_action_head.py` - 融合动作头（4种策略）
- [x] `models/dual_stream_vtla_policy.py` - 完整策略
- [x] 所有模块独立测试通过

### 2. 训练脚本 ✅
- [x] `scripts/training/train_dual_stream.py` - 训练主脚本
- [x] `scripts/training/start_dual_stream_training.sh` - 启动脚本
- [x] `scripts/training/smoke_test_dual_stream.py` - 快速测试
- [x] 脚本已添加执行权限

### 3. 配置文件 ✅
- [x] `configs/dual_stream_stage2.json` - 训练配置
- [x] `univtac_adapter/deploy_dual_stream.yml` - 部署配置

### 4. 数据与Baseline ✅
- [x] 3个优先任务的manifest已确认
  - `data_manifests/put_bottle_in_shelf_official_first50.json`
  - `data_manifests/insert_HDMI_official_first50.json`
  - `data_manifests/lift_can_official_first50.json`
- [x] ACT+UniVTAC checkpoint已找到（所有8个任务）
- [x] 官方encoder.pth已确认

### 5. 文档完备 ✅
- [x] `docs/EXPERIMENT_PLAN.md` - 完整实验计划
- [x] `docs/ACT_UNIVTAC_CHECKPOINT_REPORT.md` - Baseline报告
- [x] `docs/EXPERIMENT_READY_REPORT.md` - 准备状态
- [x] `docs/STATUS_SUMMARY.md` - 快速总览
- [x] `docs/dual_stream_architecture.md` - 架构设计
- [x] `docs/architecture_critique.md` - 原架构分析

---

## 🚀 启动训练的步骤

### Step 1: 运行Smoke Test（正在进行）
```bash
cd /home/rmc/workspace/VTLA_design
python scripts/training/smoke_test_dual_stream.py
```

**预期输出**：
- ✓ Model created successfully
- ✓ Inference successful
- ✓ Training forward successful
- ✓ Backward pass successful
- ✓ Checkpoint save/load successful
- ✓ All tests PASSED!

**如果失败**：检查错误信息，修复后重新测试

---

### Step 2: 启动阶段1训练（insert_HDMI pilot）

```bash
cd /home/rmc/workspace/VTLA_design

# 使用GPU 1
./scripts/training/start_dual_stream_training.sh insert_HDMI 1
```

**训练参数**：
- 任务: insert_HDMI
- GPU: 1张L40S
- Batch size: 8
- Epochs: 2000
- 前50条数据，seed 1 split (80/20)
- 预计时间: 8-12小时

**监控训练**：
```bash
# 查看实时日志
tail -f runs/dual_stream/dual_stream_insert_HDMI_*/train.log

# 查看指标
tail -f runs/dual_stream/dual_stream_insert_HDMI_*/metrics.jsonl
```

---

### Step 3: 训练完成后评测

```bash
# 找到run目录
RUN_DIR=$(ls -td runs/dual_stream/dual_stream_insert_HDMI_* | head -1)
echo "Run directory: $RUN_DIR"

# 启动UniVTAC评测（100 seeds）
python scripts/evaluation/run_univtac_eval.py \
  --run-dir $RUN_DIR \
  --deploy-config univtac_adapter/deploy_dual_stream.yml \
  --gpu 1 \
  --total-num 100
```

**预计评测时间**: 8-10小时

---

### Step 4: 结果分析

```bash
# 聚合评测结果
python scripts/evaluation/summarize_univtac_eval.py \
  --result-root /home/rmc/workspace/UniVTAC/eval_result/VTLA/insert_HDMI/deploy_dual_stream \
  --start-seed 1000000 \
  --end-seed 1000099 \
  --output $RUN_DIR/eval/univtac/aggregate_result.json

# 查看成功率
cat $RUN_DIR/eval/univtac/aggregate_result.json | grep success_rate
```

**期望结果**：
- Baseline (ACT+UniVTAC): 14%
- 目标 (Dual-Stream): ≥20%

**决策点**：
- ✅ **如果≥20%**: 进入阶段2（3任务验证）
- ⚠️ **如果15-20%**: 分析fusion权重，考虑调整
- ❌ **如果<15%**: 深入调试，可能需要架构调整

---

## 📊 实验目标回顾

### 阶段1：单任务pilot (2-3天)
| 任务 | Baseline | 目标 | 状态 |
|------|----------|------|------|
| insert_HDMI | 14% | ≥20% | 🔄 待执行 |

### 阶段2：3任务验证 (5-7天)
| 任务 | Baseline | 目标 | 期望提升 |
|------|----------|------|----------|
| put_bottle_in_shelf | 8% | ≥13% | +5% |
| insert_HDMI | 14% | ≥20% | +6% |
| lift_can | 29% | ≥35% | +6% |

---

## 🔍 故障排查

### 训练相关

**问题**: OOM (Out of Memory)
**解决**: 减小batch_size（8→4）或使用BF16混合精度

**问题**: Loss不收敛
**解决**: 
1. 检查学习率是否合适
2. 验证数据加载是否正确
3. 检查梯度是否正常（添加梯度裁剪）

**问题**: 训练速度慢
**解决**:
1. 确认使用GPU (`nvidia-smi`)
2. 检查数据加载是否成为瓶颈
3. 考虑使用更少的dataloader workers

### 评测相关

**问题**: UniVTAC评测失败
**解决**:
1. 确认Isaac Sim EULA已接受
2. 检查deploy配置文件是否正确
3. 验证checkpoint可以正常加载

**问题**: 成功率异常低（<5%）
**解决**:
1. 检查推理时是否使用了deterministic_latent
2. 验证temporal aggregation是否启用
3. 检查归一化统计是否正确应用

---

## 📁 关键文件路径速查

### 训练
- 启动: `./scripts/training/start_dual_stream_training.sh <task> <gpu>`
- 日志: `runs/dual_stream/dual_stream_<task>_*/train.log`
- Checkpoint: `runs/dual_stream/dual_stream_<task>_*/checkpoints/dual_stream_best.ckpt`

### 评测
- 脚本: `scripts/evaluation/run_univtac_eval.py`
- 结果: `runs/dual_stream/dual_stream_<task>_*/eval/univtac/aggregate_result.json`

### 配置
- 训练: `configs/dual_stream_stage2.json`
- 部署: `univtac_adapter/deploy_dual_stream.yml`

### 数据
- Manifest: `data_manifests/<task>_official_first50.json`
- Encoder权重: `/home/rmc/workspace/UniVTAC/policy/ACT/encoder/checkpoints/resnet18/official/encoder.pth`
- Baseline checkpoint: `/home/rmc/workspace/UniVTAC/policy/ACT/act_ckpt/act-<task>/demo-50/train_config/policy_last.ckpt`

---

## 💡 快速命令参考

```bash
# 查看GPU状态
nvidia-smi

# 查看训练进度
tail -f runs/dual_stream/*/train.log

# 查看最新run
ls -lt runs/dual_stream/ | head -5

# 查看成功率
cat runs/dual_stream/*/eval/univtac/aggregate_result.json | jq .success_rate

# 比较双流 vs baseline
echo "Dual-Stream:" && cat runs/dual_stream/*/eval/univtac/aggregate_result.json | jq .success_rate
echo "ACT+UniVTAC:" && cat /home/rmc/workspace/UniVTAC/policy/ACT/act_ckpt/act-insert_HDMI/demo-50/train_config/log.log | grep "Final Result"
```

---

## ✅ 最终检查清单

在启动训练前，确认：

- [ ] Smoke test通过
- [ ] GPU可用 (`nvidia-smi`)
- [ ] 数据集路径正确
- [ ] Manifest文件存在
- [ ] 有足够的磁盘空间（~50GB per task）
- [ ] 训练脚本有执行权限
- [ ] 已阅读实验计划

**如果所有检查都通过，可以开始训练！** 🚀

---

## 📞 需要帮助？

参考文档：
- [完整实验计划](EXPERIMENT_PLAN.md)
- [双流架构设计](dual_stream_architecture.md)
- [原架构分析](architecture_critique.md)
- [快速状态总览](STATUS_SUMMARY.md)

---

**祝实验顺利！** 🎉
