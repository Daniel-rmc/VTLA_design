# 双流VTLA训练状态报告

**启动时间**: 2026-08-13  
**任务**: insert_HDMI (阶段1 pilot)  
**状态**: ✅ 训练正在运行

---

## 🚀 训练配置

### 基础参数
- **分支**: `feature/dual-stream-architecture` ✓
- **GPU**: GPU 0 (NVIDIA L40S)
- **Tmux会话**: `dual_stream_training`

### 数据配置
- **任务**: insert_HDMI
- **Episodes**: 100 (前50条用于训练)
- **Samples**: 11,700 timesteps
- **Batch size**: 8
- **State dim**: 9 (修正后)
- **Chunk size**: 50
- **相机**: cam_high
- **触觉**: tac_left, tac_right

### 模型配置
- **架构**: Dual-Stream VTLA
- **Hidden dim**: 512
- **Encoder layers**: 4
- **Decoder layers**: 6
- **Fusion type**: gated
- **总参数**: ~110M

### 训练参数
- **Epochs**: 2000
- **Learning rate**: 1e-4
- **LR backbone**: 1e-5
- **LR tactile**: 1e-5
- **Weight decay**: 1e-4
- **每epoch batches**: 1,463

---

## 📊 当前状态

### 训练进度
```
Epoch 1/2000: 22% (318/1463)
Speed: ~8 it/s
Loss: ~2.3 (L1=0.39, KL=0.16)
```

### GPU使用情况
- **显存使用**: 3.5GB / 46GB (8%)
- **GPU利用率**: 40%
- **显存利用率**: 15%

**分析**: GPU利用率较低，可以通过增加batch size来加速训练

---

## ⏱️ 时间估算

### 当前配置 (batch_size=8)
- **每epoch时间**: ~3分钟 (1463 batches / 8 it/s)
- **2000 epochs**: ~100小时 (4.2天)
- **建议优化**: 增加batch size

### 优化后 (batch_size=16, 推荐)
- **每epoch时间**: ~1.5分钟
- **2000 epochs**: ~50小时 (2.1天)

### 激进优化 (batch_size=32)
- **每epoch时间**: ~45秒
- **2000 epochs**: ~25小时 (1天)

---

## 💡 优化建议

### 1. 增加Batch Size (推荐)
当前GPU利用率低，可以增加batch size来加速：

```bash
# 停止当前训练
tmux send-keys -t dual_stream_training C-c

# 修改启动脚本中的 --batch-size 8 改为 --batch-size 16 或 32
# 重新启动
```

### 2. 减少Epochs (可选)
- 2000 epochs可能过多
- 建议先训练500-1000 epochs观察收敛情况
- 可以随时从checkpoint恢复继续训练

### 3. 使用混合精度训练 (高级)
- 添加AMP (Automatic Mixed Precision)
- 可以进一步加速并减少显存使用

---

## 🔍 监控命令

### 查看训练进度
```bash
# 进入tmux会话
tmux attach -t dual_stream_training

# 查看最后几行（不进入会话）
tmux capture-pane -t dual_stream_training -p | tail -20

# 查看训练日志
tail -f /home/rmc/workspace/VTLA_design/training.log
```

### 查看GPU状态
```bash
watch -n 1 nvidia-smi
```

### 查看loss曲线（训练完成后）
```bash
# 找到run目录
RUN_DIR=$(ls -td runs/dual_stream/dual_stream_insert_HDMI_* | head -1)

# 查看metrics
cat $RUN_DIR/metrics.jsonl | jq '.train_loss'
```

---

## 📁 输出文件

训练产生的文件将保存在：
```
runs/dual_stream/dual_stream_insert_HDMI_<timestamp>_<git>/
├── config.json              # 配置
├── metrics.jsonl            # 训练指标
├── checkpoints/             # 模型checkpoint
│   ├── dual_stream_epoch_100.ckpt
│   ├── dual_stream_epoch_200.ckpt
│   ├── ...
│   ├── dual_stream_best.ckpt
│   └── dual_stream_last.ckpt
└── training.log (in parent)
```

---

## ⚠️ 已修复的问题

在启动过程中修复了以下问题：

1. ✅ `dataloader`导入错误：`load_data` → `create_dataloader`
2. ✅ `state_dim`错误：14 → 9（UniVTAC实际维度）
3. ✅ 缺少`lr_vision_backbone`参数
4. ✅ `build_run_config`调用错误：简化配置保存
5. ✅ Batch key错误：`action` → `actions`

---

## 🎯 下一步

### 立即决策
**是否优化batch size？**

**选项A**: 继续当前配置（batch_size=8）
- 优势：保守，不会OOM
- 劣势：训练时间长（~4天）

**选项B**: 增加到batch_size=16（推荐）
- 优势：训练时间减半（~2天）
- 劣势：需要停止并重启

**选项C**: 增加到batch_size=32（激进）
- 优势：训练时间最短（~1天）
- 劣势：可能OOM，需要测试

### 训练完成后
1. 运行UniVTAC评测（100 seeds）
2. 对比ACT+UniVTAC baseline (14%)
3. 决策是否进入阶段2

---

**当前建议**: 让训练继续运行几个epochs，观察loss收敛情况。如果GPU利用率持续低，可以考虑重启并增加batch size。

**监控重点**: 
- Loss是否稳定下降
- 是否有过拟合迹象（如果有验证集的话）
- GPU利用率是否可以进一步提升

---

**报告生成时间**: 2026-08-13  
**训练状态**: ✅ 正在运行
