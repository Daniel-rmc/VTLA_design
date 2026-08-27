# 每日 Todo

> 由 Claude 维护。用于记录当天任务、已完成事项和次日计划。

## 今日完成
- [x] 将 `models/` 目录重构为 `vtla/`、`dual_stream/`、`shared/` 三个子模块
- [x] 更新训练、评测、测试和 UniVTAC adapter 中的模型 import 路径
- [x] 验证 VTLA 和 DualStream adapter 仍然可以正常加载 checkpoint
- [x] 成功运行单元测试和 DualStream smoke test
- [x] 提交模型目录重构 commit：`0c03ceb`
- [x] 创建并维护每日 todo 文件

## 进行中
- [ ] 

## 已完成背景事项
- [x] 完成 DualStream 测评问题诊断，确认 adapter preprocessing 已修复
- [x] 将 DualStream UniVTAC adapter 同步到新的模型包结构
- [x] 确认原始 VTLA 代码仍然保留，并已和 DualStream 代码分目录管理

## 明日待办
- [ ] 决定剩余未跟踪文档和辅助脚本是提交还是清理
- [ ] 如有需要，在新目录结构下重新跑一次 VTLA `insert_hole` / `insert_HDMI` smoke test
- [ ] 在最终评测设置确认稳定后，继续启动完整 DualStream 测评
- [ ] 将新的 DualStream 结果与原始 VTLA baseline 进行对比

## 备注
- 当前仓库已经将原始 VTLA 和 DualStream 实现分别放到 `models/` 下的不同子目录中。
- 当前工作区仍有未跟踪文件：`docs/EVALUATION_READY_REPORT.md`、`docs/NEXT_TRAINING_PLAN.md`、`docs/TRAINING_CONFIG_ANALYSIS.md`、`scripts/evaluation/run_eval.sh`。
