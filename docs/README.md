# 文档索引

| 文档 | 内容 |
|---|---|
| `implementation_audit.md` | 实现审计（模块、接口、张量形状、不变量） |
| `design_decisions.md` | 关键设计决策与理由（含被否定的方案） |
| `baseline_results.md` | 阶段 1 基线：增益/方向/波形/静止/分块 |
| `stage2_design.md` | 可靠性门控（M1–M3、B1–B3）、逐带上限、`amp_floor`、抑噪 |
| `stage2_results.md` | 阶段 2 消融与真实视频结果（含负结果与冲突） |
| `phaseG_design.md` | 多帧相干积累：信号定义、假设、边界与 Gate |
| `phaseG_results.md` | 相干积累消融、噪声边界、失败案例 |
| `phaseG_decision.md` | 是否重开学习式融合的决策（NO-GO） |

> 说明：本仓库只发布 **VMM 放大算法**相关代码与文档。
> 原项目中与 DiLA/DINOv2 表征评估、融合决策相关的文档/脚本未包含在此仓库；
> 因此个别文档里的跨文件引用（如 `dino_dila_motion_probe.md`）在本仓库中不可点击，属预期。
