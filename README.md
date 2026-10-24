# VMM — 相位式 Eulerian 微小运动放大

一个**确定性、无可训练参数**的视频微小运动放大（Video Motion Magnification）实现：
复数可转向金字塔 → 局部相位增量 → 零相位时间带通 → 相位调制 → 重建。

核心公式（`alpha` 是**额外**增益，总增益 ≈ `1 + alpha`）：

```
C'(x,t) = C(x,t) · exp( j · alpha · w(x) · Phi_bp(x,t) )
```

`|C'| = |C|`：**只搬移相位、不改变幅值**，因此不破坏纹理对比度，输出由同一个小波金字塔精确重建。

## 关键性质（均有测试与实验记录）

| 性质 | 实测 |
|---|---|
| `alpha=0` 严格恒等 | `max|out-in| = 2.1e-06`（阈值 1e-4） |
| 合成已知运动增益 | `alpha=1/2/5/10` → **1.96 / 2.93 / 5.80 / 10.47**（理论 2/3/6/11） |
| 运动方向 / 波形 | 方向不反；波形相关 **≥ 0.97** |
| 纯静止输入 | 不产生任何假运动（误差 2.1e-06） |
| 长视频分块 | overlap-save，接缝比 ≈ 1.07 |

**明确的能力边界**（不要过度宣称）：

* 静止区也会被放大：活动/静止选择性只有 **约 1.5**；
* 真实视频的运动常在 **0.03–0.05 px**，接近测量器噪声底 ⇒ 只能报能量比与估计器分歧，**不能声称精确倍率**；
* 必须显式给定目标频带，运动须在带内；
* 零相位滤波使用未来帧 ⇒ **离线算法**；
* 大倍数（≥20×）会递减：合成真值上 50× 时全局位移 SSIM 降到 0.64、局部增益饱和在 ~28×。

## 安装

```bash
python -m pip install -r requirements.txt
```

（`pyrtools>=1.0.11` 提供复数可转向金字塔；LPIPS 评估另需 `torch` + `lpips`，见 requirements 注释。）

## 快速开始

```bash
# 单视频放大：总增益 ≈ 1+alpha
python scripts/run_vmm.py --video input.mp4 --out out.mp4 \
    --alpha 5 --f-lo 0.2 --f-hi 0.8 --channels color --side-by-side --metrics

# 直接按"总倍数"输出多档（5/10/20/50×），并生成左右对比视频
python scripts/amplify_levels.py --video input.mp4 --targets 5 10 20 50
```

用 B2 可靠性门控（推荐配置，需 Python API）：

```python
from vmm.video_io import read_video
from vmm.phase_magnifier import PhaseBasedMagnifier
from vmm.reliability import ReliabilityConfig

video, meta = read_video("input.mp4", max_frames=240, scale=0.5)
mag = PhaseBasedMagnifier(fps=meta.fps, f_lo=0.2, f_hi=0.8, levels=4, order=3,
                          channels="color",
                          reliability_cfg=ReliabilityConfig(mode="B2"))
out, diag = mag.amplify(video, alpha=5)
```

## 仓库结构

```
vmm/                        核心包
  phase_pyramid.py          复数可转向金字塔（pyrtools 薄封装：analyze/synthesize）
  phase_magnifier.py        主流程：相位收集 → β 计算 → 重建
  temporal_filter.py        零相位 Butterworth 带通（filtfilt）+ 频响
  reliability.py            可靠性门控：M1–M3 / B1–B3 / 逐带上限 / amp_floor
  coherent.py               多帧相干积累（Phase G；默认关闭）
  metrics.py                5 套独立位移估计器、分区带内能量、伪影、接缝
  synthetic.py              已知真值的合成序列（H/V/局部/静止/噪声）
  video_io.py               视频读写（cv2）、灰度/缩放
scripts/                    实验与工具入口
tests/                      56 个测试（恒等、位移、滤波、分块、门控、相干）
configs/vmm_stage2/         阶段 2 配置模板与噪声标定
docs/                       审计 / 设计 / 结果 / 决策
```

## 测试

```bash
python -m pytest tests/ -q      # 56 passed
```

## 验证协议（怎么判断"放大对不对"）

本仓库不接受"看起来像"或"loss 下降了"作为证据，统一用：

1. **α=0 恒等**（硬门槛 1e-4）；
2. **合成真值**：已知位移曲线 → 增益、方向、波形相关；
3. **多估计器交叉验证**（相位相关 / 金字塔粗到细 / Lucas–Kanade），以真值甄别可信估计器，分歧作为不确定度；
4. **分区能量**（活动 / 静止 / 边界环 / 远背景）与**带外 guard**（是否引入新频率成分）；
5. **伪影检查**（NaN/Inf/黑帧/饱和）与**分块接缝**。

## 引用与致谢

复数可转向金字塔使用 [pyrtools](https://github.com/LabForComputationalVision/pyrtools)。

## 许可证

未指定（如需要，请在仓库中补充 LICENSE）。
