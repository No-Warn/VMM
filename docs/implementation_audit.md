# 实现审计报告（阶段 0）

> 任务书：`从 DiLA 潜变量放大迁移至 Eulerian 微运动放大`（用户提供）
> 审计性质：**只读**。本阶段未修改任何旧代码文件，仅新增本文件；另安装了 2 个依赖（见 §2，已明确标注）。
> 审计时间：2026-10-23；仓库：`.`（branch `main`，HEAD `b04013d`）

---

## 1. 审计命令（实际执行）

```bash
pwd; git rev-parse --abbrev-ref HEAD; git log --oneline -5; git status --short
find . -maxdepth 2 -type d | sort ; find . -maxdepth 1 -type f | sort
find . -maxdepth 4 -type f \( -name '*amplif*' -o -name '*motion*' -o -name '*video*' \
  -o -name '*phase*' -o -name '*pyramid*' -o -name '*vmm*' \) -not -path '*/.git/*'
grep -rniE "steerable|riesz|monogenic|hilbert|analytic_signal|local_phase|filtfilt|butter" --include="*.py" .
python -m pytest tests/test_amplifier.py -q
python -m pip install --no-deps -q pyrtools ; python -m pip install -q pytest
python -c "…SteerablePyramidFreq 重建自洽性 / 相移-重建线性性…"
ls -la baby.mp4 ; python -c "cv2 元数据"
```

---

## 2. 环境实测（含本阶段发生的环境变更）

| 项 | 实测值 | 备注 |
|---|---|---|
| python | 3.10.18 | conda env `dila` |
| numpy / scipy / cv2 | 2.2.6 / 1.15.3 / 5.0.0 | cv2 含 `Farneback`、`DISOpticalFlow` |
| torch | 2.6.0+cu124 | CUDA 可用，A40，显存空闲 46 GB |
| skimage / matplotlib / PIL | 0.25.2 / 3.10.9 / 12.3.0 | |
| **ffmpeg / ffprobe** | **不存在** | 视频 I/O 必须走 OpenCV |
| 磁盘 `/home` | 49 G 总，46 G 可用 | |
| pip / PyPI | **可达**（`pip download` 实测成功） | |
| **`pyrtools`** | 审计前 **MISSING** → **本阶段已安装 1.0.11（`--no-deps`）** | 复数可转向金字塔 |
| **`pytest`** | 审计前 **MISSING** → **本阶段已安装 9.1.1** | 为运行既有回归套件 |
| `phasepack` | MISSING | 不需要 |

> ⚠️ 环境变更仅上述两项，均可通过 `pip uninstall pyrtools pytest` 回退。

---

## 3. Git 状态（**不得覆盖**）

* branch `main`，HEAD `b04013d Add push_to_github.sh helper…`
* **141 个已跟踪文件处于"已修改未提交"状态**（其中绝大多数在 vendored 目录 `DiLA_for_vp2/vp2/…`）
* **整个旧放大器栈是未跟踪的**：`models/amplification/`（`amplifier.py` / `laplacian_pyramid.py` / `gaussian_blur.py` / `mag_conditioning.py` / `projection.py` / `flow_warp.py` / `__init__.py`）、`configs/amplifier/`
* 其它未跟踪项：`tests/`、`scripts/`、`tools/`、`logs/`、`outputs/`、`demo_amplify.py`、`diag_amplifier.py`、`vmm_amplify.py`、`baby.mp4`、`开发文档.md`、`exit`
* 我此前改动的已跟踪文件：`train.py`(M)、`run_fd4mm_training.sh`(M)、`models/model.py`(M)
* `.gitignore` 已忽略 `__pycache__/`、`outputs/`、`checkpoints/`；**未忽略 `logs/`、`*.mp4`**

**约束（本阶段及后续遵守）**：不执行 `git clean` / `git checkout` / `git stash`；不自动提交；新代码放新目录，不改写旧文件语义；旧检查点、日志、输出一律保留。

---

## 4. 与本任务相关的既有代码：可复用 / 冻结 / 死代码

| 路径 | 实测状态 | 处置 |
|---|---|---|
| `models/amplification/laplacian_pyramid.py` | `LaplacianPyramid`(Torch)：`down/up/decompose/reconstruct/scale`；被旧 pyramid 栈使用 | 仅作参考：阶段 1 需要**复数**金字塔，故新建独立模块 |
| `models/amplification/gaussian_blur.py` | `GaussianBlur`(Torch) | 同上（`tests` 里有归一化用例） |
| `models/amplification/amplifier.py` | `LaplacianAmplifier`（方案 B：Δs 潜变量放大器） | **冻结，不再修补**（任务书 §1 禁止事项） |
| `models/amplification/flow_warp.py` | 我此前的"路线 B"帧对 warp 放大器 | **冻结保留作对照**；不作为连续视频微运动主方案 |
| `models/amplification/__init__.py`、`configs/amplifier/amplifier.yaml` | 放大器构建入口与配置（含 `mode: pyramid|warp`） | 不改动其既有语义 |
| `tests/test_amplifier.py` | **18 个用例当前全部通过**（`pytest -q` → `18 passed in 1.95s`） | **回归基线**：后续任何改动必须保持通过 |
| `tests/` 其余 6 个文件 | 原先因缺 pytest 从未被执行 | 已可用作回归套件 |
| `pretrained/dila_video_vmm.py` | **死代码**：依赖 `ffmpeg`（缺失）、`checkpoints/phase1/fd4mm-a40_final.pt`（缺失）、并调用 `Inverse_World_model.amplify_encoded`（**全仓库 grep 不到该定义**） | 保留不删，不修 |
| `configs/amplifier.yaml`（顶层） | 文件头自述 **“DEAD FILE 已废弃，请勿编辑”**；有效配置在 `configs/amplifier/amplifier.yaml` | 不编辑 |
| `scripts/probe_alpha_monotone.py`、`scripts/pixel_cv_table.py`、`tools/*.py` | 旧分析/补丁脚本 | 只读参考 |
| **相位类实现** | `grep steerable/riesz/monogenic/hilbert/filtfilt/butter` → **空** | **阶段 1 必须新建** |

---

## 5. 视频与素材

| 项 | 实测 |
|---|---|
| 用户测试视频 | `baby.mp4`：960×544，**301 帧**，30.0 fps（未跟踪，保留） |
| 既有衍生视频 | `baby_amp20.mp4`、`baby_amp50.mp4`、`my_amp5.mp4`、`my_amp20.mp4`（均由旧 DiLA 路线产出） |
| FrAM 参考实现 | `/home/user/code/FrAM-Motion-Magnification/`（numpy/scipy/cv2，可用）；含论文参考输出资产 `video_baby_original_vs_fram.mp4` 等 |
| `vmm_amplify.py`（我在阶段 0 之前写的） | 包装 FrAM/EVM 的 CLI，**已跑出** `outputs/vmm/baby_fram20.mp4` + `_sbs.mp4`（14:29）。但它是**幅度/带通式（EVM/FrAM）**，按任务书仅作**对照基线/后续方向**，不是阶段 1 的相位式主实现 |

---

## 6. 技术选型结论（决定阶段 1 的写法）

采用 **`pyrtools` 1.0.11 的 `pyrtools.pyramids.SteerablePyramidFreq(is_complex=True)`** 作为复数可转向金字塔（Simoncelli 构造）。理由与实测：

| 实测项 | 结果 | 意义 |
|---|---|---|
| 构造签名 | `SteerablePyramidFreq(image, height='auto', order=3, twidth=1, is_complex=False)`；方法只有 `recon_pyr` / `steer_coeffs`（构造即分解，**无 `build()`**） | 接口已确认 |
| 系数结构 | 96×96 图、`height=4, order=3` → 18 个条目，其中 **16 个复数带**（键形如 `(level, orient)`），另有 `residual_highpass` / `residual_lowpass` | 可直接对 (level, orientation) 逐带处理 |
| **重建自洽性** | 原始系数直接重建：`max|recon − img| = 3.82e-06` | **α=0 恒等可做到 ~4e-6 量级**（结构化保证，而非"近似一样"） |
| 系数乘 1 后重建 | 误差仍为 `3.82e-06` | 恒等路径无额外扰动 |
| **相移-重建链路** | 全部复数带统一相移 β：β=0.02/0.05/0.10 rad → `max|Δ|`=0.0056/0.0139/0.0276（系数平均幅值 0.1787） | 与 `\|1−e^{jβ}\|·幅值` 量级一致 → **局部相位修改确实转化为图像变化**，非亮度/对比度把戏 |

不引入除 pyrtools 之外的新依赖；阶段 1 不训练任何网络。

---

## 7. 阶段 1 文件计划（**全部新增，不改旧文件**）

```
vmm/
  __init__.py
  video_io.py          # cv2 读写（无 ffmpeg）；帧数/分辨率/fps/时间顺序一致性校验
  phase_pyramid.py     # pyrtools 复数可转向金字塔封装：分解/重建/系数遍历/逐带诊断
  temporal_filter.py   # 时间带通（butter + filtfilt 离线零相位）；奈奎斯特校验；首尾帧与边界检查
  phase_magnifier.py   # PhaseBasedMagnifier(fps,f_lo,f_hi,levels,order,max_gain).amplify(video, alpha)
  synthetic.py         # 合成数据：已知亚像素位移（水平/垂直/局部）、纯噪声、静止（含真值位移曲线）
  metrics.py           # 运动增益 / 活动区-静止区时间能量 / 重建误差 / NaN·Inf·饱和·闪烁
scripts/
  run_vmm.py           # CLI：--video --out --alpha --f-lo --f-hi --levels --order --max-gain --channels
  evaluate_synthetic.py
  compare_gains.py
tests/
  test_phase_identity.py   # α=0 恒等（容差写进配置）
  test_phase_shift.py      # β 单调性 / 方向不反转
  test_video_io.py         # 帧数·分辨率·顺序一致
docs/
  implementation_audit.md  # 本文件
```

阶段 1 配置（显式关闭未验证模块）：

```yaml
magnification:
  method: phase_based
  alpha: 2.0
  low_freq: null      # 由 fps/目标运动决定，运行前必须校验
  high_freq: null
  pyramid_levels: null
  orientations: null
  max_gain: null
reliability: {enabled: false}
denoising:   {enabled: false}
evaluation:  {save_comparison: true, save_metrics: true}
```

所有 `null` 在运行前**必须**校验（频率 → 奈奎斯特、`levels ≤ max_pyr_height`、`order` 与方向数、增益上限），禁止静默采用未说明的默认值。

---

## 8. 兼容性与工程风险列表

| # | 风险 | 事实/依据 | 缓解 |
|---|---|---|---|
| 1 | 旧放大器栈未提交，误覆盖不可恢复 | `models/amplification/` 整个目录 untracked | 新代码独立目录；建议用户先做快照提交（**我不自动 commit**） |
| 2 | 我的既有改动会牵连旧用例 | `tests/test_amplifier.py` 18 例现全过 | 每次改动后重跑该套件作为回归 |
| 3 | pyrtools 1.0.11 × numpy 2.2 兼容性 | 已实测可用 | 若后续报错，退回自研 FFT 版单演/Riesz 局部相位（保持同一接口 `phase_pyramid.py`） |
| 4 | 无 ffmpeg | `which ffmpeg` 为空 | 统一 `cv2.VideoWriter(mp4v)`；输出尺寸取偶数避免编码问题 |
| 5 | 长视频/多带复数系数内存 | T×H×W×bands×complex128 增长快 | 时间分块处理，块边界与滤波器上下文显式处理（避免接缝/闪烁）；首版仅离线 |
| 6 | 任务串味（FD4MM 帧对 vs 真实视频序列） | 已确认是**两个不同问题** | 阶段 1 只做真实视频；FD4MM/pyramid/warp 代码冻结保留，不混用 |
| 7 | 旧入口 `dila_video_vmm.py` 已不可运行 | 缺 ffmpeg、缺 ckpt、API 已不存在 | 保留归档，不在其上继续开发 |

---

## 9. 结论与下一步

* 阶段 0 结论：**仓库中不存在任何相位式/可转向滤波实现**；旧潜变量放大器栈（方案 B）与帧对 warp 栈（路线 B）均**冻结保留**；既有 18 个放大器用例当前全部通过；复数可转向金字塔经实测可用且重建自洽（3.82e-06），相移→重建链路有效。
* 下一步（阶段 1）：按 §7 新增 `vmm/` 独立模块与 `scripts/`、`tests/`，先跑通
  **① α=0 恒等** → **② 合成亚像素位移的运动增益（与 1+α 对照）** → **③ 纯噪声/静止不误放大** → **④ 真实视频 α=0,1,2,5,10 对照**，
  全部通过后才进入阶段 2（可靠性门控 + 放大中去噪）。