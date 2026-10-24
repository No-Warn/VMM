# Phase G 设计：面向 VMM 微小运动放大的多帧相干积累

> 代码：`vmm/coherent.py`（新增）、`vmm/phase_magnifier.py`（接入，默认关闭）、
> `vmm/reliability.py`（配置项）、`scripts/phaseG_coherent_accumulation.py`（实验入口）
> 结果：`docs/phaseG_results.md`；决策：`docs/phaseG_decision.md`
> 数据：`outputs/stage2_coherent/`

---

## 0. 研究问题与先验

**问题**：在现有 Eulerian 相位放大主干上，多帧相干积累能否在**有噪声**时提高
"真实微小运动 / 非运动扰动"的可分性？

**先验（必须在结果里检验，不能假设成立）**：
1. 主干的时间带通（0.3–1.0 Hz，`filtfilt`）本身已经是**全片长度的线性滤波**；
   任何"再对相位做时间平均"的操作都是**又一次低通**，其收益应与"同等窗长的平滑"可比。
2. 唯一的例外是**非线性检测**：对相干检测器的输出（包络/功率）做积累，
   可以降低**估计量的方差**而不改变运动的传递函数。
3. 因此本阶段把方法分成两族，并**用同一窗长的平滑作为直接对照**（任务书 §6 控制实验 1）。

---

## 1. 信号定义（任务书 §4 要求逐步写清）

| 符号 | 定义 | 形状 |
|---|---|---|
| `dphi_s,o(x,t)` | 相邻帧局部相位增量（wrap 到 (−π,π]） | `[T, h, w]` |
| `Φ_s,o(x,t)` | 累积相位 `Φ_t = Σ_{τ≤t} dphi_τ` | `[T, h, w]` |
| `Φ_bp` | 零相位带通（Butterworth，`filtfilt`，离线） | `[T, h, w]` |
| `W` | 积累窗长（帧，强制奇数） | 标量 |
| `z(t)` | 解析信号 `Φ_bp + j·H[Φ_bp]`（沿时间 FFT-Hilbert） | `[T, h, w]` complex |
| `A(t)` | 相干包络 `sqrt( (1/W) Σ_τ |z(τ)|² )` | `[T, h, w]` |
| `σ_z` | 噪声地板：在**独立**静止+噪声片段上同管线算 `A` 的稳健分位 | 每带一个标量 |
| `w(x,t)` | 可靠性权重 ∈ [0,1]；`β = α·w·Φ_bp` | `[T,h,w]`（相干族）或 `[h,w]`（B2） |

**相位参考来自哪里？** 相干检测**不需要显式的外部相位参考**：解析信号在每一点
自带瞬时相位 `ψ(t)=arg z(t)`，`|z|` 对该相位不敏感（这正是它比"直接把不同时刻的
`dphi` 相加"稳健的原因——后者会因相位反相而互相抵消）。若需要显式的频率参考
（例如已知目标 f₀），本实现**不**使用它——因为真实视频的目标频率未知，
使用它会让方法依赖真值。

**输入与输出的时间对齐**：所有操作都是**居中窗**（`mode="reflect"`），
输出帧数与输入相同、逐帧对齐；延迟 = `(W−1)/2` 帧（仅影响在线场景，本阶段离线）。

---

## 2. 三种候选（任务书 §4 的 A/B/C）

### 候选 A：固定窗口稳健统计（`mean` / `median`）—— **平滑对照族**

```
Φ_bp'(t) = window_stat( Φ_bp ; W, mean|median )
β(t)     = α · Φ_bp'(t)                     # 不引入额外权重
```

* 作用在**信号**上，改变传递函数；`median` 是非线性、抗野值。
* **按任务书要求，本族只作对照，不当作"相干积累的成功实现"。**

### 候选 B：相干证据积累（`coherent`）—— **本阶段主张的方法**

```
z(t) = Φ_bp(t) + j·H[Φ_bp](t)
A(t) = sqrt( window_mean( |z|² ; W ) )
w(t) = relu(A² − σ_z²) / (A² + ε)          # 超额功率（纯噪声→0，稳定运动→1）
β(t) = α · w(t) · Φ_bp(t)                   # Φ_bp 原样保留：放大路径一行未改
```

与候选 A 的本质区别：
* A 改的是 `Φ_bp`（运动本身），B 改的是 `w`（对运动的**可信度**）；
* `|z|²` 的窗均值是**非线性**的包络功率积累：稳定振荡的包络被保留，
  噪声的**波动**按 `1/√W` 下降；
* ⇒ B 的**运动保真度由构造保证**（增益/方向/波形只由原路径决定）。

### 候选 C：可靠性加权与回退（`coherent_rel` / `adaptive`）

* `coherent_rel` = `w_coh · w_A`（`w_A` 是阶段 2 的幅值证据，需 `σ_A` 标定）；
* `adaptive` = 帧内 `median(w_coh) < τ` 时**回退到 B2** 的归一化谱集中度权重
  （`τ=0.35`，预注册，不在测试集上调）。

---

## 3. 时间边界与代价（任务书 §4 "时间边界处理"）

| 问题 | 本实现 |
|---|---|
| 是否用未来帧 | **是**（居中窗）⇒ 离线算法 |
| 窗口是否居中 | 是（强制奇数 `W`） |
| 首尾帧 | `mode="reflect"` 对称延拓，**不丢帧**、输出帧数不变 |
| 延迟 | `(W−1)/2` 帧（W=5/9/15 → 2/4/7 帧） |
| 长视频分块 | 窗需要跨块上下文：每侧 `margin ≥ (W−1)//2`（`vmm.coherent.required_margin`）；本模块**不做**跨块积累，调用方必须保证上下文 |
| 内存 | 每带多一个复数 `[T,h,w]`（W 无关）；实测见结果文档 |
| 计算 | 每带一次 `rfft/ifft` + 一次 `uniform_filter1d`；相对 B2 的增加见结果文档 |

---

## 4. 配置与开关（默认关闭 ⇒ 与 B2/阶段 2 逐位一致）

```python
ReliabilityConfig(
    mode="B2",                 # 或 "M0"（只开相干证据时）
    coherent_mode="none",      # none | mean | median | coherent | coherent_rel | adaptive
    coherent_window=5,
    coherent_sigma={...},      # 逐带 σ_z（vmm.coherent.coherent_noise_level 标定）
    coherent_form="excess",    # excess | snr
    coherent_tau=0.35,
)
```

* `coherent_mode="none"`（默认）时，`vmm/phase_magnifier.py` 的新增分支**全部不执行**；
  已用测试 `test_m0_matches_stage1_bitwise` 与 B2 的逐位复现（本阶段 gain diff = **0.0**）守住。
* 所有新参数只在 `scripts/phaseG_coherent_accumulation.py` 的配置里出现，
  **没有修改 B2 的默认参数**。

---

## 5. 预注册 Gate（任务书 §7；跑实验前写死）

G1/G2/G3 的**计算定义沿用 Phase D**（`scripts/dino_dila_probe.py` / `probe.json`）：
"运动证据 / 噪声证据"的比值、活动区与背景的对比度、外观干扰与真实运动的比值。
**唯一的口径改动（必须声明）**：被评估的"特征层"从 DINOv2 层换成
**本阶段的相位证据图**（B2 用 `w`；相干族用**未截断的** `A`）。
理由：Phase G 的产出就是相位证据；DINOv2 与本文无关。网格 = 最细金字塔带
（与输入同分辨率 192×192，1 patch = 1 px），随机 IoU 基线 = `k/(2N−k)` ≈ 0.06。

| Gate | 定义（分子/分母） | 阈值 |
|---|---|---|
| **G1** | `evidence_active(local, 0.05 px, σ=0.02) / evidence_static(static_noise, σ=0.02)`，时间平均后取区域均值 | **≥ 2.0** |
| **G2** | 同一案例的 `evidence_active / evidence_background`（活动区外全部像素） | **≥ 1.5** |
| **G3** | `max( evidence_brightness / evidence_motion(干净 0.05 px),  evidence_texture / evidence_motion )` | **< 1.0** |

**为什么 G3 用 `A` 而不是 `w`**：`w∈[0,1]` 在运动像素饱和到 1，`G3 = w_conf/w_motion`
会**平凡地 ≤1**；用未截断的相干幅度 `A` 才能检验"外观变化是否会被误判成运动"。

**同时必须满足（任务书 §7 "Gate 判定原则"）**：
1. G1、G2 达标且 G3 达标；
2. 合成真值运动没有不可接受的幅值/方向/波形损伤（`|gain/(1+α)−1|` 与波形相关，沿用阶段 1 判据）；
3. 相对**同等窗长的平滑**（G1/G2 方法）存在额外收益；
4. 至少 3 个随机种子可复现；
5. 不能靠降低噪声能量达标（必须同时看运动保留）。

---

## 6. 数据划分（任务书 §5 "数据划分"）

| 用途 | 案例 / 种子 |
|---|---|
| **参数选择（验证集）** | `local` 0.05/0.25 px、σ=0.01、seed=0/1 —— 仅用于选 `W` 与 `τ` |
| **最终测试** | σ=0.02、seed=7/8/9；`local`/`local_lowc`/`shm_h`/`shm_v`/`static_noise` |
| **干扰项** | `brightness`（3% 亮度调制、无几何运动）、`texture_change`（局部纹理渐变） |
| **不参与调参** | 上述最终测试集在选定 W 之前不运行 |

> 本轮受算力限制，验证集只用于在 `W∈{5,9}` 中选一个，并在结果文档中**明确写出**
> "W 的选择依据来自验证集"，最终测试只跑一次。

---

## 7. 失败判据（预先写死）

出现下列任一情况即判 **NO-GO**（不得事后放宽）：

1. G1 < 2.0 或 G2 < 1.5 或 G3 ≥ 1.0；
2. 相干证据的收益与**同窗长平滑**相当（差距 < 10%），即"只是低通"；
3. 运动保真度显著下降（`|gain/(1+α)−1| > 0.2`，或波形相关 < 0.9，或方向反）；
4. 结果在 3 个种子上不可复现（G1 的符号/量级翻转）；
5. 静态噪声的伪运动或边界泄漏显著恶化。

---

## 8. 复现命令

```bash
PY=/home/user/.conda/envs/dila/bin/python
cd .

$PY scripts/phaseG_coherent_accumulation.py --stage baseline --methods G0 \
    --out-json outputs/stage2_coherent/baseline_b2.json
$PY scripts/phaseG_coherent_accumulation.py --stage ablate \
    --methods G0 G1 G2 G3 G4 G5 --windows 5 9 \
    --out-json outputs/stage2_coherent/ablation.json
$PY scripts/phaseG_coherent_accumulation.py --stage gates \
    --ablation-json outputs/stage2_coherent/ablation.json \
    --out-json outputs/stage2_coherent/gates.json
$PY -m pytest tests/test_coherent.py -q
```
