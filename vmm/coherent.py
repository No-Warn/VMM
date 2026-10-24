# -*- coding: utf-8 -*-
"""Phase G：**多帧相干积累**（相位域，离线）。

研究问题
--------
在 Eulerian 相位放大主干上，多帧相干积累能否在**噪声存在**时提高
"真实微小运动 / 非运动扰动"的可分性，而不是仅仅做时间平滑。

本模块只提供两种**互不相同**的东西，并明确区分它们：

1. **平滑族**（``mean`` / ``median``）：直接对带通相位 ``Φ_bp`` 做长度 ``W`` 的滑窗统计。
   它是**线性/非线性滤波**，会改变运动传递函数（可能削弱真实运动）——
   **按任务书要求，这一族只作为对照，不当作"相干积累的成功实现"**。
2. **相干证据族**（``coherent``）：**不改动放大路径**（``β = α·w·Φ_bp`` 里的 ``Φ_bp`` 原样保留），
   只改进**可靠性权重** ``w`` 的估计：

       z(t)  = Φ_bp(t) + j·H[Φ_bp](t)          （沿时间的解析信号，H = Hilbert）
       p(t)  = |z(t)|²
       A(t)  = sqrt( (1/W)·Σ_{τ∈窗} p(τ) )      ← **相干检测 + 包络功率积累**
       w(t)  = A² / (A² + σ_z²)                 （σ_z 在**独立**的静止+噪声片上标定）

   为什么它不等于"时间平滑"：

   * 平滑作用在**信号**上，等效于改变传递函数（``|H(f)|`` 在目标带内也会下降）；
   * ``A`` 是**包络**（对相位不敏感），对 ``p=|z|²`` 取窗均值是**非线性**操作：
     稳定振荡的包络被保留，而估计量的**方差**按 ``1/√W`` 下降；
   * 因此运动保真度**由构造保证**（放大路径一行未改），改善只体现在
     "运动证据 vs 噪声证据"的可分性上（Phase G 的 G1/G2/G3）。

边界与延迟（任务书 §4"时间边界处理"）
--------------------------------------
* **使用未来帧**（窗居中）⇒ 离线算法；``W`` 越大延迟越大（延迟 = (W−1)/2 帧）。
* 首尾用 ``mode="reflect"`` 延拓，**不**丢弃首尾帧，输出帧数与输入一致。
* 分块流式：本模块**不支持**跨块的相干积累（窗需要跨块上下文）；
  块级调用时必须由调用方保证 ``W`` 内的上下文完整（见 ``required_margin``）。
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12

__all__ = ["analytic_signal", "window_stat", "coherent_amplitude",
           "coherent_weight", "coherent_noise_level", "required_margin"]


def analytic_signal(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """沿 ``axis`` 的解析信号 ``z = x + j·H[x]``（FFT 实现，零相位）。

    实现要点：把正频率乘 2、负频率置 0，直流与奈奎斯特保持不变（偶长时奈奎斯特只乘 1）。
    对实数输入返回复数数组；``z`` 的实部与 ``x`` 逐位相同。
    """
    x = np.asarray(x)
    if not np.isrealobj(x):
        raise ValueError("analytic_signal 需要实数输入")
    T = x.shape[axis]
    X = np.fft.fft(x, axis=axis)
    h = np.zeros(T, dtype=np.float64)
    if T % 2 == 0:
        h[0] = 1.0
        h[T // 2] = 1.0
        h[1:T // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(T + 1) // 2] = 2.0
    shape = [1] * x.ndim
    shape[axis] = T
    return np.fft.ifft(X * h.reshape(shape), axis=axis)


def window_stat(x: np.ndarray, w: int, kind: str = "mean",
                mode: str = "reflect") -> np.ndarray:
    """沿时间（axis=0）的**居中**滑窗统计；``w`` 取奇数。

    ``kind``：``mean``（均值，线性）| ``median``（中位数，非线性、抗野值）。
    ``mode="reflect"``：首尾对称延拓，输出长度不变（帧数、时间对齐都不变）。
    """
    from scipy import ndimage as ndi
    w = int(w)
    if w <= 1:
        return np.asarray(x, dtype=np.float64)
    w = w | 1                                    # 强制奇数，保证窗居中
    if kind == "mean":
        return ndi.uniform_filter1d(np.asarray(x, np.float64), size=w, axis=0, mode=mode)
    if kind == "median":
        return ndi.median_filter(np.asarray(x, np.float64), size=(w, 1, 1), mode=mode)
    raise ValueError(f"window_stat kind 只能是 mean/median，收到 {kind!r}")


def coherent_amplitude(Phi_bp: np.ndarray, w: int) -> np.ndarray:
    """相干检测后的**包络**：``A = sqrt( window_mean(|analytic(Φ_bp)|²) )``。

    返回形状与输入相同（逐帧逐像素），单位与 ``Φ_bp`` 相同（弧度）。
    """
    z = analytic_signal(np.asarray(Phi_bp, np.float64))
    p = np.abs(z) ** 2
    return np.sqrt(np.maximum(window_stat(p, w, "mean"), 0.0))


def coherent_noise_level(Phi_bp_static: np.ndarray, w: int,
                         quantile: float = 0.5) -> float:
    """在**独立**的静止+噪声片段上标定 ``σ_z``（时间中位数 + 空间稳健分位）。

    只接受"确实没有运动"的 ``Φ_bp``；调用方负责保证（阶段 2 的 ``noise_seed`` 机制）。
    """
    A = coherent_amplitude(Phi_bp_static, w)
    return float(np.quantile(A, quantile))


def weight_from_amplitude(A: np.ndarray, sigma_z: float, form: str = "excess",
                          eps: float = EPS) -> np.ndarray:
    """由相干幅度 ``A`` 得到权重（与 :func:`coherent_weight` 同一公式，避免重复算 FFT）。"""
    s2 = max(float(sigma_z), 0.0) ** 2
    A2 = np.asarray(A, np.float64) ** 2
    if form == "excess":
        return np.clip((A2 - s2) / (A2 + eps), 0.0, 1.0).astype(np.float32)
    if form == "snr":
        return (A2 / (A2 + s2 + eps)).astype(np.float32)
    raise ValueError(f"form 只能是 excess/snr，收到 {form!r}")


def coherent_weight(Phi_bp: np.ndarray, w: int, sigma_z: float,
                    form: str = "excess", eps: float = EPS) -> np.ndarray:
    """相干证据权重 ``w ∈ [0,1]``（``σ_z`` 必须**同管线**标定）。

    两种形式，差别是"纯噪声像素应该给多少"：

    * ``form="excess"``（默认）——**超额功率**：``w = relu(A² − σ_z²) / (A² + ε)``。
      纯噪声（``A≈σ_z``）⇒ ``w≈0``；稳定运动（``A≫σ_z``）⇒ ``w→1``。
      这是"相干功率超出噪声地板的比例"，可解释、且让 G1 的比值有明确分母。
      代价：**弱到 ``A≤σ_z`` 的真实运动会被判 0**（保守；与阶段 2 的 `amp_floor` 同类风险）。
    * ``form="snr"``——``w = A²/(A²+σ_z²)``。纯噪声 ⇒ ``w→0.5``（不是 0），
      因此 G1 的比值天然被压到 ≤2；保留它用于对照，说明"公式选择本身影响 Gate"。

    ⚠️ ``w`` 在运动像素会**饱和到 1**，因此**不能**用 ``w`` 算 G3（混淆项/运动）——
    那会平凡地 <1。Gate 必须用**未截断的** ``A``（见 ``coherent_amplitude``）。
    """
    return weight_from_amplitude(coherent_amplitude(Phi_bp, w), sigma_z, form, eps)


def required_margin(w: int) -> int:
    """居中窗需要的上下文帧数（每侧）。分块调用时必须满足 ``margin ≥ (w−1)//2``。"""
    w = int(w) | 1
    return (w - 1) // 2
