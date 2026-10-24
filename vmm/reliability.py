# -*- coding: utf-8 -*-
"""阶段 2：微运动**可靠性门控**的确定性实现（无学习参数）。

数学定义（与任务书 §2 对齐）
----------------------------
阶段 1 的相位放大为

    Δφ_out = Δφ + α · Δφ_bp

阶段 2 改为

    Δφ_out = Δφ + α_eff · w(x, t, s, o) · Δφ_bp ,      w ∈ [0, 1]
    α_eff  = min(α, β_{s,o})                            （§3 逐尺度逐方向增益上限）

四种策略（可关闭、可组合）：

* **M0** 无门控 —— 与阶段 1 **逐位一致**（默认，保证不改变既有行为）。
* **M1** 幅值可靠性 ``w_A = A² / (A² + (g·σ_A(s,o))² + ε)``。
* **M2** M1 × 相位增量稳定性 ``w_stab = Φ_bp² / (Φ_bp² + (g·σ_φ(s,o))² + ε)``。
* **M3** M2 × 局部时空一致性 ``w_coh``，后者由**相邻系数的相位增量方向一致性**给出。

尺度统一（任务书 §2.1 的硬性要求）
----------------------------------
``A`` 是**复数可转向金字塔某带、某像素的幅值**（与输入图像同强度单位），
``σ_A`` 必须**在同一个表示上**测得 —— 做法是在**静止视频/静止 ROI** 上逐带统计
``A(x,t)`` 的**时间标准差**再取空间稳健中位数。这样 ``A²`` 与 ``σ_A²`` 同量纲，
比值 ``w_A`` 无量纲。**禁止**把像素域噪声方差直接加到归一化后的带幅值平方上
（阶段 1 早有此坑：``metrics`` 曾用"三通道平均"而当输出用 Rec.709，导致 α=0 也报 0.95）。

方向一致性只作为**证据之一**（任务书 §2.2 警告）：局部真实运动可以与背景不同向，
因此 ``w_coh`` 只要求**相位增量在很小的邻域内自相一致**，绝不去对齐"背景运动"。

`amp_floor` 的正确含义（任务书 §4）
----------------------------------
阶段 1 的 ``amp_floor`` **只是声明、从未参与计算**（审计确认）。阶段 2 把它实现为
**运动表示层的幅值可信度**，采用**软**形式，绝不清零低对比度真实运动：

    r          = A / median_band(A)          （无量纲：相对本带中位幅值）
    w_ampfloor = r² / (r² + amp_floor²)      （amp_floor=0 时关闭）

单位与作用位置：``amp_floor`` 是**相对本带中位幅值的比例**（无量纲），
作用在**运动表示层**（乘在 β 上），**不是**像素重建后的幅度裁剪；
后者若需要，请另用 ``recon_amp_clip`` 之类的独立参数名（阶段 2 未实现）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple

import numpy as np

from vmm.phase_pyramid import ComplexSteerablePyramid

EPS = 1e-12


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass
class ReliabilityConfig:
    """门控与增益上限配置（全部默认关闭 ⇒ 与阶段 1 逐位一致）。"""

    mode: str = "M0"                    # M0|M1|M2|M3|M4|B1(原始谱集中度)|B2(scalar 归一化)|B3(逐像素归一化)
    #: [B2] 谱集中度证据在哪个域上度量：``"bp"``=带通后 Φ_bp（与 B1 同数据），
    #: ``"raw"``=原始累积相位 Φ（**保留**带外信息，判别余量更大但绝对水平更低）。
    band_domain: str = "bp"
    #: 逐带噪声尺度（静止校准得到）。键为 "(level,orient)" 字符串或 (level,orient) 元组
    sigma_A: Optional[Dict] = None
    sigma_phi: Optional[Dict] = None
    sigma_scale: float = 1.0            # g：σ 的缩放（消融用；1.0 = 用校准值）
    amp_floor: float = 0.0              # §4：相对本带中位幅值的软阈值（0=关闭）
    spatial_smooth: float = 0.0         # 对 w 的空间高斯平滑 σ（单位=该带网格的 cell）
    temporal_smooth: int = 0            # 对 w 的时间对称平滑核长（帧；0=关闭）
    temporal_robust: int = 0            # §5 D2：对 Φ_bp 的稳健时间抑噪核长（帧；0=关闭）
    coh_radius: int = 1                 # M3：局部一致性邻域半径（该带网格的 cell）
    per_band_caps: Optional[Dict] = None  # §3：β_{s,o}（键同上）；None=不限制
    #: Phase G：多帧相干积累（见 `vmm/coherent.py`）。``none`` 时与阶段 2 完全一致。
    #: ``mean``/``median`` = **平滑对照**（作用在 Φ_bp 上，会改传递函数）；
    #: ``coherent`` = 只改可靠性证据（放大路径不变）；``coherent_rel`` = 再乘幅值证据；
    #: ``adaptive`` = 相干证据不可靠时回退到 B2 的谱集中度权重。
    coherent_mode: str = "none"
    coherent_window: int = 5             #: W（帧，强制奇数；延迟 (W−1)/2）
    coherent_sigma: Optional[Dict] = None  #: 逐带 σ_z（独立静止+噪声片标定；键同上）
    coherent_tau: float = 0.35           #: adaptive 回退阈值（帧内 w 中位数）
    coherent_form: str = "excess"        #: excess（纯噪声→0）| snr（纯噪声→0.5）
    eps: float = EPS

    def resolved(self) -> "ReliabilityConfig":
        """把键统一成 ``(level, orient)`` 元组，缺失值补 None。"""
        def fix(d):
            if not d:
                return None
            out = {}
            for k, v in d.items():
                if isinstance(k, str):
                    a, b = k.split("-")
                    out[(int(a), int(b))] = float(v)
                else:
                    out[(int(k[0]), int(k[1]))] = float(v)
            return out
        self.sigma_A = fix(self.sigma_A)
        self.sigma_phi = fix(self.sigma_phi)
        self.per_band_caps = fix(self.per_band_caps)
        self.coherent_sigma = fix(self.coherent_sigma)
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("sigma_A", "sigma_phi", "per_band_caps", "coherent_sigma"):
            if d.get(k):
                d[k] = {f"{a}-{b}": v for (a, b), v in d[k].items()}
        return d

    @property
    def enabled(self) -> bool:
        return ((self.mode != "M0") or (self.amp_floor > 0) or bool(self.per_band_caps)
                or (self.coherent_mode != "none"))


# --------------------------------------------------------------------------- #
# 噪声标定（静止背景，逐尺度逐方向）
# --------------------------------------------------------------------------- #
def _mask_for(mask: Optional[np.ndarray], shape) -> Optional[np.ndarray]:
    """把图像分辨率的掩码缩放到**某个带的网格**上（最近邻）。

    ⚠️ 金字塔逐级降采样：192 长边的图像最细带是 96×96、最粗 12×12……
    直接把 [H,W] 掩码索引到带幅值上会报 "boolean index did not match"（实测踩到）。
    """
    if mask is None:
        return None
    shape = tuple(int(s) for s in shape)
    if tuple(mask.shape) == shape:
        return mask
    import cv2
    m = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]),
                   interpolation=cv2.INTER_NEAREST).astype(bool)
    return m


def calibrate_noise(video_static: np.ndarray, levels: int = 4, order: int = 3,
                    mask: Optional[np.ndarray] = None, twidth: int = 1,
                    quantile: float = 0.5) -> Dict:
    """在**静止**视频（或静止 ROI）上逐带标定噪声尺度。

    Args:
        video_static: ``[T,H,W]`` 或 ``[T,H,W,3]``；假设其中**没有真实运动**。
        mask: 可选布尔掩码 ``[H,W]``（静止背景 ROI）；``None`` 用全图。

    Returns:
        ``{"sigma_A": {(lvl,ori): σ}, "sigma_phi": {(lvl,ori): σ_φ},
           "band_stats": {(lvl,ori): {...}}}``
        ``σ_A`` 是 ``A(x,t)`` 的时间标准差（**与 A 同单位**）在像素上的稳健分位数；
        ``σ_φ`` 是 ``Φ_bp(x,t)`` 的时间标准差（弧度），供 M2 使用；
        另返回逐带的 ``A`` 统计（均值/中位/分位）与 ``|Δφ|`` 分布，供 §3 的尺度分析。
    """
    from vmm.temporal_filter import bandpass_filter, bandpass_time
    gray = video_static if video_static.ndim == 3 else \
        (video_static * np.array([0.2126, 0.7152, 0.0722])).sum(axis=3)
    T = gray.shape[0]
    pyr = ComplexSteerablePyramid(height=levels, order=order, twidth=twidth)
    pyr.validate(gray[0])
    amp: Dict = {}
    phi: Dict = {}
    for t in range(T):
        coeffs = pyr.analyze(gray[t])
        for k in pyr.complex_keys:
            c = np.asarray(coeffs[k])
            if k not in amp:
                amp[k] = np.empty((T,) + c.shape, np.float32)
                phi[k] = np.empty((T,) + c.shape, np.float32)
            amp[k][t] = np.abs(c)
            phi[k][t] = np.angle(c)
    sigma_A, sigma_phi, stats = {}, {}, {}
    for k in pyr.complex_keys:
        a = amp[k].astype(np.float64)
        ph = phi[k].astype(np.float64)
        # 静止视频上 A 的时间标准差 = 该带的噪声尺度（无真实运动）
        sa = a.std(axis=0)
        sel = _mask_for(mask, a.shape[1:])
        if sel is not None and not sel.any():
            sel = None                       # 掩码在该带网格上为空 → 退回全带
        if sel is not None:
            sa = sa[sel]
        sigma_A[k] = float(np.quantile(sa, quantile))
        # 相位在静止视频上仍会因噪声抖动 → 用带通后的相位作为 M2 的噪声尺度
        d = np.empty_like(ph)
        d[0] = 0.0
        d[1:] = (ph[1:] - ph[:-1] + np.pi) % (2 * np.pi) - np.pi
        Phi = np.cumsum(d, axis=0)
        try:
            Phi_bp = bandpass_time(Phi, bandpass_filter(_fps_of(video_static), 0.5, 2.0, 2))
        except Exception:
            Phi_bp = Phi
        sp = Phi_bp.std(axis=0)
        if sel is not None:
            sp = sp[sel]
        sigma_phi[k] = float(np.quantile(sp, quantile))
        am = a[:, sel] if sel is not None else a
        absd = np.abs(d)
        stats[k] = {
            "A_mean": float(a.mean()), "A_median": float(np.median(a)),
            "A_p10": float(np.quantile(a, 0.10)), "A_p90": float(np.quantile(a, 0.90)),
            "sigma_A": sigma_A[k], "sigma_phi": sigma_phi[k],
            "dphi_abs_mean": float(absd.mean()), "dphi_abs_p90": float(np.quantile(absd, 0.9)),
            "coeff_shape": tuple(a.shape[1:]),
            "noise_over_signal_A": float(sigma_A[k] / max(np.median(a), EPS)),
        }
    return {"sigma_A": sigma_A, "sigma_phi": sigma_phi, "band_stats": stats,
            "config": {"levels": levels, "order": order, "T": int(T),
                       "quantile": quantile, "masked": mask is not None,
                       "note": "σ 在**同一金字塔带幅值单位**上标定（时间标准差+稳健分位）"}}


def _fps_of(video) -> float:
    return float(getattr(video, "fps", 30.0))


# --------------------------------------------------------------------------- #
# 权重函数（全部在 [0,1]）
# --------------------------------------------------------------------------- #
def amplitude_weight(A: np.ndarray, sigma_A: float, g: float = 1.0,
                     eps: float = EPS) -> np.ndarray:
    """``w_A = A²/(A²+(g·σ_A)²+ε)`` —— A 与 σ_A 同单位（带幅值）。"""
    s = max(float(g) * float(sigma_A), 0.0)
    return (A * A) / (A * A + s * s + eps)


def amp_floor_weight(A: np.ndarray, A_median: float, amp_floor: float,
                     eps: float = EPS) -> np.ndarray:
    """``w = r²/(r²+amp_floor²)``，``r = A/median(A)``（无量纲，软阈值，不清零）。"""
    if amp_floor <= 0:
        return np.ones_like(A)
    r = A / max(float(A_median), eps)
    f = float(amp_floor)
    return (r * r) / (r * r + f * f + eps)


def stability_weight(Phi_bp: np.ndarray, sigma_phi: float, g: float = 1.0,
                     eps: float = EPS) -> np.ndarray:
    """``w_stab = Φ_bp²/(Φ_bp²+(g·σ_φ)²+ε)`` —— 目标带内相位变化相对噪声相位底。"""
    s = max(float(g) * float(sigma_phi), 0.0)
    x = Phi_bp * Phi_bp
    return x / (x + s * s + eps)


def concentration_map(Phi: np.ndarray, fps: float, f_lo: float, f_hi: float,
                      taper: bool = True, eps: float = EPS):
    """逐像素目标带能量占比（``w_band`` 的**唯一**定义处）。

    返回 ``(conc, p, band, freqs)``：``conc = E_band/(E_band+E_outside+ε)``；
    ``p`` 是全谱功率，供调用方做"主导带内频点"等进一步处理。
    """
    T = Phi.shape[0]
    x = Phi - Phi.mean(axis=0, keepdims=True)
    if taper:
        x = x * np.hanning(T)[:, None, None]
    F = np.fft.rfft(x, axis=0)
    freqs = np.fft.rfftfreq(T, d=1.0 / float(fps))
    band = (freqs >= float(f_lo)) & (freqs <= float(f_hi))
    p = np.abs(F) ** 2
    num = p[band].sum(axis=0)
    den = p[~band].sum(axis=0)
    return (num / (num + den + eps)).astype(np.float32), p, band, freqs


def band_occupancy_weight(Phi_bp: np.ndarray, fps: float, f_lo: float, f_hi: float,
                          taper: bool = True, eps: float = EPS) -> np.ndarray:
    """**[B1] 目标带谱集中度**（长时窗相位相干证据，离线可用；保持逐位不变）。

    单个像素的相位变化若真的来自目标频带内的运动，则它随时间的变化应**集中在带内**；
    噪声引起的相位抖动则分布在带内外。于是定义（逐像素）：

        w_band = E_band / (E_band + E_outside + ε)
        E_band  = Σ_{f∈[f_lo,f_hi]} |DFT(Φ_bp − mean)|²        （带内能量）
        E_outside = Σ_{f∉[f_lo,f_hi]} |DFT(Φ_bp − mean)|²      （带外+直流能量）

    与 M2 的 ``w_stab`` 的区别：``w_stab`` 是**单帧幅度的信噪比**，
    在微运动接近噪声底时会把真实运动一起判为不可靠（阶段 2 第一轮实测的冲突）；
    本量用**整段序列的频谱分布**做判据，对"周期性真实运动 vs 非周期噪声"更敏感。

    处理细节：先去时间均值（去掉直流漂移）、可选 Hann 窗（压边界泄漏），
    ε 保证零信号时 ``w_band→0``（无证据 ⇒ 不给额外增益，保守）。

    ⚠️ **已知缺陷（阶段 2 未闭环 #1 的根因）**：本量是**无量纲比值**，其绝对
    水平与 ``T``/频带/窗/**是否已带通**强相关。实测同一纯带内正弦（0.5 Hz、
    fps=30、带 0.3–1.0 Hz）：``T=64`` 时 conc≈0.98，``T=120`` 时 conc≈0.71，
    ``T=240`` 时 conc≈0.99。因为它直接乘在 α 上，``T=120`` 就把**所有**像素的
    增益乘了 0.71（−23%），与"低对比度"无关。需要归一化请用
    :func:`band_occupancy_normalized`（[B2]）。
    """
    T = Phi_bp.shape[0]
    if T < 4:
        return np.ones(Phi_bp.shape[1:], np.float32)
    conc, _, _, _ = concentration_map(Phi_bp, fps, f_lo, f_hi, taper, eps)
    return conc


# --------------------------------------------------------------------------- #
# [B2] 归一化目标带谱集中度：**同管线**标定 ref/null，消除 T/带宽带来的整体偏移
# --------------------------------------------------------------------------- #
_OCC_REF_CACHE: Dict = {}


def occupancy_reference(T: int, fps: float, f_lo: float, f_hi: float,
                        domain: str = "bp", taper: bool = True,
                        n_noise: int = 3, seed: int = 0) -> Dict:
    """**同管线**标定 ``w_band`` 的两端：纯带内正弦（ref）与累积相位噪声（null）。

    为什么必须"同管线"：:func:`concentration_map` 的绝对值取决于
    ``T``/频带/窗/是否带通。实测（T=120、带 0.3–1.0 Hz）同一纯正弦在
    **带通前** conc≈0.83、**带通后** conc≈0.71；而带通后的**噪声** conc≈0.60。
    ⇒ 用固定阈值时，"判据的零点"本身随 `T` 漂移，B1 在 T=120 对**任何**输入
    都乘 0.71（全局降增益），在 T=64 乘 0.98。本函数把 ref/null 用**完全相同
    的预处理**测出来，于是"纯带内正弦 ⇒ w=1"在任何 T 成立，`T` 依赖被消除。

    Args:
        domain: ``"bp"``（在带通后的证据上度量，ref/null 也带通）或
            ``"raw"``（在原始累积相位上度量，**不**带通 ⇒ 保留带外信息）。
    Returns:
        ``{"refs": [每个带内频点的纯正弦 conc], "null": 噪声 conc,
           "freqs": 带内频点, "domain": domain}``
    """
    key = (int(T), round(float(fps), 6), round(float(f_lo), 6),
           round(float(f_hi), 6), str(domain), bool(taper))
    if key in _OCC_REF_CACHE:
        return _OCC_REF_CACHE[key]
    freqs = np.fft.rfftfreq(int(T), d=1.0 / float(fps))
    band = (freqs >= float(f_lo)) & (freqs <= float(f_hi))
    idx = np.where(band)[0]
    if domain == "bp":
        from vmm.temporal_filter import bandpass_filter, bandpass_time
        filt = bandpass_filter(fps, f_lo, f_hi, 2)
        prep = lambda x: bandpass_time(x, filt)          # noqa: E731
    elif domain == "raw":
        prep = lambda x: x                              # noqa: E731
    else:
        raise ValueError(f"domain 只能是 bp 或 raw，收到 {domain}")
    t = np.arange(int(T), dtype=np.float64) / float(fps)
    refs = np.empty(len(idx), np.float64)
    for j, i in enumerate(idx):
        tone = np.sin(2.0 * np.pi * freqs[i] * t)[:, None, None]
        refs[j] = float(concentration_map(prep(tone), fps, f_lo, f_hi, taper)[0].mean())
    rng = np.random.default_rng(int(seed))
    nulls = []
    for _ in range(max(1, int(n_noise))):
        rw = np.cumsum(rng.standard_normal((int(T), 1, 1)), axis=0)
        nulls.append(float(concentration_map(prep(rw), fps, f_lo, f_hi, taper)[0].mean()))
    out = {"refs": refs, "null": float(np.mean(nulls)),
           "freqs": freqs[idx], "domain": domain, "T": int(T)}
    _OCC_REF_CACHE[key] = out
    return out


def band_occupancy_normalized(evidence: np.ndarray, fps: float, f_lo: float, f_hi: float,
                              domain: str = "bp", mode: str = "scalar",
                              taper: bool = True, floor: float = 0.02,
                              upper_clip: Optional[float] = None,
                              eps: float = EPS) -> np.ndarray:
    """**[B2/B3] 归一化目标带谱集中度**：把"任意尺度"的 conc 变成"相对纯带内正弦"的可靠性。

    为什么必须归一化：:func:`band_occupancy_weight`（B1）直接拿 conc 乘 α，而 conc
    的绝对水平取决于 ``T``/带宽/窗/是否带通（实测同一 0.5 Hz 纯正弦：T=64→0.998、
    T=120→0.719、T=240→0.994）。于是**同一条视频只因截断长度不同，用户看到的放大
    倍率就差 23%** —— 这就是未闭环 #1 的根因，与"低对比度"无关。

    两种归一化（都是可关闭的实验变体）：

    * ``mode="scalar"``（**B2**）：以**全片主导带内频点**的纯正弦 conc 作分母
      ``w = conc / c_ref``。只改**绝对增益**、**完全保持**像素间的比值（选择性不变），
      因此专门用来消除 T 依赖；上界默认 ``1/c_ref``（允许可靠性高于"参考纯正弦"的
      像素按比例多放大一点，而不是被裁成 1 而丢失选择性）。
    * ``mode="pixel"``（**B3**）：逐像素用其**主导频点**的纯正弦 conc 作分母，并减去
      噪声 null ``w = (conc−null)/(c_ref(b*)−null)``。对**正好落在频点上**的带内正弦
      给出 1，但对**偏离频点/贴近带边**的真实运动会被过度压制（实测 0.35 Hz、
      T=120、带 0.3–1.0 Hz 时降为 0）—— 属于"窄带 + 短窗"的固有限制，如实保留。

    Args:
        evidence: ``[T,h,w]``。``domain="bp"`` 时必须**已经是带通后的** ``Φ_bp``
            （不再重复滤波）；``domain="raw"`` 时传原始累积相位 ``Φ``。
        floor: 仅 ``mode="pixel"`` 用；给 ``c_ref−null`` 设下限，避免窄带下除以近零。
        upper_clip: 上界；``None`` 时 scalar 用 ``1/c_ref``、pixel 用 1。
    """
    T = evidence.shape[0]
    if T < 8:
        return np.ones(evidence.shape[1:], np.float32)
    conc, p, band, _ = concentration_map(evidence, fps, f_lo, f_hi, taper, eps)
    ref = occupancy_reference(T, fps, f_lo, f_hi, domain, taper)
    if len(ref["refs"]) == 0:
        return np.ones(evidence.shape[1:], np.float32)
    pb = p[band]
    if mode == "scalar":
        axes = tuple(range(1, pb.ndim))
        b = int(np.argmax(pb.sum(axis=axes)))
        c_ref = float(ref["refs"][b])
        w = conc / max(c_ref, 1e-6)
        hi = float(upper_clip) if upper_clip is not None else float(1.0 / max(c_ref, 1e-6))
    elif mode == "pixel":
        b_star = np.argmax(pb, axis=0)
        c_ref = ref["refs"][np.minimum(b_star, len(ref["refs"]) - 1)]
        denom = np.maximum(c_ref - ref["null"], float(floor))
        w = (conc - ref["null"]) / denom
        hi = float(upper_clip) if upper_clip is not None else 1.0
    else:
        raise ValueError(f"mode 只能是 scalar 或 pixel，收到 {mode}")
    return np.clip(w, 0.0, hi).astype(np.float32)


def robust_temporal_smooth(Phi_bp: np.ndarray, k: int = 5,
                           mode: str = "median") -> np.ndarray:
    """**稳健时间抑噪**（非线性，任务书 §5 D2）：对相位变化做时间滑动中位数/均值。

    与既有零相位 Butterworth 带通的**区别**（任务书要求说明，不得重复叠同一滤波器）：

    * 带通是**线性时不变**、已存在的阶段 1 组件；
    * 本函数是**非线性**（滑动中位数），抗脉冲性噪声，且**不改变零相位性质**；
    * 它不重复带通，而是对**已带通**的相位做稳健平滑 —— 目的是抑制个别帧的相位野值。

    频响用 :func:`robust_temporal_gain` 实测（中位数滤波不是 LTI，只在正弦上测等效增益）。
    边界处理：``mode='nearest'``（两端复制），不引入边界振荡。
    """
    from scipy import ndimage as ndi
    k = int(k)
    if k <= 1:
        return Phi_bp
    k = k | 1                                   # 取奇数核，保证时间对称
    if mode == "mean":
        return ndi.uniform_filter1d(Phi_bp, size=k, axis=0, mode="nearest")
    return ndi.median_filter(Phi_bp, size=(k, 1, 1), mode="nearest")


def robust_temporal_gain(fps: float, k: int, freqs, mode: str = "median",
                         n: int = 256) -> Dict[float, float]:
    """实测稳健时间抑噪的**等效频响**（正弦输入，取中段幅度比）。

    这不是 LTI 系统的传递函数，而是"正弦通过后的等效增益"，用于说明该步骤
    在目标频带内**不应显著衰减**、在带外（尤其高频）应有所抑制。
    """
    t = np.arange(n) / float(fps)
    seg = slice(n // 4, 3 * n // 4)
    out = {}
    for f in freqs:
        x = np.sin(2 * np.pi * float(f) * t)[:, None, None] * np.ones((1, 4, 4))
        y = robust_temporal_smooth(x, k, mode)
        out[float(f)] = float(np.abs(y[seg]).max() / max(np.abs(x[seg]).max(), EPS))
    return out


def coherence_weight(inc_complex: np.ndarray, radius: int = 1,
                     eps: float = EPS) -> np.ndarray:
    """**局部空间一致性**：``w = |<unit(Δφ)>|``，在 ``radius`` 邻域内取平均。

    ``inc_complex`` 是**单位相位增量复数** ``exp(i·δφ)``（形状 ``[T, h, w]``）。
    邻域内的相位增量方向越一致，模长越接近 1；在活动区边界处会自然下降，
    因此它**同时**压低渗漏（这正是 §7.3 要观察的）。
    **不**与背景运动对齐 —— 只要求局部自洽（任务书 §2.2 的警告）。
    时间维的一致性由 :func:`smooth_weight` 的 ``temporal_kernel`` 承担。
    """
    from scipy import ndimage as ndi
    z = inc_complex
    if radius <= 0:
        return np.abs(z)
    k = 2 * int(radius) + 1
    re = ndi.uniform_filter(z.real, size=(1, k, k), mode="reflect")
    im = ndi.uniform_filter(z.imag, size=(1, k, k), mode="reflect")
    return np.sqrt(re * re + im * im)


# --------------------------------------------------------------------------- #
# 标定与配置的落盘/读取（实验可复现）
# --------------------------------------------------------------------------- #
def save_calibration(calib: Dict, path) -> None:
    """把 :func:`calibrate_noise` 的结果写成 JSON（键转字符串）。"""
    import json
    from pathlib import Path
    d = {"sigma_A": {f"{a}-{b}": v for (a, b), v in calib["sigma_A"].items()},
         "sigma_phi": {f"{a}-{b}": v for (a, b), v in calib["sigma_phi"].items()},
         "band_stats": {f"{a}-{b}": v for (a, b), v in calib["band_stats"].items()},
         "config": calib["config"]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(d, ensure_ascii=False, indent=2, default=str),
                          encoding="utf-8")


def load_calibration(path) -> Dict:
    import json
    from pathlib import Path
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    def unfix(dd):
        return {tuple(int(x) for x in k.split("-")): v for k, v in dd.items()}
    return {"sigma_A": unfix(d["sigma_A"]), "sigma_phi": unfix(d["sigma_phi"]),
            "band_stats": {k: v for k, v in d["band_stats"].items()},
            "config": d.get("config", {})}


def config_from_yaml(path) -> ReliabilityConfig:
    """从 YAML 读阶段 2 配置。

    支持 ``sigma_from: <calibration.json>``（把标定结果填进 ``sigma_A``/``sigma_phi``），
    以及 ``per_band_caps: {"0-0": 2.0, ...}``。
    """
    import yaml
    from pathlib import Path
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    sigma_from = raw.pop("sigma_from", None)
    cfg = ReliabilityConfig(**{k: v for k, v in raw.items()
                               if k in ReliabilityConfig.__dataclass_fields__})
    if sigma_from:
        calib = load_calibration(sigma_from)
        cfg.sigma_A = calib["sigma_A"]
        cfg.sigma_phi = calib["sigma_phi"]
    return cfg.resolved()


def smooth_weight(w: np.ndarray, spatial_sigma: float = 0.0,
                  temporal_kernel: int = 0, eps: float = EPS
                  ) -> np.ndarray:
    """对权重做空间高斯 + 时间对称（零相位）平滑，避免空间/时间跳变。

    ⚠️ 平滑是**代价与收益并存**的（任务书 §2.3 要求检查）：空间平滑会让运动区
    向背景"泄漏"（权重被抬高），时间平滑会让短时真实运动被抹掉。
    因此增益/保留率必须实测，不能默认平滑有益。
    """
    from scipy import ndimage as ndi
    out = w.astype(np.float32, copy=True)
    if spatial_sigma and spatial_sigma > 0:
        out = ndi.gaussian_filter(out, sigma=(0.0, spatial_sigma, spatial_sigma),
                                  mode="reflect")
    if temporal_kernel and temporal_kernel > 1:
        k = int(temporal_kernel) | 1                       # 取奇数，保证对称
        out = ndi.uniform_filter1d(out, size=k, axis=0, mode="nearest")
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# 逐尺度增益上限（§3）：只提供"从统计出发"的建议，绝不静默自动生效
# --------------------------------------------------------------------------- #
def cap_suggestions(band_stats: Dict, alpha_max: float = 10.0,
                    snr_floor: float = 1.0) -> Dict:
    """由实测统计给出**建议**的逐带上限（供消融设计，不自动应用）。

    规则（可解释、可复现）：相位噪声比 ``sigma_phi`` 越大、``dphi_abs_mean`` 越小，
    说明该带的相位变化越可能是噪声 ⇒ 该带的额外增益上限应越小。
    ``β = alpha_max · clip(dphi_abs_mean / (snr_floor·sigma_phi), 0, 1)``。
    **注意**：这只是实验设计的起点，任务书禁止无实验地手填数值，
    因此本函数只返回建议，脚本必须打印出来并逐项消融验证。
    """
    out = {}
    for k, st in band_stats.items():
        num = st["dphi_abs_mean"]
        den = max(snr_floor * st["sigma_phi"], EPS)
        out[k] = float(alpha_max * min(1.0, num / den))
    return out


# --------------------------------------------------------------------------- #
# 权重合成
# --------------------------------------------------------------------------- #
def combine_weights(w_list, mode: str = "product") -> np.ndarray:
    """把多种证据合成为最终权重。默认相乘（独立证据），并截断到 [0,1]。"""
    w = None
    for x in w_list:
        w = x if w is None else (w * x if mode == "product" else np.maximum(w, x))
    return np.clip(w, 0.0, 1.0) if w is not None else np.ones(1, np.float32)