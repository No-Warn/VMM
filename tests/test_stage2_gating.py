# -*- coding: utf-8 -*-
"""阶段 2 门控测试：M0 不变性、门控范围、α=0 恒等、逐带上限、噪声标定、掩码工具。

设计原则：**默认配置必须与阶段 1 逐位一致**（这是阶段 2 的硬性前提），
因此 ``test_m0_matches_stage1_bitwise`` 用同一随机种子比较 M0 与"无 cfg"两条路径。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vmm import metrics, synthetic                       # noqa: E402
from vmm.phase_magnifier import PhaseBasedMagnifier      # noqa: E402
from vmm.reliability import (ReliabilityConfig, amp_floor_weight,   # noqa: E402
                             amplitude_weight, band_occupancy_normalized,
                             band_occupancy_weight, calibrate_noise,
                             coherence_weight, occupancy_reference,
                             stability_weight, smooth_weight)


def _seq(kind="shm_h", T=48, S=96, noise=0.0, seed=3):
    fr, gt = synthetic.make_sequence(kind=kind, T=T, H=S, W=S, fps=30.0,
                                     amp_px=0.5, freq=0.5, noise_sigma=noise, seed=seed)
    return fr, gt


def _mag(**kw):
    base = dict(fps=30.0, f_lo=0.3, f_hi=1.0, levels=3, order=3, channels="gray")
    base.update(kw)
    return PhaseBasedMagnifier(**base)


# --------------------------------------------------------------------------- #
# M0 不变性：阶段 2 的默认路径必须与阶段 1 完全一致
# --------------------------------------------------------------------------- #
def test_m0_matches_stage1_bitwise():
    fr, _ = _seq()
    a = _mag().amplify(fr, 5.0)[0]
    b = _mag(reliability_cfg=ReliabilityConfig(mode="M0")).amplify(fr, 5.0)[0]
    assert np.array_equal(a, b), "M0 默认路径与阶段 1 不一致（必须逐位相同）"


def test_alpha_zero_identity_with_gate_enabled():
    """α=0 必须仍然是严格恒等，即使门控/上限已开启（不得破坏恒等）。"""
    fr, _ = _seq()
    cfg = ReliabilityConfig(mode="M3", sigma_A={(0, 0): 0.05}, sigma_phi={(0, 0): 0.1},
                            amp_floor=0.3, per_band_caps={(0, 0): 1.0})
    out, _ = _mag(reliability_cfg=cfg).amplify(fr, 0.0)
    assert np.abs(out - fr).max() <= 1e-4


# --------------------------------------------------------------------------- #
# 权重函数的数学性质（范围、单调性、不清零）
# --------------------------------------------------------------------------- #
def test_amplitude_weight_range_and_monotone():
    A = np.linspace(0, 1, 50)
    w = amplitude_weight(A, sigma_A=0.1)
    assert w.min() >= 0.0 and w.max() <= 1.0
    assert np.all(np.diff(w) >= -1e-12), "w_A 必须随 A 单调不减"
    assert w[0] < 0.01 and w[-1] > 0.98


def test_amp_floor_is_soft_not_hard_zero():
    """§4 要求：amp_floor 是软阈值，不得把低幅值系数清零。"""
    A = np.array([1e-4, 1e-3, 0.01, 0.1, 1.0])
    w = amp_floor_weight(A, A_median=0.1, amp_floor=0.5)
    assert np.all(w > 0), "amp_floor 不得产生硬 0（会丢低对比度真实运动）"
    assert np.all(np.diff(w) >= -1e-12) and w[-1] > w[0]
    assert amp_floor_weight(A, 0.1, 0.0).tolist() == [1.0] * 5, "amp_floor=0 必须完全关闭"


def test_stability_and_coherence_range():
    x = np.linspace(-2, 2, 41)[:, None, None] * np.ones((1, 8, 8))
    ws = stability_weight(x, sigma_phi=0.5)
    assert ws.min() >= 0 and ws.max() <= 1
    z = np.ones((6, 8, 8), np.complex64)                 # 完全一致的相位增量
    assert coherence_weight(z, radius=1).min() > 0.99
    z2 = np.exp(1j * np.pi * np.arange(6)[:, None, None])  # 相邻帧反向
    assert coherence_weight(z2 * np.ones((1, 8, 8)), radius=0).max() <= 1.0 + 1e-6


def test_band_occupancy_separates_inband_from_broadband():
    """长时窗谱集中度：**带内正弦** ⇒ ≈1；**宽带白噪声** ⇒ 明显更低。

    注意测试用**较高**的目标带（2–5 Hz）：低带（如 0.3–1 Hz）下 Hann 窗主瓣
    与 1/f 噪声都会落进带内，机制本身就不灵敏 —— 这一点在真实低带上的表现
    由 `scripts/stage2_ablation.py` 如实报告，不在单元测试里假装它一定成立。
    """
    from vmm.reliability import band_occupancy_weight
    T, H, W, fps = 120, 8, 8, 30.0
    t = np.arange(T) / fps
    inband = np.sin(2 * np.pi * 3.5 * t)[:, None, None] * np.ones((1, H, W))
    rng = np.random.default_rng(0)
    white = rng.normal(0, 1, (T, H, W))                     # 白噪声（宽带）
    w_in = band_occupancy_weight(inband, fps, 2.0, 5.0)
    w_no = band_occupancy_weight(white, fps, 2.0, 5.0)
    assert 0.0 <= w_in.min() and w_in.max() <= 1.0
    assert w_in.mean() > 0.95, f"带内正弦的谱集中度应≈1，实测 {w_in.mean():.3f}"
    assert w_no.mean() < w_in.mean() - 0.2, f"宽带噪声应明显更低：{w_no.mean():.3f}"
    # 零信号（无证据）必须给 0，而不是给 1
    assert band_occupancy_weight(np.zeros((T, H, W)), fps, 2.0, 5.0).max() < 1e-6


def test_smooth_weight_stays_bounded_and_preserves_shape():
    rng = np.random.default_rng(0)
    w = rng.random((10, 16, 16)).astype(np.float32)
    s = smooth_weight(w, spatial_sigma=1.0, temporal_kernel=3)
    assert s.shape == w.shape and s.min() >= 0 and s.max() <= 1
    # 平滑后总变差应下降（说明确实在压跳变）
    tv = lambda a: np.abs(np.diff(a, axis=1)).sum() + np.abs(np.diff(a, axis=2)).sum()
    assert tv(s) < tv(w)


# --------------------------------------------------------------------------- #
# [B2] 归一化谱集中度：修掉 B1 的 **T 依赖整体偏移**（阶段 2 未闭环 #1）
# --------------------------------------------------------------------------- #
def test_b1_offset_depends_on_T_while_b2_is_T_invariant():
    """同一**纯带内正弦**（0.5 Hz、带 0.3–1.0 Hz、fps=30）：

    * B1 的 w 随序列长度大幅漂移（实测 0.998 / 0.719 / 0.994 @ T=64/120/240）——
      它被直接乘在 α 上，于是**同一条视频只因截断长度不同，放大倍率就差 23%**；
      B1 本身没有任何错，错的是"把无量纲比值当成概率直接乘 α"。
    * B2（scalar 归一化）必须把这三者拉回 ≈1。

    这是未闭环 #1 的**回归测试**：若有人改回"直接用 B1 的 w 乘 α"，此测试会失败。
    """
    from vmm.reliability import band_occupancy_normalized
    from vmm.temporal_filter import bandpass_filter, bandpass_time
    fps, flo, fhi, f0 = 30.0, 0.3, 1.0, 0.5
    filt = bandpass_filter(fps, flo, fhi, 2)
    w1, w2 = [], []
    for T in (64, 120, 240):
        t = np.arange(T) / fps
        tone = np.sin(2 * np.pi * f0 * t)[:, None, None] * np.ones((1, 4, 4))
        bp = bandpass_time(tone, filt)
        w1.append(float(band_occupancy_weight(bp, fps, flo, fhi).mean()))
        w2.append(float(band_occupancy_normalized(bp, fps, flo, fhi, domain="bp",
                                                  mode="scalar").mean()))
    assert max(w1) - min(w1) > 0.2, f"B1 应复现 T 依赖（未闭环 #1）：{w1}"
    assert min(w1) < 0.75, f"B1 在 T=120 的偏移应≈0.72：{w1}"
    assert max(w2) - min(w2) < 0.05, f"B2 归一化后必须与 T 无关：{w2}"
    assert min(w2) > 0.95, f"纯带内正弦的 B2 应≈1：{w2}"


def test_b2_scalar_normalization_is_not_a_noise_detector():
    """**已知局限（如实固定，不是"调测试到通过"）**：

    B2 的参考 ``c_ref`` 取自**本片自己的主导带内频点**。若输入**完全没有运动、
    只有噪声**，主导频点就是噪声的峰 ⇒ ``c_ref`` 被拉到噪声自身的谱形状 ⇒
    ``w≈1``（实测 0.85–1.11），**不会拒绝噪声**。

    因此 B2 的定位只能是"**消除 B1 的 T 依赖增益漂移**"，不是"噪声判别器"。
    端到端上它仍未比 E0 差（`static_noise` α=5、T=120：E0 1.151 → B2 1.038），
    因为真实输入里总有运动把主导频点钉在目标带上；但**不得**把 B2 当成独立抑噪手段
    （见 `docs/stage2_results.md` §11.5）。
    """
    from vmm.temporal_filter import bandpass_filter, bandpass_time
    fps, flo, fhi, T = 30.0, 0.3, 1.0, 240
    filt = bandpass_filter(fps, flo, fhi, 2)
    t = np.arange(T) / fps
    tone = np.sin(2 * np.pi * 0.5 * t)[:, None, None] * np.ones((1, 4, 4))
    w_tone = float(band_occupancy_normalized(
        bandpass_time(tone, filt), fps, flo, fhi, mode="scalar").mean())
    w_noise = []
    for seed in range(3):
        rw = np.cumsum(np.random.default_rng(seed).standard_normal((T, 4, 4)), axis=0)
        w_noise.append(float(band_occupancy_normalized(
            bandpass_time(rw, filt), fps, flo, fhi, mode="scalar").mean()))
    assert w_tone > 0.95, f"纯正弦必须≈1，实测 {w_tone:.3f}"
    assert max(w_noise) > 0.85, (
        f"若哪天 B2 真能拒绝纯噪声，说明参考改成了数据无关的标定 —— "
        f"那时请更新本文档与本测试。实测 {w_noise} vs {w_tone:.3f}")


def test_occupancy_reference_shapes():
    from vmm.reliability import occupancy_reference
    ref = occupancy_reference(120, 30.0, 0.3, 1.0, domain="bp")
    assert len(ref["refs"]) == 3, "带 0.3–1.0 Hz @fps=30、T=120 应有 3 个频点"
    assert np.all((ref["refs"] > 0.0) & (ref["refs"] <= 1.0))
    assert 0.0 <= ref["null"] < 1.0
    raw = occupancy_reference(120, 30.0, 0.3, 1.0, domain="raw")
    assert len(raw["refs"]) == 3
    assert raw["null"] < ref["null"], "原始相位的噪声 null 应低于带通后（带通把噪声挤进带内）"


def test_b1_keeps_its_exact_reference_formula():
    """重构出 ``concentration_map`` 后，B1 的数值必须与第一轮**逐字公式**一致。"""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((120, 8, 8))
    w = band_occupancy_weight(x, 30.0, 0.3, 1.0)
    T = x.shape[0]
    y = x - x.mean(axis=0, keepdims=True)
    y = y * np.hanning(T)[:, None, None]
    F = np.fft.rfft(y, axis=0)
    fq = np.fft.rfftfreq(T, d=1.0 / 30.0)
    b = (fq >= 0.3) & (fq <= 1.0)
    p = np.abs(F) ** 2
    num = p[b].sum(axis=0)
    den = p[~b].sum(axis=0) + 1e-12
    ref = (num / (num + den + 1e-12)).astype(np.float32)
    assert np.allclose(w, ref, atol=1e-6), "B1 的数值被重构改变了（必须逐位等价）"


def test_b2_alpha_zero_still_identity():
    """B2 开启时 α=0 仍必须严格恒等（新增门控不得破坏恒等性）。"""
    fr, _ = _seq()
    cfg = ReliabilityConfig(mode="B2")
    out, _ = _mag(reliability_cfg=cfg).amplify(fr, 0.0)
    assert np.abs(out - fr).max() <= 1e-4


def test_b2_matches_e0_gain_on_clean_motion():
    """干净局部运动：B1 在 T=120 丢 ~23% 增益，B2 必须与 E0 一致（±3%）。"""
    from vmm.reliability import ReliabilityConfig
    fr, gt = synthetic.make_sequence(kind="local", T=120, H=96, W=96, fps=30.0,
                                     amp_px=0.5, freq=0.5, patch=32, seed=42)
    roi = synthetic.measurement_roi(gt)

    def gain(cfg):
        mag = PhaseBasedMagnifier(fps=30.0, f_lo=0.3, f_hi=1.0, levels=3, order=3,
                                  channels="gray", reliability_cfg=cfg)
        out, _ = mag.amplify(fr, 5.0)
        a_in = metrics.waveform_amplitude(
            metrics.lk_displacement_to_reference(fr, ref="first", roi=roi))[0]
        a_out = metrics.waveform_amplitude(
            metrics.lk_displacement_to_reference(out, ref="first", roi=roi))[0]
        return a_out / a_in

    g0 = gain(ReliabilityConfig(mode="M0"))
    g1 = gain(ReliabilityConfig(mode="B1"))
    g2 = gain(ReliabilityConfig(mode="B2"))
    assert g1 / g0 < 0.95, f"B1 在 T=120 应复现 ~23% 增益损失：g1/g0={g1/g0:.3f}"
    assert g2 / g0 > 0.97, f"B2 必须恢复 E0 的增益：g2/g0={g2/g0:.3f}"


# --------------------------------------------------------------------------- #
# 噪声标定（尺度统一）与逐带上限
# --------------------------------------------------------------------------- #
def test_calibrate_noise_shapes_and_scale():
    fr, _ = _seq(kind="static_noise", noise=0.02, seed=11)
    calib = calibrate_noise(fr, levels=3, order=3)
    n_bands = len(calib["sigma_A"])
    assert n_bands > 0 and n_bands == len(calib["sigma_phi"]) == len(calib["band_stats"])
    for k, st in calib["band_stats"].items():
        assert calib["sigma_A"][k] > 0 and calib["sigma_phi"][k] > 0
        # σ 必须与 A 同量级（尺度统一）：比值落在合理范围
        assert 1e-6 < st["noise_over_signal_A"] < 10.0
    # 噪声越大，标定出的 σ_A 应越大（单调性检查）
    fr2, _ = _seq(kind="static_noise", noise=0.06, seed=11)
    calib2 = calibrate_noise(fr2, levels=3, order=3)
    k0 = next(iter(calib["sigma_A"]))
    assert calib2["sigma_A"][k0] > calib["sigma_A"][k0]


def test_per_band_cap_limits_effective_gain():
    """逐带上限 α_eff=min(α,β)：门控诊断里必须能看到 α_eff 被压制。"""
    fr, _ = _seq()
    cfg = ReliabilityConfig(mode="M0", per_band_caps={(0, 0): 1.0})
    _, d = _mag(reliability_cfg=cfg).amplify(fr, 10.0)
    for name, rec in d["per_band"].items():
        if name == "0-0":
            assert rec["alpha_eff"] == pytest.approx(1.0)
            assert rec["cap"] == pytest.approx(1.0)
        else:
            assert rec["alpha_eff"] == pytest.approx(10.0)


def test_gate_reduces_suppressed_fraction_on_noise_only():
    """静止噪声输入上，M1 门控应把更多像素判为不可靠（w<0.1 比例上升）。"""
    fr, _ = _seq(kind="static_noise", noise=0.04, seed=5)
    calib = calibrate_noise(fr, levels=3, order=3)
    cfg = ReliabilityConfig(mode="M1", sigma_A=calib["sigma_A"])
    _, d = _mag(reliability_cfg=cfg).amplify(fr, 5.0)
    fr_h, _ = _seq(kind="shm_h", noise=0.0, seed=5)
    calib_h = calibrate_noise(fr_h, levels=3, order=3)
    cfg_h = ReliabilityConfig(mode="M1", sigma_A=calib_h["sigma_A"])
    _, d_h = _mag(reliability_cfg=cfg_h).amplify(fr_h, 5.0)
    f_noise = np.mean([r["w_suppressed_frac"] for r in d["per_band"].values()])
    f_motion = np.mean([r["w_suppressed_frac"] for r in d_h["per_band"].values()])
    assert f_noise > f_motion, f"噪声输入的抑制比例应更高：{f_noise} vs {f_motion}"


# --------------------------------------------------------------------------- #
# 空间渗漏掩码
# --------------------------------------------------------------------------- #
def test_ring_and_far_masks():
    m = np.zeros((64, 64), bool)
    m[24:40, 24:40] = True
    ring, far = metrics.ring_and_far_masks(m, ring_px=4, far_px=12)
    assert ring.any() and far.any()
    assert not (ring & m).any(), "环不得包含活动区"
    assert not (far & m).any() and not (far & ring).any()
    assert far.sum() > ring.sum()