# -*- coding: utf-8 -*-
"""Phase G：多帧相干积累的单元/集成测试。

覆盖任务书 §9 要求：α=0 恒等、静态干净、静态噪声、H/V/局部、不同噪声、
不同窗长、首尾帧、分块边界、NaN/Inf、帧数不足窗长、可靠性不足回退。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vmm import metrics, synthetic                                  # noqa: E402
from vmm.coherent import (analytic_signal, coherent_amplitude,      # noqa: E402
                          coherent_noise_level, coherent_weight,
                          required_margin, window_stat)
from vmm.phase_magnifier import PhaseBasedMagnifier                 # noqa: E402
from vmm.reliability import ReliabilityConfig                       # noqa: E402


def _mag(**kw):
    base = dict(fps=30.0, f_lo=0.3, f_hi=1.0, levels=3, order=3, channels="gray")
    base.update(kw)
    return PhaseBasedMagnifier(**base)


def _seq(kind="local", T=64, S=96, amp=0.5, noise=0.0, seed=42):
    return synthetic.make_sequence(kind=kind, T=T, H=S, W=S, fps=30.0, amp_px=amp,
                                   freq=0.5, noise_sigma=noise, patch=S // 3,
                                   seed=seed, noise_seed=seed + 7)


# --------------------------------------------------------------------------- #
# 数学性质
# --------------------------------------------------------------------------- #
def test_analytic_signal_real_part_and_envelope():
    T, fps, f, A = 240, 30.0, 3.0, 0.7
    t = np.arange(T) / fps
    x = A * np.sin(2 * np.pi * f * t)[:, None, None] * np.ones((1, 4, 4))
    z = analytic_signal(x)
    assert np.allclose(z.real, x, atol=1e-12), "解析信号实部必须等于输入"
    assert np.allclose(np.abs(z).mean(axis=0), A, rtol=2e-3), "纯正弦的包络 |z| 应≈幅度"
    assert np.allclose(analytic_signal(np.zeros((16, 2, 2))), 0)


def test_window_stat_identity_and_robustness():
    x = np.ones((21, 4, 4))
    assert np.allclose(window_stat(x, 1), x)
    assert np.allclose(window_stat(x, 9), x)
    # 中位数对单个野值稳健，均值不稳健
    y = np.zeros((21, 1, 1))
    y[0] = 100.0
    assert abs(float(window_stat(y, 5, "median").mean())) < 1e-9
    assert float(window_stat(y, 5, "mean").mean()) > 1.0
    # 偶数窗强制取奇数（对称，避免相位偏移）
    assert window_stat(np.zeros((10, 2, 2)), 6).shape == (10, 2, 2)


def test_coherent_amplitude_keeps_tone_and_stabilizes_noise():
    """相干检测：纯正弦包络与 W 无关；噪声的**波动**随 W 下降（均值不变）。"""
    T, fps, f, A = 240, 30.0, 3.0, 0.5
    t = np.arange(T) / fps
    tone = A * np.sin(2 * np.pi * f * t)[:, None, None] * np.ones((1, 8, 8))
    vals = [float(coherent_amplitude(tone, w).mean()) for w in (1, 3, 5, 9, 15)]
    assert max(vals) - min(vals) < 0.02 * max(vals), f"纯正弦包络不应随 W 变化：{vals}"
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((T, 8, 8))
    cv = [float(coherent_amplitude(noise, w).std() / coherent_amplitude(noise, w).mean())
          for w in (1, 9)]
    assert cv[1] < 0.6 * cv[0], f"噪声包络的相对波动应随 W 下降：{cv}"


def test_coherent_weight_forms():
    """excess 形式：纯噪声→0；snr 形式：纯噪声→0.5（说明公式选择影响 Gate）。"""
    T, fps = 120, 30.0
    t = np.arange(T) / fps
    rng = np.random.default_rng(1)
    noise = rng.standard_normal((T, 1, 1))
    sigma = float(coherent_amplitude(noise, 5).mean())
    w_ex = coherent_weight(noise, 5, sigma, form="excess")
    w_sn = coherent_weight(noise, 5, sigma, form="snr")
    assert w_ex.mean() < 0.25, f"excess 对纯噪声应接近 0：{w_ex.mean()}"
    assert 0.35 < w_sn.mean() < 0.65, f"snr 对纯噪声应在 0.5 附近：{w_sn.mean()}"
    tone = 5.0 * np.sin(2 * np.pi * 3.0 * t)[:, None, None]
    assert coherent_weight(tone, 5, sigma, form="excess").min() > 0.9


def test_required_margin():
    assert required_margin(1) == 0
    assert required_margin(9) == 4
    assert required_margin(15) == 7


# --------------------------------------------------------------------------- #
# 集成：恒等 / 静态 / NaN / 帧数不足
# --------------------------------------------------------------------------- #
def test_alpha_zero_identity_with_coherent_modes():
    fr, _ = _seq(T=32)
    for mode in ("coherent", "mean", "median"):
        cfg = ReliabilityConfig(mode="M0", coherent_mode=mode, coherent_window=5,
                                coherent_sigma={(0, 0): 0.1, (0, 1): 0.1, (0, 2): 0.1,
                                                (0, 3): 0.1, (1, 0): 0.1, (1, 1): 0.1,
                                                (1, 2): 0.1, (1, 3): 0.1, (2, 0): 0.1,
                                                (2, 1): 0.1, (2, 2): 0.1, (2, 3): 0.1})
        out, _ = _mag(reliability_cfg=cfg).amplify(fr, 0.0)
        assert np.abs(out - fr).max() <= 1e-4, f"{mode} 下 α=0 必须恒等"


def test_static_clean_stays_identity():
    fr, _ = _seq(kind="static", T=32)
    cfg = ReliabilityConfig(mode="M0", coherent_mode="mean", coherent_window=9)
    out, _ = _mag(reliability_cfg=cfg).amplify(fr, 5.0)
    assert np.abs(out - fr).max() <= 1e-4


def test_static_noise_finite_and_no_nan():
    fr, _ = _seq(kind="static_noise", T=48, noise=0.02)
    cfg = ReliabilityConfig(mode="M0", coherent_mode="coherent", coherent_window=9,
                            coherent_sigma={(l, o): 0.1 for l in range(3) for o in range(4)})
    out, d = _mag(reliability_cfg=cfg).amplify(fr, 5.0)
    assert np.all(np.isfinite(out))
    art = metrics.artifact_report(out)
    assert art["has_nan"] == 0.0 and art["has_inf"] == 0.0


def test_window_longer_than_sequence():
    """T=20 < W=31：必须靠 reflect 延拓正常运行，不崩、不 NaN。

    注意：零相位带通的 `filtfilt` 需要 ``T > padlen``（order=2 时 15 帧），
    因此这里取 T=20（>15）而窗长 31（>T）—— 测的才是"窗长超过序列"。
    """
    fr, _ = _seq(T=20, S=64)
    cfg = ReliabilityConfig(mode="M0", coherent_mode="mean", coherent_window=31)
    out, _ = _mag(reliability_cfg=cfg).amplify(fr, 2.0)
    assert out.shape == fr.shape and np.all(np.isfinite(out))


def test_adaptive_fallback_on_static_noise():
    """静态+噪声上相干证据整体不可靠 ⇒ 必须回退（coherent_fallback=1）。"""
    fr, _ = _seq(kind="static_noise", T=64, noise=0.03)
    cfg = ReliabilityConfig(mode="M0", coherent_mode="adaptive", coherent_window=9,
                            coherent_tau=0.9,
                            coherent_sigma={(l, o): 1e3 for l in range(3) for o in range(4)})
    _, d = _mag(reliability_cfg=cfg).amplify(fr, 5.0)
    flags = [r.get("coherent_fallback") for r in d["per_band"].values()
             if "coherent_fallback" in r]
    assert flags and all(f == 1.0 for f in flags), f"应回退到 B2：{flags}"


def test_hv_local_motion_direction_kept():
    """H/V/局部运动的方向不能反（与阶段 1 同一判据：运动轴相关 > 0）。"""
    for kind in ("shm_h", "shm_v", "local"):
        fr, gt = _seq(kind=kind, T=64, amp=0.5)
        cfg = ReliabilityConfig(mode="M0", coherent_mode="coherent", coherent_window=5,
                                coherent_sigma={(l, o): 0.05 for l in range(3) for o in range(4)})
        out, _ = _mag(reliability_cfg=cfg).amplify(fr, 5.0)
        roi = synthetic.measurement_roi(gt)
        w_in = metrics.lk_displacement_to_reference(fr, ref="first", roi=roi)
        w_out = metrics.lk_displacement_to_reference(out, ref="first", roi=roi)
        gta = synthetic.gt_amplitude(gt)
        axis = 0 if gta[0] >= gta[1] else 1
        corr = metrics.waveform_correlation(w_in, w_out)[axis]
        assert corr > 0.5, f"{kind} 波形相关过低（方向/波形被破坏）：{corr}"


def test_chunked_boundary_with_coherent_window():
    """分块时窗需要跨块上下文：margin ≥ (W−1)/2 时接缝不应显著变差。"""
    fr, _ = _seq(kind="shm_h", T=160, S=64, amp=0.5)
    cfg = ReliabilityConfig(mode="M0", coherent_mode="mean", coherent_window=9)
    mag = _mag(reliability_cfg=cfg)
    out, meta = mag.amplify_chunked(fr, 3.0, chunk_frames=60, margin_frames=8)
    assert meta["chunked"] is True or fr.shape[0] <= 60 + 16
    assert out.shape == fr.shape and np.all(np.isfinite(out))
    seam, _, _ = metrics.seam_score(out, [60, 120])
    assert seam < 5.0, f"接缝分数异常：{seam}"
