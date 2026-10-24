# -*- coding: utf-8 -*-
"""阶段 1 相移/增益测试。

测试只覆盖**已经验证过的线性区**（alpha ≤ 2），因为实测发现 alpha ≥ 5 时相位平移
超出线性区，输出不再是干净平移（增益饱和到 ~5，而不是理论 1+alpha），
与估计器可靠性问题叠加后不再适合作为断言对象。该适用域边界本身作为
`scripts/evaluate_synthetic.py` 的"附加发现"上报，不伪装成通过项。

测振幅一律用**相对参考帧的位移波形**（独立估计器 = 相位相关），
不读算法内部相位。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vmm import metrics, synthetic                  # noqa: E402
from vmm.phase_magnifier import PhaseBasedMagnifier  # noqa: E402


def _run(frames, alpha, **kw):
    base = dict(fps=30.0, f_lo=0.3, f_hi=1.0, levels=3, order=3, channels="gray")
    base.update(kw)
    out, _ = PhaseBasedMagnifier(**base).amplify(frames, alpha=alpha)
    return out


def _amp(video):
    wave = metrics.displacement_to_reference(video)
    a, _, _ = metrics.waveform_amplitude(wave)
    return a


def test_amplitude_increases_with_alpha_in_linear_regime():
    frames, _ = synthetic.make_sequence(kind="shm_h", T=60, H=96, W=96, fps=30.0,
                                        amp_px=0.5, freq=0.5, seed=4)
    amps = [_amp(_run(frames, a)) for a in (0.0, 1.0, 2.0)]
    assert amps[1] > amps[0] * 1.3, f"alpha=1 未带来明显增益：{amps}"
    assert amps[2] > amps[1], f"增益未随 alpha 单调（线性区）：{amps}"


def test_still_amplifies_beyond_linear_regime_but_saturates():
    """alpha=5 仍在放大（但不是 1+alpha）—— 记录适用域边界，不假装它等于理论值。"""
    frames, _ = synthetic.make_sequence(kind="shm_h", T=60, H=96, W=96, fps=30.0,
                                        amp_px=0.5, freq=0.5, seed=4)
    a0 = _amp(_run(frames, 0.0))
    a5 = _amp(_run(frames, 5.0))
    assert a5 > 1.5 * a0, f"alpha=5 未放大：{a0:.3f} -> {a5:.3f}"


def test_direction_not_reversed():
    """方向不反转 ⇒ 输入/输出**位移波形**正相关（线性区）。

    注意：不能用相邻帧速度序列做相关 —— 0.5px/0.5Hz 运动的逐帧速度只有约
    0.05 px，而相位相关的噪声底约 0.10 px，输入侧基本是噪声，会把相关系数
    压到 0.4 以下，从而**误判为方向反转**（实测教训，见 vmm/metrics.py）。
    """
    frames, gt = synthetic.make_sequence(kind="shm_h", T=60, H=96, W=96, fps=30.0,
                                         amp_px=0.5, freq=0.5, seed=5)
    out = _run(frames, 2.0)
    a = metrics.displacement_to_reference(frames)
    b = metrics.displacement_to_reference(out)
    corr, _ = metrics.waveform_correlation(a, b)
    assert corr > 0.5, f"运动方向可能反转：位移波形相关={corr:.3f}"


def test_lk_estimator_measures_localized_motion_with_roi():
    """局部运动必须在**内缩窗口**上测，且 LK 必须收到 roi。

    回归价值：曾出现 LK 拿不到 roi 就在整帧上跟踪静止背景，把 0.5 px 的局部运动
    报成 0.0001 px，导致整个 local 档被误判为"未放大"。
    """
    S = 128
    frames, gt = synthetic.make_sequence(kind="local", T=60, H=S, W=S, fps=30.0,
                                         amp_px=0.5, freq=0.5, patch=S // 3, seed=8)
    roi = synthetic.measurement_roi(gt)
    assert roi is not None, "local 合成数据应带内缩测量窗口"
    w_roi = metrics.lk_displacement_to_reference(frames, ref="first", roi=roi)
    a_roi, _, _ = metrics.waveform_amplitude(w_roi)
    assert abs(a_roi - 0.5) < 0.1, f"内缩窗口下 LK 应测出 ≈0.5px，实际 {a_roi:.4f}"
    # 不传 roi（整帧）会被静止背景主导 → 明显偏小，这正是不传 roi 的危害
    w_full = metrics.lk_displacement_to_reference(frames, ref="first")
    a_full, _, _ = metrics.waveform_amplitude(w_full)
    assert a_full < 0.3 * a_roi, f"整帧测量本应被背景主导（{a_full:.4f} vs {a_roi:.4f}）"


def test_no_motion_stays_unchanged_for_all_alpha():
    frames, _ = synthetic.make_sequence(kind="static", T=36, H=96, W=96, fps=30.0,
                                        seed=6)
    for a in (0.0, 1.0, 5.0):
        out = _run(frames, a)
        err = np.abs(out - frames).max()
        assert err <= 1e-4, f"静止输入在 alpha={a} 被改变了：max|out-in|={err:.3e}"