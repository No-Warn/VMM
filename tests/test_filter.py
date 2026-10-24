# -*- coding: utf-8 -*-
"""时间带通滤波的测试（任务书 §3.4 要求逐条可验证）。

覆盖：非法频带必须报错（不允许静默取默认）、通带中心增益、直流与带外抑制、
**零相位**（filtfilt 前后波形不移位）、以及 ``filtfilt`` 的双向特性（幅度是单次的平方）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vmm.temporal_filter import (bandpass_filter, bandpass_gain_at,   # noqa: E402
                                 bandpass_gain_profile, bandpass_time,
                                 validate_band)


def test_validate_band_rejects_above_nyquist():
    with pytest.raises(ValueError):
        validate_band(fps=30.0, f_lo=0.3, f_hi=20.0)      # nyquist=15


def test_validate_band_rejects_zero_and_inverted():
    with pytest.raises(ValueError):
        validate_band(fps=30.0, f_lo=0.0, f_hi=1.0)
    with pytest.raises(ValueError):
        validate_band(fps=30.0, f_lo=2.0, f_hi=1.0)


def test_validate_band_rejects_none_and_bad_order():
    with pytest.raises(ValueError):
        validate_band(fps=30.0, f_lo=None, f_hi=1.0)
    with pytest.raises(ValueError):
        validate_band(fps=30.0, f_lo=0.3, f_hi=1.0, order=0)


def test_validate_band_requires_enough_cycles():
    # 10 帧 @30fps 在 0.3Hz 下只有 0.1 个周期 → 必须报错而不是硬算
    with pytest.raises(ValueError):
        validate_band(fps=30.0, f_lo=0.3, f_hi=1.0, n_frames=10)


def test_gain_at_center_is_near_one_and_squared_by_filtfilt():
    filt = bandpass_filter(30.0, 0.3, 1.0, order=2)
    g1 = bandpass_gain_at(filt, 0.5, 30.0, passes=1)
    g2 = bandpass_gain_at(filt, 0.5, 30.0, passes=2)
    assert 0.9 < g1 <= 1.0 + 1e-9
    assert abs(g2 - g1 ** 2) < 1e-12


def test_attenuates_dc_and_out_of_band():
    filt = bandpass_filter(30.0, 0.3, 1.0, order=2)
    g_dc = bandpass_gain_at(filt, 1e-6, 30.0, passes=2)
    g_hi = bandpass_gain_at(filt, 10.0, 30.0, passes=2)
    assert g_dc < 1e-6, f"直流未被抑制：{g_dc}"
    assert g_hi < 1e-3, f"带外未被抑制：{g_hi}"


def test_filtfilt_is_zero_phase():
    """零相位：通带内正弦经 filtfilt 后**滞后必须为 0**（用互相关滞后判定）。

    不用"峰值位置"判位移（在采样/幅值微小抖动下不稳健），改用中心段的互相关峰位置；
    幅值另测（filtfilt 的有效增益是 |H(f)|²，不等于 1，故不能要求逐点相等）。
    """
    fps, T, f = 30.0, 300, 0.5
    t = np.arange(T) / fps
    x = np.sin(2 * np.pi * f * t)
    filt = bandpass_filter(fps, 0.3, 1.0, order=2)
    y = bandpass_time(x[:, None], filt)[:, 0]

    seg = slice(50, T - 50)
    a = x[seg] - x[seg].mean()
    b = y[seg] - y[seg].mean()
    cc = np.correlate(b, a, mode="full")
    lag = int(np.argmax(cc) - (len(a) - 1))
    assert lag == 0, f"零相位滤波发生了 {lag} 个采样的滞后"

    # 幅值：有效增益应接近 |H(f)|²。注意两件事：
    #  ① filtfilt 双向 → 增益是 |H|²（不是 |H|）；
    #  ② 边界（padtype='odd' 的延拓）会产生约 8% 的瞬态超调，故取严格中段测。
    seg = slice(100, T - 100)
    ratio = float(np.abs(y[seg]).max() / np.abs(x[seg]).max())
    g2 = bandpass_gain_at(filt, f, fps, passes=2)
    assert 0.8 < ratio < 1.2, f"通带中心幅值异常：{ratio}"
    assert abs(ratio - g2) < 0.15, f"幅值与 |H(f)|²={g2:.3f} 不符：{ratio:.3f}"


def test_bandpass_profile_shape():
    filt = bandpass_filter(30.0, 0.3, 1.0, order=2)
    freqs, gains = bandpass_gain_profile(filt, 30.0, n=64)
    assert freqs.shape == gains.shape == (64,)
    assert gains.max() <= 1.0 + 1e-9


def test_zero_input_stays_exactly_zero():
    """纯静止序列（相位恒定）对应的相位差为 0，带通后必须严格为 0 —— 恒等的结构保证。"""
    filt = bandpass_filter(30.0, 0.3, 1.0, order=2)
    zeros = np.zeros((60, 8, 8))
    out = bandpass_time(zeros, filt)
    assert np.all(out == 0.0), "零信号的带通输出必须严格为零"