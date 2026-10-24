# -*- coding: utf-8 -*-
"""时间带通滤波（离线、零相位）与频带合法性校验。

设计要点（任务书 §3.4 的要求逐条对应）：
* 滤波沿**同一空间位置、同一尺度、同一方向**的时间序列进行 → 统一约定输入为
  ``[T, ...]``，``axis=0`` 为时间。
* 频率必须由**实际帧率**换算，且不得超过奈奎斯特频率；非法参数直接抛错，
  不允许静默取默认值。
* 使用 ``scipy.signal.butter`` + ``filtfilt``：零相位（无时间位移），
  **使用未来帧 → 仅适用于离线处理**，不得称为实时算法。
* 提供 ``bandpass_gain_at``：报告滤波器在目标频率处的增益 ``|H(f)|``。
  这一点对解释"实测增益为什么不是恰好 1+α"很关键（若 |H(f)|<1，实际增益
  会低于理论值），必须写进实验报告而不是含糊过去。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal


def validate_band(fps: float, f_lo: float, f_hi: float, order: int = 2,
                  min_cycles: float = 1.0, n_frames: int | None = None) -> dict:
    """校验频带参数；非法则 ``ValueError``。返回规范化后的参数与提示信息。"""
    if fps <= 0:
        raise ValueError(f"fps 必须为正，收到 {fps}")
    nyq = fps / 2.0
    if f_lo is None or f_hi is None:
        raise ValueError("f_lo/f_hi 不能为 None —— 必须显式给出目标运动频带")
    if not (0 < f_lo < f_hi):
        raise ValueError(f"要求 0 < f_lo < f_hi，收到 f_lo={f_lo}, f_hi={f_hi}")
    if f_hi >= nyq:
        raise ValueError(
            f"f_hi={f_hi} Hz 超过奈奎斯特频率 {nyq} Hz（fps={fps}）—— 必须降低 f_hi 或提高采样率"
        )
    if f_lo <= 0:
        raise ValueError("f_lo 必须 > 0（带通不能包含直流）")
    if int(order) < 1:
        raise ValueError("order 必须 >= 1")
    info = {
        "fps": float(fps), "f_lo": float(f_lo), "f_hi": float(f_hi),
        "nyquist": float(nyq), "order": int(order),
        "period_frames_lo": float(fps / f_lo), "period_frames_hi": float(fps / f_hi),
        "filter_type": "butter bandpass + filtfilt (零相位, 离线)",
    }
    if n_frames is not None:
        cycles = float(f_lo) * n_frames / float(fps)
        info["cycles_in_signal"] = cycles
        if cycles < min_cycles:
            raise ValueError(
                f"信号长度 {n_frames} 帧在 f_lo={f_lo}Hz 下只有 {cycles:.2f} 个周期，"
                f"少于 {min_cycles} —— 带通不可靠，请延长视频或降低 f_lo"
            )
    return info


def bandpass_filter(fps: float, f_lo: float, f_hi: float,
                    order: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """返回 ``(b, a)``；调用前建议先用 :func:`validate_band` 校验。"""
    nyq = fps / 2.0
    return signal.butter(int(order), [f_lo / nyq, f_hi / nyq], btype="bandpass")


def bandpass_time(x: np.ndarray, filt: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """沿 ``axis=0``（时间）做零相位带通；输入 ``[T, ...]``。"""
    b, a = filt
    return signal.filtfilt(b, a, x, axis=0)


def bandpass_gain_at(filt: Tuple[np.ndarray, np.ndarray], f: float,
                     fps: float, passes: int = 2) -> float:
    """``filtfilt`` 在频率 ``f`` 处的**有效**幅频响应。

    ⚠️ 注意阶数：``filtfilt`` 是双向滤波，幅度响应是单次的平方，即
    ``|H(f)|**passes``（相位被抵消）。这一点直接影响"理论增益"的写法：
    带通后相位被放大 α 倍，实际总增益 ≈ ``1 + α·|H(f)|²``，而不是 ``1 + α·|H(f)|``。
    （本项目实验里通带中心 ``|H|≈1.000``，两种写法数值相同；但定义必须写对。）
    """
    b, a = filt
    w = 2.0 * np.pi * float(f) / float(fps)
    _, h = signal.freqz(b, a, worN=[w])
    return float(np.abs(h[0]) ** int(passes))


def bandpass_gain_profile(filt: Tuple[np.ndarray, np.ndarray], fps: float,
                          n: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """返回 ``(freqs, gains)`` 便于画/记录通带形状（诊断用）。"""
    b, a = filt
    w, h = signal.freqz(b, a, worN=n)
    return w * fps / (2.0 * np.pi), np.abs(h)