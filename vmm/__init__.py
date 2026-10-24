# -*- coding: utf-8 -*-
"""连续视频**微运动放大**（阶段 1：相位式 Eulerian 基线）。

本包与 DiLA 的潜变量放大器（``models/amplification``）**完全独立**：

* 输入：连续视频 ``[T,H,W,C]`` + 用户指定的额外增益 ``alpha``
* 输出：等帧数、等分辨率、同时间顺序的放大视频
* 运动表示：复数可转向金字塔的**局部相位**（Phase-Based VMM，Wadhwa 类思路）
* 阶段 1 不含任何学习模块：``reliability``/``denoising`` 显式关闭

模块划分见 ``docs/implementation_audit.md`` §7。
"""

from vmm.phase_magnifier import PhaseBasedMagnifier
from vmm.phase_pyramid import ComplexSteerablePyramid
from vmm.temporal_filter import bandpass_filter, bandpass_time, validate_band
from vmm import metrics, synthetic, video_io

__all__ = [
    "PhaseBasedMagnifier",
    "ComplexSteerablePyramid",
    "bandpass_filter",
    "bandpass_time",
    "validate_band",
    "metrics",
    "synthetic",
    "video_io",
]