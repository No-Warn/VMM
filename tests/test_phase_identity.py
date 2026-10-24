# -*- coding: utf-8 -*-
"""阶段 1 恒等性测试：alpha=0 时输出必须与输入一致（数值容差内）。

容差来自阶段 0 实测：pyrtools 复数可转向金字塔的重建误差约 3.8e-06，
因此 ``alpha=0`` 的合理容差取 1e-4（比重建误差高一个量级，留浮点累计余地）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vmm import synthetic                      # noqa: E402
from vmm.phase_magnifier import PhaseBasedMagnifier  # noqa: E402

TOL = 1e-4


def _mag(**kw):
    base = dict(fps=30.0, f_lo=0.3, f_hi=1.0, levels=3, order=3, channels="gray")
    base.update(kw)
    return PhaseBasedMagnifier(**base)


@pytest.mark.parametrize("kind", ["static", "shm_h"])
def test_alpha_zero_is_identity(kind):
    frames, _ = synthetic.make_sequence(kind=kind, T=36, H=96, W=96, fps=30.0,
                                        amp_px=0.5, freq=0.5, seed=1)
    out, _ = _mag().amplify(frames, alpha=0.0)
    err = np.abs(out - frames).max()
    assert err <= TOL, f"alpha=0 恒等失败（{kind}）：max|out-in|={err:.3e} > {TOL}"


def test_alpha_zero_identity_color_mode():
    frames, _ = synthetic.make_sequence(kind="shm_h", T=24, H=96, W=96, fps=30.0,
                                        amp_px=0.5, freq=0.5, seed=2)
    out, _ = _mag(channels="color", levels=3).amplify(frames, alpha=0.0)
    err = np.abs(out - frames).max()
    assert err <= TOL, f"彩色模式 alpha=0 恒等失败：max|out-in|={err:.3e}"


def test_gray_input_output_is_3channel():
    frames, _ = synthetic.make_sequence(kind="static", T=12, H=64, W=64, fps=30.0,
                                        seed=3)
    out, _ = _mag(levels=2).amplify(frames[:, :, :, 0], alpha=0.0)
    assert out.ndim == 4 and out.shape[3] == 3