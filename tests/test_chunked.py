# -*- coding: utf-8 -*-
"""分块流式（overlap-save）测试：分块结果必须与整段一致，且**无接缝**。

任务书 §3.5：分块处理必须保证"滤波器的时间上下文与分块边界处理正确，
避免块与块之间产生接缝或闪烁"；§9 要求检查"明显的分块接缝"。
这里用**更高**的低截止频率让所需 margin 变小，从而能在测试里真正触发分块路径
（margin 与 1/f_lo 成正比：f_lo=2Hz@30fps → 45 帧；0.3Hz → 300 帧）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vmm import metrics, synthetic                 # noqa: E402
from vmm.phase_magnifier import PhaseBasedMagnifier  # noqa: E402


def _mag(**kw):
    base = dict(fps=30.0, f_lo=2.0, f_hi=4.0, levels=3, order=3, channels="gray")
    base.update(kw)
    return PhaseBasedMagnifier(**base)


def _video(T=420, H=96, W=96, freq=3.0, amp=0.5, seed=11):
    frames, gt = synthetic.make_sequence(kind="shm_h", T=T, H=H, W=W, fps=30.0,
                                         amp_px=amp, freq=freq, seed=seed)
    return frames, gt


def test_required_margin_scales_with_low_cutoff():
    m_hi = _mag(f_lo=2.0, f_hi=4.0).required_margin()
    m_lo = _mag(f_lo=0.3, f_hi=1.0).required_margin()
    assert m_hi == 45 and m_lo == 300, (m_hi, m_lo)


def test_chunked_matches_full_processing():
    """分块输出与整段输出必须几乎相同（差异只来自块边界的滤波上下文）。"""
    frames, _ = _video()
    mag = _mag()
    margin = mag.required_margin()
    full, d_full = mag.amplify(frames, alpha=5.0)
    chunked, d_ch = mag.amplify_chunked(frames, alpha=5.0, chunk_frames=100,
                                        margin_frames=margin)
    assert d_ch["chunked"] is True, "该配置应当真正走分块路径"
    diff = np.abs(full - chunked).max()
    assert diff < 0.02, f"分块与整段差异过大：{diff:.4f}"


def test_chunked_has_no_seam():
    frames, _ = _video()
    mag = _mag()
    chunked, d_ch = mag.amplify_chunked(frames, alpha=5.0, chunk_frames=100)
    score, med_all, med_b = metrics.seam_score(chunked, d_ch["chunk_boundaries"])
    assert score < 2.0, f"块边界出现接缝：score={score:.2f}（全片中位 {med_all:.2e}，边界 {med_b:.2e}）"


def test_chunked_identity_at_alpha_zero():
    frames, _ = _video()
    out, d = _mag().amplify_chunked(frames, alpha=0.0, chunk_frames=100)
    err = np.abs(out - frames).max()
    assert err <= 1e-4, f"分块下 α=0 不恒等：{err:.3e}"


def test_short_video_falls_back_to_full():
    frames, _ = _video(T=60)
    out, d = _mag().amplify_chunked(frames, alpha=5.0, chunk_frames=100)
    assert d["chunked"] is False and "整段处理" in d["note"]


def test_float16_phase_storage_still_identity():
    frames, _ = _video(T=60)
    out, _ = _mag(phi_dtype="float16").amplify(frames, alpha=0.0)
    assert np.abs(out - frames).max() <= 1e-4