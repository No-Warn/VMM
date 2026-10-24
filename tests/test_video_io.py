# -*- coding: utf-8 -*-
"""视频读写与一致性校验测试（帧数/分辨率/顺序/值域）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vmm import video_io   # noqa: E402


def _ramp_video(T=12, H=48, W=64):
    t = np.linspace(0, 1, T)[:, None, None, None]
    yy = np.linspace(0, 1, H)[None, :, None, None]
    xx = np.linspace(0, 1, W)[None, None, :, None]
    return np.repeat(0.3 + 0.4 * xx + 0.2 * yy + 0.1 * t, 3, axis=3)


def test_write_read_roundtrip(tmp_path):
    frames = _ramp_video()
    p = tmp_path / "rt.mp4"
    video_io.write_video(frames, p, fps=30.0)
    back, meta = video_io.read_video(p)
    assert back.shape[0] == frames.shape[0], "帧数不一致"
    assert (back.shape[1], back.shape[2]) == (frames.shape[1], frames.shape[2])
    assert back.dtype == np.float64
    assert 0.0 <= back.min() and back.max() <= 1.0
    assert abs(meta.fps - 30.0) < 1e-6


def test_consistency_checker_flags_mismatch():
    meta = video_io.VideoMeta("x.mp4", width=64, height=48, fps=30.0, n_frames=12)
    ok = video_io.assert_consistent(meta, _ramp_video(), 30.0)
    assert ok["all_ok"], ok
    bad = video_io.assert_consistent(meta, _ramp_video(T=10), 30.0)
    assert not bad["all_ok"] and not bad["n_frames_match"]


def test_to_gray_and_back():
    frames = _ramp_video(T=4, H=16, W=16)
    g = video_io.to_gray(frames)
    assert g.shape == (4, 16, 16)
    rgb = video_io.to_rgb_from_gray(g)
    assert rgb.shape == frames.shape