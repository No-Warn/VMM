# -*- coding: utf-8 -*-
"""视频读写与一致性校验。

工程约束（来自阶段 0 审计）：
* 本机**没有 ffmpeg/ffprobe**，因此统一走 OpenCV（``cv2.VideoCapture`` /
  ``cv2.VideoWriter`` + ``mp4v``）。不要引入 ffmpeg 依赖。
* 帧统一为 ``float64``、值域 ``[0,1]``、形状 ``[T,H,W,C]``（C=3，RGB 顺序）。
  值域与顺序必须在读写处一次性固定，避免下游各模块各自假设。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoMeta:
    path: str
    width: int
    height: int
    fps: float
    n_frames: int


def video_meta(path: str | Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"打不开视频：{path}")
    meta = VideoMeta(
        path=str(path),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return meta


def read_video(
    path: str | Path,
    max_frames: Optional[int] = None,
    scale: float = 1.0,
    start: int = 0,
) -> Tuple[np.ndarray, VideoMeta]:
    """读取视频为 ``[T,H,W,3] float64``（RGB，值域 [0,1]）。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"打不开视频：{path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames: list[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= start:
            if scale != 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0)
            if max_frames is not None and len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    if not frames:
        raise IOError(f"读不到帧：{path}")
    arr = np.stack(frames, axis=0)
    meta = VideoMeta(path=str(path), width=arr.shape[2], height=arr.shape[1],
                     fps=fps, n_frames=arr.shape[0])
    return arr, meta


def write_video(frames: np.ndarray, path: str | Path, fps: float) -> None:
    """把 ``[T,H,W,3] float64 [0,1]`` 写成 mp4（mp4v）。"""
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(f"write_video 需要 [T,H,W,3]，收到 {frames.shape}")
    T, H, W, _ = frames.shape
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (W, H))
    if not vw.isOpened():
        raise IOError(f"VideoWriter 打不开：{path}")
    for t in range(T):
        img = (np.clip(frames[t], 0.0, 1.0) * 255.0).round().astype(np.uint8)
        vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    vw.release()


def write_side_by_side(orig: np.ndarray, amp: np.ndarray, path: str | Path,
                       fps: float, pad: int = 8) -> None:
    if orig.shape != amp.shape:
        raise ValueError("并排输出要求两者形状一致")
    T, H, W, _ = orig.shape
    sep = np.full((T, H, pad, 3), 0.5)
    write_video(np.concatenate([orig, sep, amp], axis=2), path, fps)


def to_gray(frames: np.ndarray) -> np.ndarray:
    """``[T,H,W,3] -> [T,H,W]``，按 Rec.709 亮度权重。"""
    if frames.ndim == 3:
        return frames.astype(np.float64, copy=False)
    w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    return (frames * w).sum(axis=3)


def to_rgb_from_gray(gray: np.ndarray) -> np.ndarray:
    return np.repeat(gray[:, :, :, None], 3, axis=3)


def assert_consistent(ref: VideoMeta, out: np.ndarray, out_fps: float,
                      tol_frames: int = 0) -> dict:
    """校验输出与输入在**帧数/分辨率/时间顺序**上一致；返回诊断字典。"""
    checks = {
        "n_frames_match": out.shape[0] == ref.n_frames - tol_frames,
        "height_match": out.shape[1] == ref.height,
        "width_match": out.shape[2] == ref.width,
        "channels_is_3": out.shape[3] == 3,
        "fps_match": abs(float(out_fps) - float(ref.fps)) < 1e-6,
        "dtype_ok": out.dtype == np.float64,
    }
    checks["all_ok"] = all(v for k, v in checks.items() if k != "all_ok")
    return checks