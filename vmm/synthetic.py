# -*- coding: utf-8 -*-
"""合成测试视频：带**已知真值位移曲线**的微运动序列。

任务书 §4.1 要求：合成数据必须保存真实位移曲线，作为定量评估的参考。
本模块的每个序列都返回 ``gt`` 字典，含 ``dx``/``dy``（逐帧真值位移，像素）、
``amp_px``、``freq``、``kind``，以及可用于分区能量统计的 ``active_mask``。

四类输入（与任务书一致）：
* ``static``       —— 完全静止（无运动、无噪声）：要求输出=输入
* ``static_noise`` —— 静止 + 可控噪声：测"噪声是否被误放大"
* ``shm_h``/``shm_v`` —— 已知频率与振幅的**亚像素**水平/垂直正弦位移
* ``local``        —— 局部运动（小方块）与静止背景共存，用于测空间选择性
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage


def texture_image(H: int = 256, W: int = 256, seed: int = 0,
                  blur: float = 2.0, contrast: float = 0.5) -> np.ndarray:
    """生成带 2D 结构的纹理底图（避免 1D 条纹造成相位相关的孔径问题）。"""
    rng = np.random.default_rng(seed)
    img = rng.random((H, W))
    img = ndimage.gaussian_filter(img, blur)
    img = (img - img.min()) / (np.ptp(img) + 1e-12)
    # 叠一点几何结构（方块/圆），提高运动的可观测性
    yy, xx = np.mgrid[0:H, 0:W]
    for _ in range(6):
        cx, cy = rng.integers(0, W), rng.integers(0, H)
        r = rng.integers(6, max(8, min(H, W) // 8))
        m = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
        img[m] += 0.35
    img = (img - img.min()) / (np.ptp(img) + 1e-12)
    return 0.5 + contrast * (img - 0.5)


def _shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    return ndimage.shift(img, shift=(dy, dx), order=3, mode="reflect")


def make_sequence(
    kind: str = "shm_h",
    T: int = 120,
    H: int = 256,
    W: int = 256,
    fps: float = 30.0,
    amp_px: float = 0.5,
    freq: float = 0.5,
    noise_sigma: float = 0.0,
    patch: int = 64,
    seed: int = 0,
    noise_seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """生成合成序列。

    Returns:
        ``frames [T,H,W,3] float64``（值域 clamp 到 [0,1]）, ``gt`` 字典
    """
    base = texture_image(H, W, seed=seed)
    # 噪声用**独立**随机流：``noise_seed`` 与 ``seed`` 分开，就能在**同一底图**上
    # 生成不同噪声实现 —— 阶段 2 的 σ_A 标定必须这样做，否则幅值尺度会随底图变化
    # （实测：底图不同会让 M1 幅值门控完全失效，而相位证据 σ_φ 不受影响）。
    rng = np.random.default_rng((seed + 12345) if noise_seed is None else int(noise_seed))
    t = np.arange(T, dtype=np.float64) / float(fps)
    kinds = {"static", "static_noise", "shm_h", "shm_v", "local"}
    if kind not in kinds:
        raise ValueError(f"kind 必须是 {sorted(kinds)}，收到 {kind}")

    if kind in ("static", "static_noise"):
        dx = np.zeros(T); dy = np.zeros(T)
    elif kind == "shm_h":
        dx = amp_px * np.sin(2 * np.pi * freq * t); dy = np.zeros(T)
    elif kind == "shm_v":
        dy = amp_px * np.sin(2 * np.pi * freq * t); dx = np.zeros(T)
    else:  # local
        dx = amp_px * np.sin(2 * np.pi * freq * t); dy = np.zeros(T)

    frames = np.empty((T, H, W, 3), dtype=np.float64)
    active = np.zeros((H, W), dtype=bool)
    roi: Optional[Tuple[slice, slice]] = None
    roi_core: Optional[Tuple[slice, slice]] = None
    if kind == "local":
        y0 = (H - patch) // 2
        x0 = (W - patch) // 2
        roi = (slice(y0, y0 + patch), slice(x0, x0 + patch))
        active[roi] = True
        # ⚠️ 测量窗口必须比 patch **内缩一圈**：patch 是"从平移后的底图裁出的窗口"，
        # 其裁切边界在窗口内是**不动**的（LK 会锚在边界上、报出假零位移；
        # 相位相关读到的则是"内部运动 + 边界静止"的混合，振幅偏大 49%）。
        m = max(4, patch // 8)
        roi_core = (slice(y0 + m, y0 + patch - m), slice(x0 + m, x0 + patch - m))

    for i in range(T):
        if kind == "local":
            frame = base.copy()
            frame[roi] = _shift(base, dx[i], dy[i])[roi]
        else:
            frame = _shift(base, dx[i], dy[i])
            if kind in ("shm_h", "shm_v"):
                active[:] = True
        if noise_sigma > 0:
            frame = frame + rng.normal(0.0, noise_sigma, size=frame.shape)
        frames[i] = np.clip(np.repeat(frame[:, :, None], 3, axis=2), 0.0, 1.0)

    # 局部运动的"活动掩码"用真值给出；全局运动时用整幅
    if kind in ("static", "static_noise"):
        active[:] = False
    gt = {
        "kind": kind, "T": T, "H": H, "W": W, "fps": float(fps),
        "amp_px": float(amp_px), "freq": float(freq),
        "noise_sigma": float(noise_sigma),
        "dx": dx, "dy": dy, "active_mask": active, "roi": roi,
        "roi_core": roi_core,
        "has_motion": kind in ("shm_h", "shm_v", "local"),
        "note": "真值位移曲线 dx/dy（像素）由本函数生成时直接记录，未经过任何估计",
    }
    return frames, gt


def measurement_roi(gt: Dict):
    """**测量**该用的窗口：有内缩窗口就用它（避开裁切边界的伪静止/混合）。"""
    return gt.get("roi_core") or gt.get("roi")


def gt_amplitude(gt: Dict) -> Tuple[float, float]:
    """真值位移的振幅（(max-min)/2），单位像素。"""
    return (float(gt["dx"].max() - gt["dx"].min()) / 2.0,
            float(gt["dy"].max() - gt["dy"].min()) / 2.0)