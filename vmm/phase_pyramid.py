# -*- coding: utf-8 -*-
"""复数可转向金字塔（pyrtools）封装：分解 / 系数遍历 / 重建。

为什么用 pyrtools 的 ``SteerablePyramidFreq``（阶段 0 实测，见
``docs/implementation_audit.md`` §6）：

* 它给出经典的 Simoncelli **复数可转向金字塔**，系数为复数 → 可直接取
  ``幅值 A`` 与``局部相位 phi``，这是相位式运动放大所需的最小充分表示；
* **重建自洽**：原始系数重建误差实测 ``max|recon-img| = 3.8e-06``，
  因此 ``alpha=0``（相位不变）时的输出可以做到与输入几乎逐位一致，
  这是阶段 1 的硬性验收项，不能靠"看起来一样"；
* 金字塔是**逐级降采样**的（最细层为全分辨率，之后逐级减半），
  所以每个带的相位都定义在各自的空间尺度上 —— 后续按带设置增益上限时要记住这一点。

接口设计上刻意只暴露三件事：``analyze`` / ``synthesize`` / ``complex_keys``，
避免把 pyrtools 的类型泄漏到算法代码里（后续若换成自研 FFT 版单演表示，
只需替换本文件）。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from pyrtools.pyramids import SteerablePyramidFreq

RESIDUAL_KEYS = ("residual_highpass", "residual_lowpass")


class ComplexSteerablePyramid:
    """复数可转向金字塔的薄封装（在同一分辨率下可复用实例重建）。"""

    def __init__(self, height: int = 4, order: int = 3, twidth: int = 1) -> None:
        if int(height) < 1:
            raise ValueError("height 必须 >= 1")
        if int(order) < 1:
            raise ValueError("order 必须 >= 1")
        self.height = int(height)
        self.order = int(order)
        self.twidth = int(twidth)
        self._recon_sp: SteerablePyramidFreq | None = None
        self._complex_keys: List[Tuple[int, int]] | None = None
        self._shape: Tuple[int, int] | None = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def max_height(image: np.ndarray, order: int = 3, twidth: int = 1) -> int:
        """该分辨率下允许的最大层数。

        直接**实测构造**来判定，而不是调用 ``pyrtools.pyramids.max_pyr_height``：
        后者在不同版本里签名不一致（1.0.11 需要额外的 ``filtsz`` 参数），
        依赖它会把版本细节泄漏进来。这里从 1 层往上试，构造失败即到顶。
        """
        h = 1
        while h < 12:
            try:
                SteerablePyramidFreq(np.asarray(image, dtype=np.float64),
                                     height=h + 1, order=order, twidth=twidth,
                                     is_complex=True)
            except Exception:
                break
            h += 1
        return h

    def validate(self, image: np.ndarray) -> dict:
        """分辨率相关的合法性与形状校验（任务书要求：不允许静默默认）。"""
        H, W = image.shape[:2]
        mh = self.max_height(image, self.order, self.twidth)
        if self.height > mh:
            raise ValueError(
                f"pyramid height={self.height} 超过该分辨率允许的最大值 {mh}（图 {H}x{W}）"
            )
        info = {"image_hw": (H, W), "height": self.height, "order": self.order,
                "max_height_allowed": mh, "twidth": self.twidth}
        return info

    # ------------------------------------------------------------------ #
    def analyze(self, image: np.ndarray) -> Dict:
        """分解单帧 ``[H,W]``，返回 pyrtools 的系数字典（含复数带与残差带）。

        第一次调用会保留一个实例用于后续重建（``synthesize`` 只替换系数，
        复用其滤波器），从而避免每帧都重建滤波器。
        """
        img = np.asarray(image, dtype=np.float64)
        if img.ndim != 2:
            raise ValueError(f"analyze 需要 [H,W] 灰度帧，收到 {img.shape}")
        sp = SteerablePyramidFreq(img, height=self.height, order=self.order,
                                  twidth=self.twidth, is_complex=True)
        if self._recon_sp is None:
            self._recon_sp = sp
            self._complex_keys = [k for k in sp.pyr_coeffs if isinstance(k, tuple)]
            self._shape = img.shape
        elif img.shape != self._shape:
            raise ValueError(
                f"同一实例只能处理同一分辨率：期望 {self._shape}，收到 {img.shape}"
            )
        return sp.pyr_coeffs

    # ------------------------------------------------------------------ #
    @property
    def complex_keys(self) -> List[Tuple[int, int]]:
        if self._complex_keys is None:
            raise RuntimeError("请先调用 analyze() 以确定金字塔结构")
        return list(self._complex_keys)

    @property
    def residual_keys(self) -> List[str]:
        return list(RESIDUAL_KEYS)

    def band_shape(self, key) -> Tuple[int, ...]:
        if self._recon_sp is None:
            raise RuntimeError("请先调用 analyze()")
        return tuple(np.asarray(self._recon_sp.pyr_coeffs[key]).shape)

    def synthesize(self, coeffs: Dict) -> np.ndarray:
        """由系数字典重建图像（``dict`` 必须含与 analyze 相同的键）。"""
        if self._recon_sp is None:
            raise RuntimeError("请先调用 analyze()")
        self._recon_sp.pyr_coeffs = dict(coeffs)
        return np.asarray(self._recon_sp.recon_pyr(), dtype=np.float64)

    # ------------------------------------------------------------------ #
    def band_amp_phase(self, coeffs: Dict, key) -> Tuple[np.ndarray, np.ndarray]:
        """取某个复数带的 ``(幅值 A, 局部相位 phi)``。"""
        c = np.asarray(coeffs[key])
        return np.abs(c), np.angle(c)

    def describe(self) -> dict:
        if self._recon_sp is None:
            raise RuntimeError("请先调用 analyze()")
        bands = {f"{k[0]}-{k[1]}": self.band_shape(k) for k in self.complex_keys}
        return {"height": self.height, "order": self.order, "twidth": self.twidth,
                "n_complex_bands": len(self.complex_keys), "band_shapes": bands}