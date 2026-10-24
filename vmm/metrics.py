# -*- coding: utf-8 -*-
"""评价指标：运动增益、分区时间能量、重建误差、伪影检查。

任务书 §5 的硬性要求（逐条对应）：

* **运动增益** ``G = A_out / A_in``：运动振幅必须**独立测量**（用相位相关，
  而不是读算法内部的相位），并且当输入振幅接近 0（无稳定分母）时**只报绝对误差**，
  不算比值；真实视频没有真值时必须写明"估计增益"及其不确定性。
* **分区时间能量**：活动区 ``M`` 与静止背景 ``1-M`` **分别**统计，
  不允许只报整幅视频的时间方差（真实运动与噪声会被混在一起）。
* **重建与伪影**：NaN/Inf/黑帧/饱和比例、``alpha=0`` 重建误差、时间闪烁。
* SSIM/PSNR 仅作补充，且必须说明参考图像含义（真实视频的原始帧**不是**
  放大结果的正确参考真值）。

⚠️ 关于"用哪种位移量"的重要说明（实测教训）
--------------------------------------------
本模块**不再**用"相邻帧位移序列的 (max-min)/2"当作运动振幅。原因是一个被实测
抓到的坑：对 30fps、0.5Hz、0.5px 的正弦微运动，**逐帧速度**只有
``2π·f·A/fps ≈ 0.052 px``，而相位相关在这种纹理上的**噪声底约 0.10 px** ——
信号被噪声淹没，输入振幅会被高估（实测 0.114 vs 真值 0.052），
于是"增益"被系统性低估（实测 5.28，而真值就是 ``1+alpha=11``）。

正确做法是测量**相对参考帧（时间平均帧）的位移波形** ``D(t)``：其振幅是
``A``（0.5 px），远高于噪声底，且波形本身可以与输入波形做相关来检验方向。
本模块因此提供 :func:`displacement_to_reference` 与 :func:`waveform_amplitude`，
逐帧速度序列仅作为诊断量保留。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# 独立运动估计（相位相关，亚像素）
# --------------------------------------------------------------------------- #
def _hann(shape: Tuple[int, int]) -> np.ndarray:
    wy = np.hanning(shape[0])[:, None]
    wx = np.hanning(shape[1])[None, :]
    return (wy * wx).astype(np.float32)


def phase_corr_disp(a: np.ndarray, b: np.ndarray,
                    window: bool = True) -> Tuple[float, float, float]:
    """相位相关估计 ``a -> b`` 的位移（像素，亚像素精度）。

    Returns: ``(dx, dy, response)``；``response`` 是峰值响应（低响应意味着不可信）。
    """
    a32 = np.ascontiguousarray(a, dtype=np.float32)
    b32 = np.ascontiguousarray(b, dtype=np.float32)
    if window:
        w = _hann(a32.shape)
        a32 = a32 * w
        b32 = b32 * w
    (dx, dy), resp = cv2.phaseCorrelate(a32, b32)
    return float(dx), float(dy), float(resp)


def _warp_shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """平移图像（亚像素，双线性），用于相位相关的"估计-回warp-再估计"。"""
    h, w = img.shape[:2]
    M = np.float32([[1.0, 0.0, -float(dx)], [0.0, 1.0, -float(dy)]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def phase_corr_disp_pyr(a: np.ndarray, b: np.ndarray, levels: int = 3,
                        window: bool = True) -> Tuple[float, float, float]:
    """**粗到细**相位相关（金字塔 + 预对齐），比单层更抗大位移与形变。

    单层相位相关在大位移/形变下峰值响应下降，会把"测不准"误报成"没放大"。
    这里在最粗层估计、逐层×2 上采样并预对齐后再细化残差，累加得到总位移。
    """
    A = [np.asarray(a, dtype=np.float32)]
    B = [np.asarray(b, dtype=np.float32)]
    for _ in range(int(levels) - 1):
        A.append(cv2.pyrDown(A[-1]))
        B.append(cv2.pyrDown(B[-1]))
    est = np.zeros(2, dtype=np.float64)
    resp = 1.0
    for lvl in range(len(A) - 1, -1, -1):
        if lvl < len(A) - 1:
            est *= 2.0
        Bl = B[lvl]
        if np.abs(est).max() > 1e-3:
            Bl = _warp_shift(Bl, est[0], est[1])
        dx, dy, resp = phase_corr_disp(A[lvl], Bl, window=window)
        est = est + np.array([dx, dy], dtype=np.float64)
    return float(est[0]), float(est[1]), float(resp)


def _gray(video: np.ndarray) -> np.ndarray:
    """转灰度。

    ⚠️ 必须与 :func:`vmm.video_io.to_gray` 用**同一套亮度权重**（Rec.709）。
    早期版本这里用的是"三通道简单平均"，而输出灰度是 Rec.709 亮度，
    导致真实视频 α=0（严格恒等）时能量比却报成 0.95 —— 度量定义不一致造成的假失配。
    """
    if video.ndim == 3:
        return video
    w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    return (video * w).sum(axis=3)


def displacement_series(video: np.ndarray,
                        roi: Optional[Tuple[slice, slice]] = None,
                        window: bool = True) -> np.ndarray:
    """逐帧**相邻**位移 ``[T-1, 3]``：``(dx, dy, response)``。

    ⚠️ 这是**速度**量（相邻帧差），对小振幅微运动会被噪声底淹没，
    只作诊断用；测振幅/增益请用 :func:`displacement_to_reference`。
    """
    g = _gray(video)
    out = []
    for t in range(1, g.shape[0]):
        a, b = g[t - 1], g[t]
        if roi is not None:
            a, b = a[roi], b[roi]
        out.append(phase_corr_disp(a, b, window=window))
    return np.asarray(out, dtype=np.float64)


def displacement_to_reference(video: np.ndarray,
                              roi: Optional[Tuple[slice, slice]] = None,
                              ref: str = "first",
                              window: bool = True,
                              method: str = "plain",
                              levels: int = 3) -> np.ndarray:
    """相对参考帧的位移波形 ``D(t)`` ``[T, 3]``：``(dx, dy, response)``。

    ``ref``：``'first'``（默认）用**首帧**，``'mean'`` 用时间平均帧。
    ⚠️ 实测：微运动序列的时间平均帧被运动模糊，参考帧纹理被抹掉后精度下降
    （干净序列波形相关 0.815，同样运动加噪后反而 0.986）。

    ``method``：``'plain'``（默认，单层相位相关）或 ``'pyr'``（粗到细）。

    ⚠️ **实测没有单一赢家**（见 ``docs/baseline_results.md`` §3）：
    干净带限序列上 ``plain`` 准确（0.4976 vs 真值 0.500）、``pyr`` 低估 3 倍；
    带噪序列上 ``pyr`` 更准（0.5508）、``plain/first`` 高估 25%。
    因此**不要只信一个估计器**：合成数据上应用真值甄别可信者，
    真实视频上必须把多估计器的分歧当作**不确定度**上报。
    """
    g = _gray(video)
    ref_frame = g.mean(axis=0) if ref == "mean" else g[0]
    if roi is not None:
        ref_frame = ref_frame[roi]
    fn = phase_corr_disp_pyr if method == "pyr" else phase_corr_disp
    out = []
    for t in range(g.shape[0]):
        cur = g[t][roi] if roi is not None else g[t]
        if method == "pyr":
            dx, dy, resp = fn(ref_frame, cur, levels=levels, window=window)
        else:
            dx, dy, resp = fn(ref_frame, cur, window=window)
        out.append((dx, dy, resp))
    return np.asarray(out, dtype=np.float64)


#: 合成验证里同时使用的估计器配置（method, ref）；真实视频也用同一组来报不确定度
ESTIMATOR_CONFIGS = (("plain", "mean"), ("plain", "first"), ("pyr", "mean"), ("pyr", "first"))


def measure_all_estimators(video: np.ndarray,
                           roi: Optional[Tuple[slice, slice]] = None,
                           fps: Optional[float] = None) -> Dict[str, Dict[str, float]]:
    """用全部配置（含 LK 与投影相位）测振幅，返回每个配置的 ``{amp_x, amp_y, resp}``。

    这是"输入/输出运动振幅均可独立测量"的实现方式：结果的分歧即**不确定度**。
    """
    res: Dict[str, Dict[str, float]] = {}
    waves = {}
    for meth, ref in ESTIMATOR_CONFIGS:
        w = displacement_to_reference(video, roi=roi, ref=ref, method=meth)
        ax, ay, resp = waveform_amplitude(w)
        key = f"pc_{meth}_{ref}"
        res[key] = {"amp_x": ax, "amp_y": ay, "resp": resp}
        waves[key] = w
    wl = lk_displacement_to_reference(video, ref="first", roi=roi)
    ax, ay, resp = waveform_amplitude(wl)
    res["lk_first"] = {"amp_x": ax, "amp_y": ay, "resp": resp}
    waves["lk_first"] = wl
    res["_waves"] = waves                                   # 供波形相关使用
    return res


def waveform_amplitude(series: np.ndarray, margin_frac: float = 0.1
                       ) -> Tuple[float, float, float]:
    """由**位移波形**估计振幅 ``(max-min)/2``（去掉首尾 ``margin_frac``）。"""
    if series.shape[0] == 0:
        return 0.0, 0.0, 0.0
    n = series.shape[0]
    m = int(round(n * margin_frac))
    core = series[m: n - m] if n - 2 * m >= 3 else series
    return (float(core[:, 0].max() - core[:, 0].min()) / 2.0,
            float(core[:, 1].max() - core[:, 1].min()) / 2.0,
            float(np.median(series[:, 2])))


def oscillation_amplitude(series: np.ndarray, margin_frac: float = 0.15
                          ) -> Tuple[float, float, float]:
    """**诊断用**：相邻帧速度序列的幅度（不要用来算增益，见模块 docstring）。"""
    return waveform_amplitude(series, margin_frac=margin_frac)


def waveform_correlation(a: np.ndarray, b: np.ndarray, margin_frac: float = 0.1
                         ) -> Tuple[float, float]:
    """两条位移波形的相关系数 ``(corr_x, corr_y)``；用于检验方向是否反转。"""
    n = min(a.shape[0], b.shape[0])
    m = int(round(n * margin_frac))
    idx = slice(m, n - m) if n - 2 * m >= 3 else slice(0, n)
    out = []
    for c in (0, 1):
        x, y = a[idx, c], b[idx, c]
        if x.std() < 1e-9 or y.std() < 1e-9:
            out.append(float("nan"))
        else:
            out.append(float(np.corrcoef(x, y)[0, 1]))
    return out[0], out[1]


def projection_phase_displacement(video: np.ndarray, axis: str = "x") -> np.ndarray:
    """投影傅里叶相位测位移。

    ⚠️ **实测在本项目的随机纹理上不可用**：平滑随机纹理的 1D 投影主频落在
    ``k0=1``，位移灵敏度为 ``2π·k0/N``，噪声完全主导（实测真值 0.5 px 时报
    0.002 px）。保留此函数是为了留下记录，不要用它做主估计器；
    真正独立于相位相关的估计器请用 :func:`lk_displacement_to_reference`。
    """
    g = _gray(video)
    proj = g.mean(axis=1) if axis == "x" else g.mean(axis=2)      # [T, N]
    ref = proj.mean(axis=0)
    spec = np.fft.rfft(ref)
    k0 = int(np.argmax(np.abs(spec[1:])) + 1)
    k0 = max(1, min(k0, proj.shape[1] // 2 - 1))
    ph = np.angle(np.fft.rfft(proj, axis=1)[:, k0])
    ph = np.unwrap(ph)
    d = -(ph - ph.mean()) / (2.0 * np.pi * k0)                    # 像素
    return d


def projection_waveform(video: np.ndarray) -> np.ndarray:
    """把 :func:`projection_phase_displacement` 包装成与波形接口一致的 ``[T,3]``。"""
    dx = projection_phase_displacement(video, "x")
    dy = projection_phase_displacement(video, "y")
    resp = np.ones_like(dx)
    return np.stack([dx, dy, resp], axis=1)


def lk_displacement_to_reference(video: np.ndarray, ref: str = "first",
                                 max_corners: int = 400,
                                 min_distance: int = 6,
                                 roi: Optional[Tuple[slice, slice]] = None
                                 ) -> np.ndarray:
    """**独立的第二估计器**：Lucas–Kanade 特征跟踪（空间域，与相位相关原理不同）。

    在参考帧上取角点，逐帧用金字塔 LK 跟踪到该帧，取**中位位移**作为该帧的
    ``D(t)``（相对参考帧）。返回 ``[T, 3]``：``(dx, dy, 有效点比例)``。

    ⚠️ ``ref`` 默认用**首帧**而不是时间平均帧：微运动序列的平均帧是被"运动模糊"
    过的，角点会被抹掉（实测对干净合成序列报出 0.0001 px 的假零位移）；
    首帧是清晰的。对整周期正弦（首帧位移=0），两者在数学上等价。

    ⚠️ ``roi`` 必须传：**局部运动**若不裁剪窗口，LK 会锚在占多数的静止背景上，
    报出 0.0001 px 的假零位移（实测）。窗口大小也会随 ROI 收缩，避免孔径问题。

    为什么要第二估计器：相位相关依赖全局 2D 峰值，当输出被放大到大幅形变时，
    它的峰值响应会下降，"测到的增益"可能反映的是**估计器失效**而不是算法饱和。
    LK 用局部特征点，可作为交叉验证。
    """
    g = _gray(video)
    # OpenCV 5.0 的 calcOpticalFlowPyrLK 只接受 8U（旧版可收 float32）
    g8 = (np.clip(g, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    T = g8.shape[0]
    ref_frame = g8.mean(axis=0).astype(np.uint8) if ref == "mean" else g8[0]
    if roi is not None:
        ref_frame = np.ascontiguousarray(ref_frame[roi])
    h, w = ref_frame.shape[:2]
    win = int(max(7, min(21, min(h, w) // 4)))
    pts = cv2.goodFeaturesToTrack(ref_frame, maxCorners=int(max_corners),
                                  qualityLevel=0.01, minDistance=int(min_distance))
    if pts is None or len(pts) < 4:
        return np.full((T, 3), np.nan)
    out = np.zeros((T, 3), dtype=np.float64)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    for t in range(T):
        cur = np.ascontiguousarray(g8[t][roi]) if roi is not None else g8[t]
        p1, st, _ = cv2.calcOpticalFlowPyrLK(ref_frame, cur, pts, None,
                                             winSize=(win, win), maxLevel=3,
                                             criteria=crit)
        if p1 is None:
            out[t] = np.nan
            continue
        ok = st.ravel() == 1
        if ok.sum() < 4:
            out[t] = np.nan
            continue
        d = (p1[ok] - pts[ok]).reshape(-1, 2)
        out[t, 0] = float(np.median(d[:, 0]))
        out[t, 1] = float(np.median(d[:, 1]))
        out[t, 2] = float(ok.mean())
    return out


# --------------------------------------------------------------------------- #
# 分区时间能量
# --------------------------------------------------------------------------- #
#: 输入时间能量低于此值时，**不允许**计算比值（任务书：接近零运动只报绝对误差）
ENERGY_FLOOR = 1e-5
def temporal_energy_map(video: np.ndarray) -> np.ndarray:
    """逐像素时间标准差 ``[H,W]``（灰度）。"""
    return _gray(video).std(axis=0)


def energy_ratio(video_in: np.ndarray, video_out: np.ndarray,
                 mask: np.ndarray) -> float:
    """``mask`` 区域内 输出/输入 的逐像素时间能量比。

    ⚠️ 输入能量接近零（例如完全静止的背景）时返回 ``nan``，
    调用方必须改报绝对误差 —— 否则会出现 1e8 这种由除零放大的假比值。
    """
    if mask is None or not mask.any():
        return float("nan")
    e_in = temporal_energy_map(video_in)[mask].mean()
    e_out = temporal_energy_map(video_out)[mask].mean()
    if e_in < ENERGY_FLOOR:
        return float("nan")
    return float(e_out / e_in)


def energy_absolute(video_in: np.ndarray, video_out: np.ndarray,
                    mask: np.ndarray) -> Tuple[float, float]:
    """绝对时间能量 ``(in, out)``：零运动区域的唯一合法报法。"""
    if mask is None or not mask.any():
        return float("nan"), float("nan")
    return (float(temporal_energy_map(video_in)[mask].mean()),
            float(temporal_energy_map(video_out)[mask].mean()))


def masks_from_motion(video: np.ndarray, active_pct: float = 90.0,
                      static_pct: float = 40.0, gt_mask: Optional[np.ndarray] = None
                      ) -> Dict[str, np.ndarray]:
    """由**输入视频**的时间能量定义活动区/静止区掩码（不用输出，避免自证）。"""
    e = temporal_energy_map(video)
    if gt_mask is not None and gt_mask.any():
        active = gt_mask.copy()
    else:
        active = e >= np.percentile(e, active_pct)
    static = e <= np.percentile(e, static_pct)
    return {"active": active, "static": static, "energy": e}


def region_energy(video: np.ndarray, mask: np.ndarray) -> float:
    if mask is None or not mask.any():
        return float("nan")
    return float(temporal_energy_map(video)[mask].mean())


def band_energy_map(video: np.ndarray, fps: float, f_lo: float, f_hi: float
                    ) -> np.ndarray:
    """目标频带内的时间能量（逐像素）：先用带通滤波再取时间 std。"""
    from vmm.temporal_filter import bandpass_filter, bandpass_time
    filt = bandpass_filter(fps, f_lo, f_hi, 2)
    return bandpass_time(_gray(video), filt).std(axis=0)


def band_energy_ratio(video_in: np.ndarray, video_out: np.ndarray, mask: np.ndarray,
                      fps: float, f_lo: float, f_hi: float) -> float:
    """目标频带内 输出/输入 的能量比（噪声是否被一起放大）。

    输入能量接近零时返回 ``nan``（见 :data:`ENERGY_FLOOR`），调用方改报绝对值。
    """
    if mask is None or not mask.any():
        return float("nan")
    bi = band_energy_map(video_in, fps, f_lo, f_hi)[mask].mean()
    bo = band_energy_map(video_out, fps, f_lo, f_hi)[mask].mean()
    if bi < ENERGY_FLOOR:
        return float("nan")
    return float(bo / bi)


def band_energy_absolute(video_in: np.ndarray, video_out: np.ndarray, mask: np.ndarray,
                         fps: float, f_lo: float, f_hi: float) -> Tuple[float, float]:
    if mask is None or not mask.any():
        return float("nan"), float("nan")
    return (float(band_energy_map(video_in, fps, f_lo, f_hi)[mask].mean()),
            float(band_energy_map(video_out, fps, f_lo, f_hi)[mask].mean()))


# --------------------------------------------------------------------------- #
# 重建误差 / 伪影 / 闪烁
# --------------------------------------------------------------------------- #
def recon_error(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
            "rmse": float(np.sqrt((d ** 2).mean()))}


def artifact_report(video: np.ndarray) -> Dict[str, float]:
    v = np.asarray(video, np.float64)
    black = float(np.mean([(np.abs(v[t]) < 1e-6).mean() > 0.99
                           for t in range(v.shape[0])]))
    sat = float(np.mean((v <= 0.0) | (v >= 1.0)))
    per_frame_mean = v.mean(axis=(1, 2, 3)) if v.ndim == 4 else v.mean(axis=(1, 2))
    return {
        "has_nan": float(np.isnan(v).any()),
        "has_inf": float(np.isinf(v).any()),
        "black_frame_frac": black,
        "saturation_frac": sat,
        "min": float(v.min()), "max": float(v.max()),
        "flicker_std": float(per_frame_mean.std()),
    }


def seam_score(video: np.ndarray, boundaries, radius: int = 1
               ) -> Tuple[float, float, float]:
    """块边界处帧间差分能量 vs 全片中位数。

    返回 ``(score, median_all, median_at_boundaries)``；``score`` 接近 1 说明**没有接缝**，
    明显 >2 则说明分块处理在边界留下了可见跳变（任务书 §9 要求检查分块接缝）。
    """
    gray = _gray(video)
    d = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2)) if gray.ndim == 3 else \
        np.abs(np.diff(gray, axis=0))
    med = float(np.median(d))
    vals = []
    for b in boundaries:
        if 0 < b < len(d):
            vals.append(d[max(0, b - radius):min(len(d), b + radius)])
    vv = np.concatenate(vals) if vals else np.array([np.nan])
    med_b = float(np.median(vv))
    return float(med_b / max(med, 1e-12)), med, med_b


def ring_and_far_masks(mask: np.ndarray, ring_px: int = 10,
                       far_px: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """由活动区掩码生成**边界环**与**远背景**掩码（任务书 §7.3 空间渗漏）。

    * ``ring``：距离活动区 ``≤ ring_px`` 但**不在**活动区内的像素（放大渗漏最直接的表现处）。
    * ``far``：距离活动区 ``> far_px`` 的像素（应当几乎不受影响）。

    ⚠️ 不允许用整幅均值代替空间选择性指标 —— 渗漏恰恰集中在环上，全图平均会把它稀释掉。
    """
    from scipy import ndimage as ndi
    m = np.asarray(mask, bool)
    inner = ndi.binary_dilation(m, iterations=int(ring_px))
    ring = inner & ~m
    far = ~ndi.binary_dilation(m, iterations=int(far_px))
    return ring, far


def psnr_ssim(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """补充指标；**必须**同时说明参考图像是什么（见调用处的 note）。"""
    from skimage.metrics import structural_similarity as ssim
    a8 = np.clip(a, 0, 1); b8 = np.clip(b, 0, 1)
    mse = float(np.mean((a8 - b8) ** 2))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))
    gray_a = _gray(a8); gray_b = _gray(b8)
    vals = [ssim(gray_a[t], gray_b[t], data_range=1.0)
            for t in range(gray_a.shape[0])]
    return {"psnr": psnr, "ssim_mean": float(np.mean(vals))}