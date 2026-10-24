# -*- coding: utf-8 -*-
"""相位式 Eulerian 微运动放大（阶段 1 主实现）。

运动表示的数学定义（与任务书 §3.2 对齐）
----------------------------------------
对连续视频 ``I_t``，逐帧做复数可转向金字塔分解，得到各尺度 ``s``、各方向 ``o`` 的
复数系数

    C_{s,o}(x,t) = A_{s,o}(x,t) · exp( j·phi_{s,o}(x,t) )

* **无环绕的相位增量**（避免 2π 跳变）：``dphi_t = wrap( phi_t - phi_{t-1} )``，
  等价于任务书写的 ``arg(C_t · conj(C_{t-1}))``（因为 ``|C|`` 不影响幅角差）。
* **累积相位**：``Phi_t = sum_{tau<=t} dphi_tau``（即时间上解缠后的局部相位）。
* **零相位时间带通**：``Phi_bp = B_[f_l, f_h](Phi)``。
* **额外增益**（alpha 是**额外**的，不是总量）：``Phi_out = Phi + alpha · Phi_bp``。
* **重建系数**：``C' = C · exp( j · alpha · Phi_bp )``（幅值不变 → 纯几何运动）。
* ``alpha = 0`` ⇒ ``C' = C`` ⇒ 重建即输入（误差只有金字塔重建误差 ~4e-6）。

实现上的两个关键决定
--------------------
1. **只存局部相位、重建时重新分解**：第一遍逐帧只保留 ``phi``（float32），
   算出 ``beta = alpha·Phi_bp`` 后，第二遍再逐帧分解并把 ``exp(j·beta)`` 乘到
   **原始复数系数**上。这样 (a) 不存 ``|C|`` 也不做 exp/angle 往返，
   ``alpha=0`` 时是真·恒等（不是靠近似）；(b) 内存只与 ``phi`` 成正比
   （约 ``T·H·W·∑bands·4B``），彩色模式下还能逐通道流式处理。
2. **逐带相位上限** ``phase_clip``：相位平移过大（``|beta|→π``）会让系数失控，
   表现为纹理破坏/振铃。任务书 §3.3 要求"根据尺度设置合理的最大增益"，
   这里先提供全局 ``phase_clip`` 与可选 ``amp_floor``（低幅值系数限幅，
   阶段 2 的可靠性门控接口）。
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np

from vmm import video_io
from vmm.phase_pyramid import ComplexSteerablePyramid
from vmm.temporal_filter import (bandpass_filter, bandpass_gain_at,
                                 bandpass_time, validate_band)


def _wrap_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


class PhaseBasedMagnifier:
    """相位式微运动放大器（无学习参数，确定性算法）。

    Args:
        fps: 视频帧率（Hz）—— 频率换算的唯一依据。
        f_lo/f_hi: 目标微运动频带（Hz），必须显式给出且满足奈奎斯特条件。
        levels: 可转向金字塔层数。
        order: 方向数（pyrtools 的 ``order``）。
        max_gain: 顶点带宽增益上限（先小后大；``None`` 表示不限制）。
        channels: ``gray``（灰度处理）或 ``color``（用亮度求相位、三通道各自搬移，
            保持色彩与几何一致性）。
        filter_order: 时间带通滤波器阶数。
        phase_clip: 相位平移的逐带上限（弧度）；``None``（默认）表示不限制。
            阶段 1 默认不限，是为了**如实测出算法本身的增益**；真实视频建议显式
            给出上限（例如 1.2），否则大 alpha 会破坏纹理。自适应逐带上限属于阶段 2。
        amp_floor: 幅值门限比例（``A < amp_floor · band_median_A`` 的像素不放大，
            默认 0 关闭 —— 阶段 1 要如实测出未加门控的基线噪声行为）。
    """

    def __init__(self, fps: float, f_lo: float, f_hi: float, levels: int = 4,
                 order: int = 3, max_gain: Optional[float] = None,
                 channels: str = "gray", filter_order: int = 2,
                 phase_clip: Optional[float] = None,
                 amp_floor: float = 0.0,
                 phi_dtype: str = "float32",
                 reliability_cfg=None,
                 debug_maps: bool = False) -> None:
        self.band_info = validate_band(fps, f_lo, f_hi, filter_order)
        self.fps = float(fps)
        self.f_lo, self.f_hi = float(f_lo), float(f_hi)
        self.levels = int(levels)
        self.order = int(order)
        self.max_gain = max_gain
        self.channels = str(channels)
        if self.channels not in ("gray", "color"):
            raise ValueError("channels 只能是 gray 或 color")
        self.filter_order = int(filter_order)
        self.phase_clip = None if phase_clip is None else float(phase_clip)
        self.amp_floor = float(amp_floor)
        # ── 阶段 2：可靠性门控配置（默认 None ⇒ 与阶段 1 逐位一致）──
        # 说明（审计结论）：阶段 1 的 ``amp_floor`` 只被声明与存储、**从未参与计算**；
        # 阶段 2 起它被真正接入，含义 = 运动表示层的**幅值可信度软阈值**
        # （相对本带中位幅值的比例，无量纲，不会把低对比度真实运动清零）。
        from vmm.reliability import ReliabilityConfig
        if reliability_cfg is None:
            reliability_cfg = ReliabilityConfig()
        elif isinstance(reliability_cfg, dict):
            reliability_cfg = ReliabilityConfig(**reliability_cfg)
        if amp_floor and not reliability_cfg.amp_floor:
            reliability_cfg.amp_floor = float(amp_floor)     # 兼容阶段 1 的旧入口
        self.reliability_cfg = reliability_cfg.resolved()
        self.amp_floor = float(self.reliability_cfg.amp_floor)
        # Phase G：可选导出逐带权重图（用于 G1/G2/G3 的可分性/定位评估；默认关闭）
        self.debug_maps = bool(debug_maps)
        # 局部相位存储精度：长视频可用 float16 省一半内存（α=0 走不改系数的路径，
        # 恒等性不受影响；α>0 时 β 引入 ~1e-3·α 弧度的量化误差）
        self.phi_dtype = np.dtype(phi_dtype)
        if self.phi_dtype not in (np.dtype("float32"), np.dtype("float16")):
            raise ValueError("phi_dtype 只支持 float32 / float16")
        self.filt = bandpass_filter(fps, f_lo, f_hi, filter_order)

    # ------------------------------------------------------------------ #
    # 分块流式（长视频）
    # ------------------------------------------------------------------ #
    def required_margin(self) -> int:
        """零相位滤波在块边界需要丢弃的帧数。

        由**低截止频率**决定滤波器记忆长度：``margin = max(3·fps/f_lo, 30)`` 帧。
        例：fps=30、f_lo=0.3Hz → 300 帧（10 秒）；f_lo=2Hz → 45 帧。
        分块只在 ``T > chunk + 2·margin`` 时才有收益（否则边界开销吃掉全部节省）。
        """
        return int(max(np.ceil(3.0 * self.fps / max(self.f_lo, 1e-6)), 30))

    def amplify_chunked(self, video: np.ndarray, alpha: float,
                        chunk_frames: int = 120,
                        margin_frames: Optional[int] = None
                        ) -> Tuple[np.ndarray, Dict]:
        """overlap-save 分块放大：每块取 ``[s−margin, e+margin)`` 整段滤波，只保留中段。

        这样块内的零相位滤波拥有正确的"未来帧"上下文，**块边界不会产生接缝**；
        代价是 ``2·margin`` 的重复计算。
        """
        margin = int(margin_frames if margin_frames is not None
                     else self.required_margin())
        T = video.shape[0]
        need = chunk_frames + 2 * margin
        if T <= need:
            out, diag = self.amplify(video, alpha)
            diag["chunked"] = False
            diag["required_margin"] = margin
            diag["note"] = (f"全长 {T} 帧 ≤ chunk+2·margin = {need}，整段处理更省"
                            f"（分块无收益）")
            return out, diag
        out = np.empty_like(video, dtype=np.float64)
        bounds = []
        for s in range(0, T, chunk_frames):
            e = min(s + chunk_frames, T)
            a, b = max(0, s - margin), min(T, e + margin)
            seg_out, _ = self.amplify(video[a:b], alpha)
            out[s:e] = seg_out[s - a: e - a]
            bounds.append(e)
        return out, {"chunked": True, "chunk_frames": int(chunk_frames),
                     "margin_frames": margin, "required_margin": margin,
                     "chunk_boundaries": bounds[:-1], "T": int(T),
                     "alpha": float(alpha),
                     "note": "overlap-save：边界上下文完整，无接缝（用 seam_score 复核）"}

    # ------------------------------------------------------------------ #
    # 第一遍：只存局部相位
    # ------------------------------------------------------------------ #
    def _collect_phase(self, gray: np.ndarray, pyr: ComplexSteerablePyramid,
                       collect_amp: bool = False) -> Tuple[Dict, Dict]:
        T = gray.shape[0]
        phi: Dict = {}
        amp: Dict = {}
        for t in range(T):
            coeffs = pyr.analyze(gray[t])
            for k in pyr.complex_keys:
                c = np.asarray(coeffs[k])
                if k not in phi:
                    phi[k] = np.empty((T,) + c.shape, dtype=self.phi_dtype)
                    if collect_amp:
                        amp[k] = np.empty((T,) + c.shape, dtype=np.float32)
                phi[k][t] = np.angle(c)
                if collect_amp:
                    amp[k][t] = np.abs(c)
        return phi, {"T": T, "n_bands": len(pyr.complex_keys),
                     "amp": amp if collect_amp else None}

    # ------------------------------------------------------------------ #
    # 第二遍之前的中间量：逐带 beta（含阶段 2 可靠性门控与逐带增益上限）
    # ------------------------------------------------------------------ #
    def _compute_beta(self, phi: Dict, alpha: float, pyr: ComplexSteerablePyramid,
                      amp: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        from vmm.reliability import (amp_floor_weight, amplitude_weight,
                                     band_occupancy_normalized, band_occupancy_weight,
                                     coherence_weight, combine_weights,
                                     robust_temporal_smooth, smooth_weight,
                                     stability_weight)
        cfg = self.reliability_cfg

        def lookup(d, key):
            if not d:
                return None
            return d.get(key) if key in d else d.get(tuple(key))

        beta: Dict = {}
        diag: Dict = {"per_band": {}, "reliability": cfg.to_dict()}
        for k, ph32 in phi.items():
            ph = ph32.astype(np.float64)
            dphi = np.empty_like(ph)
            dphi[0] = 0.0
            dphi[1:] = _wrap_pi(ph[1:] - ph[:-1])
            Phi = np.cumsum(dphi, axis=0)
            Phi_bp = bandpass_time(Phi, self.filt)
            # §5 D2：稳健时间抑噪（非线性，作用于**运动表示**本身的相位变化）
            if cfg.temporal_robust and cfg.temporal_robust > 1:
                Phi_bp = robust_temporal_smooth(Phi_bp, cfg.temporal_robust)
            # ---- Phase G **平滑对照**族：直接改 Φ_bp（会改变传递函数，仅作对照）----
            if cfg.coherent_mode in ("mean", "median"):
                from vmm.coherent import window_stat
                Phi_bp = window_stat(Phi_bp, cfg.coherent_window, cfg.coherent_mode)
            # ---- §3 逐尺度逐方向增益上限：α_eff = min(α, β_{s,o}) ----
            cap = lookup(cfg.per_band_caps, k)
            a_eff = float(alpha) if cap is None else float(min(alpha, cap))
            # ---- 可靠性门控 w(x,t,s,o) ∈ [0,1] ----
            w = None
            w_stats: Dict[str, float] = {}
            if (cfg.mode in ("M1", "M2", "M3", "M4", "B1", "B2", "B3")
                    or cfg.coherent_mode in ("coherent", "coherent_rel", "adaptive")):
                parts = []
                sa = lookup(cfg.sigma_A, k)
                # 只有真的要用幅值证据时才强制要求 collect_amp（B2/B3 默认只用相位证据）
                if (sa is not None or cfg.amp_floor > 0) and amp is None:
                    raise RuntimeError(
                        "启用幅值证据需要逐带幅值：请用 collect_amp=True 收集")
                if sa is not None:
                    A = amp[k].astype(np.float64)
                    wa = amplitude_weight(A, sa, cfg.sigma_scale)
                    parts.append(wa)
                    w_stats["w_A_mean"] = float(wa.mean())
                if cfg.amp_floor > 0:
                    A = amp[k].astype(np.float64)
                    wf = amp_floor_weight(A, float(np.median(A)), cfg.amp_floor)
                    parts.append(wf)
                    w_stats["w_ampfloor_mean"] = float(wf.mean())
                if cfg.mode in ("M2", "M4"):
                    sp = lookup(cfg.sigma_phi, k)
                    if sp is not None:
                        ws = stability_weight(Phi_bp, sp, cfg.sigma_scale)
                        parts.append(ws)
                        w_stats["w_stab_mean"] = float(ws.mean())
                if cfg.mode in ("B1", "M4"):
                    # 长时窗目标带谱集中度（离线；用整段序列的频谱分布而非单帧幅度）
                    wb = band_occupancy_weight(Phi_bp, self.fps, self.f_lo, self.f_hi)
                    parts.append(wb)
                    w_stats["w_band_mean"] = float(wb.mean())
                if cfg.mode in ("B2", "B3"):
                    # [B2] scalar 归一化（消除 T 依赖、**保持选择性**）/
                    # [B3] 逐像素主导频点归一化（对偏频点/带边会过度压制）。
                    # domain="raw"：在**原始累积相位**上度量 —— 带通会抹掉带外信息，
                    # 实测判别余量从 0.20 降到 0.09（见 docs/stage2_results.md）。
                    ev = Phi_bp if cfg.band_domain == "bp" else Phi
                    wb = band_occupancy_normalized(
                        ev, self.fps, self.f_lo, self.f_hi, domain=cfg.band_domain,
                        mode=("scalar" if cfg.mode == "B2" else "pixel"))
                    parts.append(wb)
                    w_stats["w_band_mean"] = float(wb.mean())
                    w_stats["w_band_suppressed"] = float(np.mean(wb < 0.1))
                if cfg.mode == "M3":
                    wc = coherence_weight(np.exp(1j * dphi), cfg.coh_radius)
                    parts.append(wc)
                    w_stats["w_coh_mean"] = float(wc.mean())
                if cfg.coherent_mode in ("coherent", "coherent_rel", "adaptive"):
                    # Phase G：**只改可靠性证据、不改放大路径**（Φ_bp 原样进 β）。
                    # A = sqrt(window_mean(|analytic(Φ_bp)|²))；w = A²/(A²+σ_z²)，
                    # σ_z 必须在**独立**静止+噪声片上同管线标定（见 vmm/coherent.py）。
                    from vmm.coherent import coherent_amplitude, weight_from_amplitude
                    sz = lookup(cfg.coherent_sigma, k)
                    if sz is None:
                        raise RuntimeError(
                            "coherent_mode 需要逐带 σ_z：请用 vmm.coherent.coherent_noise_level "
                            "在独立静止+噪声片上标定后写入 cfg.coherent_sigma")
                    A_ce = coherent_amplitude(Phi_bp, cfg.coherent_window)
                    wg = weight_from_amplitude(A_ce, sz, cfg.coherent_form)
                    w_stats["w_ce_mean"] = float(wg.mean())
                    # Gate 的原始量是**未截断**的 A（w 会饱和到 1，不能用于混淆比）
                    if k == (0, 0) and self.debug_maps:
                        diag["a_map"] = np.asarray(A_ce.mean(axis=0), dtype=np.float32)
                    if cfg.coherent_mode == "adaptive":
                        # 回退：帧内相干证据整体不可靠（中位数 < τ）⇒ 用 B2 的谱集中度权重。
                        if float(np.median(wg)) < float(cfg.coherent_tau):
                            wg = band_occupancy_normalized(
                                Phi_bp, self.fps, self.f_lo, self.f_hi,
                                domain="bp", mode="scalar")
                            w_stats["coherent_fallback"] = 1.0
                        else:
                            w_stats["coherent_fallback"] = 0.0
                    parts.append(wg)
                if parts:
                    w = combine_weights(parts)
                    w = smooth_weight(w, cfg.spatial_smooth, cfg.temporal_smooth)
            b = (a_eff * Phi_bp) if w is None else (a_eff * w * Phi_bp)
            clipped = 0.0
            if self.phase_clip is not None:
                n_clip = float(np.mean(np.abs(b) > self.phase_clip))
                b = np.clip(b, -self.phase_clip, self.phase_clip)
                clipped = n_clip
            beta[k] = b.astype(np.float32)
            rec = {
                "shape": tuple(ph.shape[1:]),
                "alpha_eff": float(a_eff),
                "cap": (None if cap is None else float(cap)),
                "abs_bp_phase_mean": float(np.abs(Phi_bp).mean()),
                "abs_bp_phase_max": float(np.abs(Phi_bp).max()),
                "dphi_abs_mean": float(np.abs(dphi).mean()),
                "dphi_abs_p90": float(np.quantile(np.abs(dphi), 0.90)),
                "beta_max": float(np.abs(beta[k]).max()),
                "clipped_frac": clipped,
            }
            if w is not None:
                rec.update({"w_mean": float(w.mean()),
                            "w_median": float(np.median(w)),
                            "w_p10": float(np.quantile(w, 0.10)),
                            "w_suppressed_frac": float(np.mean(w < 0.1)),
                            **w_stats})
            if (w is not None and self.debug_maps and k == (0, 0)):
                # 供 Phase G 的 G1/G2/G3 评估：最细带的时间平均权重图。
                # ⚠️ B1/B2 的权重是**全片静态**的（谱比值，形状 [h,w]），
                # 只有相干证据是逐帧的 [T,h,w] —— 两种都要能导出。
                wm = w.mean(axis=0) if np.ndim(w) == 3 else w
                diag["w_map"] = np.asarray(wm, dtype=np.float32)
                diag["w_map_band"] = [0, 0]
                diag["w_map_is_temporal"] = bool(np.ndim(w) == 3)
            diag["per_band"][f"{k[0]}-{k[1]}"] = rec
        k0 = next(iter(beta))
        diag["edge_check"] = {
            "first_frame_beta_mean": float(np.abs(beta[k0][0]).mean()),
            "last_frame_beta_mean": float(np.abs(beta[k0][-1]).mean()),
            "mid_frame_beta_mean": float(np.abs(beta[k0][beta[k0].shape[0] // 2]).mean())}
        return beta, diag

    # ------------------------------------------------------------------ #
    # 第三遍：重建（alpha=0 时逐位恒等）
    # ------------------------------------------------------------------ #
    def _synthesize(self, frames_gray: np.ndarray, pyr: ComplexSteerablePyramid,
                    beta: Optional[Dict]) -> np.ndarray:
        T, H, W = frames_gray.shape
        out = np.empty((T, H, W), dtype=np.float64)
        for t in range(T):
            coeffs = pyr.analyze(frames_gray[t])
            if beta is not None:
                for k in pyr.complex_keys:
                    if k in beta:
                        coeffs[k] = np.asarray(coeffs[k]) * np.exp(1j * beta[k][t])
            out[t] = pyr.synthesize(coeffs)
        return out

    # ------------------------------------------------------------------ #
    def amplify(self, video: np.ndarray, alpha: float
                ) -> Tuple[np.ndarray, Dict]:
        """放大整段视频。

        Args:
            video: ``[T,H,W,3]``（或 ``[T,H,W]`` 灰度），float，值域 ``[0,1]``。
            alpha: **额外**运动增益（0 = 不放大；理论总量约 ``1+alpha``）。

        Returns:
            ``(out_video [T,H,W,3], diagnostics)``
        """
        if video.ndim not in (3, 4):
            raise ValueError(f"视频需要 [T,H,W] 或 [T,H,W,3]，收到 {video.shape}")
        t_start = time.time()
        rgb_input = video.ndim == 4
        gray = video_io.to_gray(video) if rgb_input else np.asarray(video, np.float64)
        T = gray.shape[0]
        # 分辨率相关校验（不允许静默默认）
        probe_pyr = ComplexSteerablePyramid(self.levels, self.order)
        shape_info = probe_pyr.validate(gray[0])

        # ---------- 阶段 A：亮度相位 ----------
        pyr_luma = probe_pyr
        # 只有启用门控时才收集逐带幅值（省一半内存；M0 与阶段 1 逐位一致）
        need_amp = bool(self.reliability_cfg.enabled
                        and (self.reliability_cfg.sigma_A
                             or self.reliability_cfg.amp_floor > 0)
                        and (self.reliability_cfg.mode in ("M1", "M2", "M3", "M4",
                                                          "B1", "B2", "B3")
                             or self.reliability_cfg.coherent_mode in ("coherent_rel",)))
        phi_luma, collect_info = self._collect_phase(gray, pyr_luma,
                                                    collect_amp=need_amp)
        # ---------- 阶段 B：beta ----------
        if alpha == 0.0:
            # alpha=0 时 beta 恒为 0：直接走"不修改系数"的路径，
            # 保证恒等性只受金字塔重建误差影响（不离散地乘 exp(0)）。
            beta_luma = None
            diag = {"note": "alpha=0：跳过相位修改，输出=输入（误差仅金字塔重建误差）"}
        else:
            beta_luma, diag = self._compute_beta(phi_luma, alpha, pyr_luma,
                                             amp=collect_info.get("amp"))
        del phi_luma
        diag.update({"alpha": float(alpha), "shape_info": shape_info,
                     "collect": collect_info, "channels": self.channels,
                     "band_info": self.band_info,
                     "filter_gain_at_f_mid": bandpass_gain_at(
                         self.filt, 0.5 * (self.f_lo + self.f_hi), self.fps),
                     "phase_clip": self.phase_clip,
                     "amp_floor": self.amp_floor})
        # ---------- 阶段 C：重建 ----------
        if self.channels == "gray" or not rgb_input:
            out_gray = self._synthesize(gray, probe_pyr, beta_luma)
            out = video_io.to_rgb_from_gray(out_gray)
        else:
            # 彩色：同一个 beta（来自亮度）作用于三通道各自的系数 → 几何一致的搬移
            out = np.empty_like(video, dtype=np.float64)
            for c in range(3):
                ch_gray = video[:, :, :, c]
                pyr_c = ComplexSteerablePyramid(self.levels, self.order)
                pyr_c.analyze(ch_gray[0])          # 固定重建实例
                if beta_luma is not None:
                    miss = set(beta_luma) - set(pyr_c.complex_keys)
                    if miss:
                        raise RuntimeError(f"通道 {c} 的金字塔结构与亮度不一致：{miss}")
                out[:, :, :, c] = self._synthesize(ch_gray, pyr_c, beta_luma)
        diag["elapsed_s"] = time.time() - t_start
        diag["seconds_per_frame"] = diag["elapsed_s"] / max(T, 1)
        return out, diag