#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 `w_band`（目标带谱集中度）的判别力与 T 依赖（阶段 2 未闭环 #1）。

背景（见 `docs/stage2_results.md` §10.3）：
    同一案例 `local_lowc`、α=5，T=64 时 B1 只损失 0.4% 增益，T=120 时损失 23%。
隔离实验已排除逐带上限与 D2，责任在 `w_band`。

本脚本**不改任何行为**，只做度量，回答三个问题：

1. `w_band` 在**真实目标信号**（纯带内正弦）、**带通噪声**（静止区实际看到的东西）、
   **白噪声**上分别取什么值？三者差多少 = 真实判别余量。
2. 余量随序列长度 T 如何变化？（T∈{32,64,120,240}）
3. 把 `w_band` 改在**带通之前**的原始累积相位 `Φ` 上算（信息未被带通破坏），
   判别余量是否显著更大？

输出：每个（案例/区域/层带）的集中度均值与**判别余量**（活动区 − 静止区），
供决定是否应采用"归一化 w_band"或"在 Φ 而非 Φ_bp 上度量"。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vmm import metrics, synthetic                     # noqa: E402
from vmm.phase_pyramid import ComplexSteerablePyramid  # noqa: E402
from vmm.temporal_filter import bandpass_filter, bandpass_time  # noqa: E402

CASES = {
    "shm_h": ("shm_h", 0.0, False),
    "local": ("local", 0.0, False),
    "local_lowc": ("local", 0.0, True),
    "local_noise": ("local", 0.02, False),
    "static": ("static", 0.0, False),
    "static_noise": ("static_noise", 0.02, False),
}


def concentration(x: np.ndarray, fps: float, f_lo: float, f_hi: float,
                  taper: bool = True) -> np.ndarray:
    """逐像素目标带能量占比（与 `reliability.band_occupancy_weight` 同一定义）。"""
    T = x.shape[0]
    y = x - x.mean(axis=0, keepdims=True)
    if taper:
        y = y * np.hanning(T)[:, None, None]
    F = np.fft.rfft(y, axis=0)
    freqs = np.fft.rfftfreq(T, d=1.0 / float(fps))
    band = (freqs >= float(f_lo)) & (freqs <= float(f_hi))
    p = np.abs(F) ** 2
    num = p[band].sum(axis=0)
    den = p[~band].sum(axis=0)
    return num / (num + den + 1e-12)


def band_phases(video: np.ndarray, levels: int, order: int):
    """逐带：原始累积相位 Φ 与带通后 Φ_bp（与 PhaseBasedMagnifier 完全同义）。"""
    gray = video if video.ndim == 3 else metrics._gray(video)
    T = gray.shape[0]
    pyr = ComplexSteerablePyramid(height=levels, order=order)
    pyr.validate(gray[0])
    phi = {}
    for t in range(T):
        coeffs = pyr.analyze(gray[t])
        for k in pyr.complex_keys:
            c = np.asarray(coeffs[k])
            if k not in phi:
                phi[k] = np.empty((T,) + c.shape, np.float32)
            phi[k][t] = np.angle(c)
    return gray, pyr, phi


def _region_means(x: np.ndarray, masks: dict) -> dict:
    out = {}
    for name, m in masks.items():
        if m is None or not m.any():
            out[name] = float("nan")
            continue
        out[name] = float(x[m].mean())
    return out


def reference_values(T: int, fps: float, f_lo: float, f_hi: float,
                     seed: int = 0) -> dict:
    """同一 T/带下，纯带内正弦 / 带通噪声 / 白噪声 的集中度（判别余量的两端）。"""
    freqs = np.fft.rfftfreq(T, d=1.0 / fps)
    idx = np.where((freqs >= f_lo) & (freqs <= f_hi))[0]
    t = np.arange(T) / fps
    tones = [float(concentration(
        (np.sin(2 * np.pi * freqs[i] * t))[:, None, None] * np.ones((1, 2, 2)),
        fps, f_lo, f_hi).mean()) for i in idx]
    rng = np.random.default_rng(seed)
    rw = np.cumsum(rng.standard_normal((T, 2, 2)), axis=0)      # 累积相位（随机游走）
    w = rng.standard_normal((T, 2, 2))
    filt = bandpass_filter(fps, f_lo, f_hi, 2)
    return {"n_bins": int(len(idx)),
            "c_null_white": float(((freqs >= f_lo) & (freqs <= f_hi)).mean()),
            "c_tone_best": float(max(tones)) if tones else float("nan"),
            "c_tone_worst": float(min(tones)) if tones else float("nan"),
            "c_tone_all": tones,
            "c_noise_white": float(concentration(w, fps, f_lo, f_hi).mean()),
            "c_noise_cumwalk": float(concentration(rw, fps, f_lo, f_hi).mean()),
            "c_noise_bp": float(concentration(bandpass_time(rw, filt), fps, f_lo, f_hi).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=["local_lowc", "local", "local_noise",
                                                  "shm_h", "static_noise"])
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--video", default="baby.mp4")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=240)
    ap.add_argument("--T", type=int, nargs="+", default=[64, 120])
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--freq", type=float, default=0.5)
    ap.add_argument("--amp-px", type=float, default=0.5)
    ap.add_argument("--f-lo", type=float, default=0.3)
    ap.add_argument("--f-hi", type=float, default=1.0)
    ap.add_argument("--real-f-lo", type=float, default=0.2)
    ap.add_argument("--real-f-hi", type=float, default=0.8)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    rep = {"config": vars(args), "runs": {}}

    def analyse(tag, video, fps, f_lo, f_hi, masks, extra):
        gray, pyr, phi = band_phases(video, args.levels, args.order)
        filt = bandpass_filter(fps, f_lo, f_hi, 2)
        T = gray.shape[0]
        ref = reference_values(T, fps, f_lo, f_hi)
        per_band = {}
        for k, ph in phi.items():
            p = ph.astype(np.float64)
            dphi = np.empty_like(p)
            dphi[0] = 0.0
            dphi[1:] = (p[1:] - p[:-1] + np.pi) % (2 * np.pi) - np.pi
            Phi = np.cumsum(dphi, axis=0)
            Phi_bp = bandpass_time(Phi, filt)
            c_bp = concentration(Phi_bp, fps, f_lo, f_hi)
            c_raw = concentration(Phi, fps, f_lo, f_hi)
            m = {n: (None if mm is None else mm) for n, mm in masks.items()}
            # 掩码要逐带重采样
            from vmm.reliability import _mask_for
            mk = {n: (_mask_for(v, Phi.shape[1:]) if v is not None else None) for n, v in m.items()}
            rec = {"conc_bp": _region_means(c_bp, mk), "conc_raw": _region_means(c_raw, mk),
                   "shape": list(Phi.shape[1:])}
            if "active" in mk and mk["active"] is not None and "static" in mk \
                    and mk["static"] is not None:
                a_bp = float(c_bp[mk["active"]].mean())
                s_bp = float(c_bp[mk["static"]].mean())
                a_raw = float(c_raw[mk["active"]].mean())
                s_raw = float(c_raw[mk["static"]].mean())
                rec["margin_bp"] = a_bp - s_bp
                rec["margin_raw"] = a_raw - s_raw
            per_band[f"{k[0]}-{k[1]}"] = rec
        pooled = {}
        for key in ("conc_bp", "conc_raw"):
            pooled[key] = {n: float(np.nanmean([per_band[b][key][n] for b in per_band]))
                           for n in masks}
        rep["runs"][tag] = {"T": int(T), "fps": float(fps), "band": [f_lo, f_hi],
                            "reference": ref, "pooled": pooled,
                            "margins": {"bp": float(np.nanmean([per_band[b].get("margin_bp", np.nan) for b in per_band])),
                                        "raw": float(np.nanmean([per_band[b].get("margin_raw", np.nan) for b in per_band]))},
                            "per_band": per_band, **extra}
        print(f"\n[{tag}] T={T} band={f_lo}-{f_hi}Hz  ref: tone_best={ref['c_tone_best']:.3f} "
              f"edge={ref['c_tone_worst']:.3f} bp_noise={ref['c_noise_bp']:.3f} "
              f"white={ref['c_noise_white']:.3f} (nbin={ref['n_bins']})")
        for key in ("conc_bp", "conc_raw"):
            print("   %-9s " % key + "  ".join(
                f"{n}={v:.3f}" for n, v in pooled[key].items() if np.isfinite(v)))
        print(f"   margin(bp)={rep['runs'][tag]['margins']['bp']:+.3f}  "
              f"margin(raw)={rep['runs'][tag]['margins']['raw']:+.3f}")

    if args.real:
        from vmm import video_io
        video, meta = video_io.read_video(args.video, max_frames=args.max_frames,
                                          scale=args.scale)
        m = metrics.masks_from_motion(video, active_pct=90, static_pct=40)
        analyse("real", video, meta.fps, args.real_f_lo, args.real_f_hi,
                {"active": m["active"], "static": m["static"]}, {})
    else:
        for case in args.cases:
            kind, noise, lowc = CASES[case]
            for T in args.T:
                fr, gt = synthetic.make_sequence(kind=kind, T=T, H=args.size, W=args.size,
                                                 fps=args.fps, amp_px=args.amp_px,
                                                 freq=args.freq, noise_sigma=noise,
                                                 patch=args.size // 3, seed=42)
                if lowc:
                    fr = np.clip(0.5 + 0.15 * (fr - 0.5), 0, 1)
                active = gt["active_mask"]
                masks = {"active": active if active.any() else None,
                         "static": ~active if active.any() else np.ones_like(active)}
                if active.any():
                    ring, far = metrics.ring_and_far_masks(active, ring_px=10, far_px=30)
                    masks["ring"], masks["far"] = ring, far
                analyse(f"{case}_T{T}", fr, args.fps, args.f_lo, args.f_hi, masks,
                        {"case": case, "noise_sigma": noise, "low_contrast": lowc})

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(rep, ensure_ascii=False, indent=2,
                                              default=str), encoding="utf-8")
    print(f"\n[save] {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
