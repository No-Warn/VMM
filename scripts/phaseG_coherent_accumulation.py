#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase G：多帧相干积累的消融与 Gate 判定。

方法（任务书 §6）：
    G0  B2 基线（阶段 2 推荐配置）
    G1  固定窗口**均值**（作用在 Φ_bp 上 —— 平滑对照）
    G2  稳健窗口**中位数**（同上，非线性）
    G3  **相干证据**：A=sqrt(window_mean(|analytic(Φ_bp)|²))，w=A²/(A²+σ_z²)
        —— 放大路径一行不改，只改可靠性权重
    G4  G3 + 幅值证据 w_A（需阶段 2 的 σ_A 标定）
    G5  自适应：帧内 w 中位数 < τ 时回退到 B2 的归一化谱集中度

统一条件（任务书 §3）：192×192、T=120、fps=30、f=0.5Hz、带 0.3–1.0Hz、
levels=4、order=3、channels=gray；噪声 = 独立高斯，随机种子显式记录。

用法::

    PY=/home/user/.conda/envs/dila/bin/python
    $PY scripts/phaseG_coherent_accumulation.py --stage baseline \
        --out-json outputs/stage2_coherent/baseline_b2.json
    $PY scripts/phaseG_coherent_accumulation.py --stage ablate \
        --out-json outputs/stage2_coherent/ablation.json
    $PY scripts/phaseG_coherent_accumulation.py --stage gates \
        --ablation-json outputs/stage2_coherent/ablation.json \
        --out-json outputs/stage2_coherent/gates.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vmm import metrics, synthetic                      # noqa: E402
from vmm.coherent import coherent_noise_level           # noqa: E402
from vmm.phase_magnifier import PhaseBasedMagnifier      # noqa: E402
from vmm.phase_pyramid import ComplexSteerablePyramid    # noqa: E402
from vmm.reliability import ReliabilityConfig, calibrate_noise  # noqa: E402
from vmm.temporal_filter import bandpass_filter, bandpass_time  # noqa: E402

# 预注册 Gate（与技术方案无关，跑实验前写死；对应 Phase E 的 G1/G2/G3）
GATES = {"G1_motion_over_static": 2.0, "G2_localization_contrast": 1.5,
         "G3_confounder_over_motion": 1.0}


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def make_confounder(case: str, base: np.ndarray, T: int, fps: float, freq: float,
                    patch: int = 64) -> tuple[np.ndarray, dict]:
    """亮度变化 / 局部纹理变化：**无几何运动**（Phase D 同类干扰项，改为 VMM 分辨率）。"""
    H, W = base.shape
    t = np.arange(T) / float(fps)
    frames = np.empty((T, H, W, 3), np.float64)
    active = np.zeros((H, W), bool)
    if case == "brightness":
        b = 1.0 + 0.03 * np.sin(2 * np.pi * freq * t)
        for i in range(T):
            frames[i] = np.repeat(np.clip(base * b[i], 0, 1)[:, :, None], 3, axis=2)
    elif case == "texture_change":
        y0, x0 = (H - patch) // 2, (W - patch) // 2
        active[y0:y0 + patch, x0:x0 + patch] = True
        rng = np.random.default_rng(7)
        other = rng.random((H, W))
        from scipy import ndimage
        other = ndimage.gaussian_filter(other, 1.5)
        other = 0.5 + 0.5 * (other - other.min()) / (np.ptp(other) + 1e-12)
        w2 = active.astype(float)
        w2 = ndimage.gaussian_filter(w2, patch // 8)
        w2 /= max(w2.max(), 1e-12)
        alpha = 0.5 * (1 - np.cos(2 * np.pi * freq * t))
        for i in range(T):
            f = base * (1 - w2) + w2 * ((1 - alpha[i]) * base + alpha[i] * other)
            frames[i] = np.repeat(np.clip(f, 0, 1)[:, :, None], 3, axis=2)
    else:
        raise ValueError(case)
    return frames, {"case": case, "has_motion": False, "active_mask": active,
                    "dx": np.zeros(T), "dy": np.zeros(T)}


def build_case(case: str, T: int, size: int, fps: float, freq: float, amp: float,
               noise: float, seed: int, base_seed: int = 42):
    if case in ("brightness", "texture_change"):
        base = synthetic.texture_image(size, size, seed=base_seed)
        frames, gt = make_confounder(case, base, T, fps, freq)
        gt["amp_px"] = 0.0
        if noise > 0:
            rng = np.random.default_rng(seed + 500)
            frames = np.clip(frames + rng.normal(0, noise, frames.shape), 0, 1)
        return frames, gt
    kind = "local" if case == "local_lowc" else case
    fr, gt = synthetic.make_sequence(kind=kind, T=T, H=size, W=size, fps=fps,
                                     amp_px=amp, freq=freq, noise_sigma=noise,
                                     patch=size // 3, seed=base_seed, noise_seed=seed + 100)
    if case == "local_lowc":
        fr = np.clip(0.5 + 0.15 * (fr - 0.5), 0, 1)
    gt["has_motion"] = kind in ("shm_h", "shm_v", "local")
    return fr, gt


# --------------------------------------------------------------------------- #
# σ_z 标定（独立静止+噪声片，同管线）
# --------------------------------------------------------------------------- #
def calibrate_sigma_z(w: int, T: int, size: int, fps: float, f_lo: float, f_hi: float,
                      noise: float, levels: int, order: int) -> dict:
    fr, _ = synthetic.make_sequence(kind="static_noise", T=T, H=size, W=size, fps=fps,
                                    noise_sigma=noise, seed=42, noise_seed=99991 + w)
    gray = metrics._gray(fr)
    pyr = ComplexSteerablePyramid(levels, order)
    pyr.validate(gray[0])
    filt = bandpass_filter(fps, f_lo, f_hi, 2)
    phi = {}
    for t in range(T):
        c = pyr.analyze(gray[t])
        for k in pyr.complex_keys:
            if k not in phi:
                phi[k] = np.empty((T,) + np.asarray(c[k]).shape, np.float64)
            phi[k][t] = np.angle(np.asarray(c[k]))
    out = {}
    for k, ph in phi.items():
        d = np.empty_like(ph)
        d[0] = 0.0
        d[1:] = (ph[1:] - ph[:-1] + np.pi) % (2 * np.pi) - np.pi
        Phi_bp = bandpass_time(np.cumsum(d, axis=0), filt)
        out[k] = coherent_noise_level(Phi_bp, w)
    return out


# --------------------------------------------------------------------------- #
# 运行与度量
# --------------------------------------------------------------------------- #
def region_metrics(video_in, video_out, fps, f_lo, f_hi, masks):
    out = {}
    b_in = metrics.band_energy_map(video_in, fps, f_lo, f_hi)
    b_out = metrics.band_energy_map(video_out, fps, f_lo, f_hi)
    for name, m in masks.items():
        if m is None or not m.any():
            out[name] = float("nan")
            continue
        i, o = float(b_in[m].mean()), float(b_out[m].mean())
        out[name] = (o / i) if i > metrics.ENERGY_FLOOR else float("nan")
    return out


def wmap_stats(wmap: np.ndarray, active: np.ndarray) -> dict:
    """G1/G2 的原始量：活动区权重、静止区权重、定位对比与 IoU。"""
    nan = float("nan")
    if wmap is None:
        return {"w_active": nan, "w_static": nan, "contrast_active_bg": nan,
                "iou": nan, "leak_far": nan}
    wmap = np.asarray(wmap, dtype=np.float64)
    if wmap.ndim == 3:                       # 防御：万一拿到 [T,h,w]
        wmap = wmap.mean(axis=0)
    if wmap.ndim != 2:
        return {"w_active": nan, "w_static": nan, "contrast_active_bg": nan,
                "iou": nan, "leak_far": nan, "note": f"bad_wmap_ndim={wmap.ndim}"}
    h, w = wmap.shape
    if active is None or not np.asarray(active).any():
        return {"w_active": nan, "w_static": float(wmap.mean()),
                "contrast_active_bg": nan, "iou": nan, "leak_far": nan}
    active = np.asarray(active)
    if active.shape != wmap.shape:
        import cv2
        active = cv2.resize(active.astype(np.uint8), (w, h),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
    a = float(wmap[active].mean())
    s = float(wmap[~active].mean()) if (~active).any() else nan
    k = int(active.sum())
    flat = wmap.ravel()
    th = np.partition(flat, flat.size - k)[flat.size - k]
    top = wmap >= th
    iou = float((top & active).sum() / max((top | active).sum(), 1))
    ring, far = metrics.ring_and_far_masks(active, ring_px=10, far_px=30)
    leak = float(wmap[far].mean() / max(a, 1e-12)) if far.any() else nan
    return {"w_active": a, "w_static": s, "contrast_active_bg": a / max(s, 1e-12),
            "iou": iou, "leak_far": leak}


def build_cfg(method: str, W: int, sigma_z: dict, sigma_A: dict, tau: float) -> ReliabilityConfig:
    cfg = ReliabilityConfig(mode="M0")
    if method == "G0":
        cfg.mode = "B2"
    elif method in ("G1", "G2"):
        cfg.coherent_mode = "mean" if method == "G1" else "median"
        cfg.coherent_window = W
    elif method == "G3":
        cfg.coherent_mode = "coherent"
        cfg.coherent_window = W
        cfg.coherent_sigma = sigma_z
    elif method == "G4":
        cfg.coherent_mode = "coherent_rel"
        cfg.coherent_window = W
        cfg.coherent_sigma = sigma_z
        cfg.sigma_A = sigma_A
    elif method == "G5":
        cfg.coherent_mode = "adaptive"
        cfg.coherent_window = W
        cfg.coherent_sigma = sigma_z
        cfg.coherent_tau = tau
    else:
        raise ValueError(method)
    return cfg


def run_one(frames, gt, args, cfg, alpha: float):
    mag = PhaseBasedMagnifier(fps=args.fps, f_lo=args.f_lo, f_hi=args.f_hi,
                              levels=args.levels, order=args.order, channels="gray",
                              reliability_cfg=cfg, debug_maps=True)
    t0 = time.time()
    out, diag = mag.amplify(frames, alpha)
    dt = time.time() - t0
    roi = synthetic.measurement_roi(gt)
    rec = {"alpha": alpha, "seconds": dt,
           "identity_err_max": metrics.recon_error(frames, out)["max_abs"],
           "artifacts": metrics.artifact_report(out)}
    if gt.get("has_motion"):
        in_wave = metrics.lk_displacement_to_reference(frames, ref="first", roi=roi)
        o_wave = metrics.lk_displacement_to_reference(out, ref="first", roi=roi)
        gta = synthetic.gt_amplitude(gt)
        axis = 0 if gta[0] >= gta[1] else 1
        a_in = metrics.waveform_amplitude(in_wave)[axis]
        a_out = metrics.waveform_amplitude(o_wave)[axis]
        rec.update({"in_amp_px": a_in, "out_amp_px": a_out,
                    "gain": (a_out / a_in) if a_in > 0.05 else float("nan"),
                    "wave_corr": metrics.waveform_correlation(in_wave, o_wave)[axis],
                    "gt_amp_px": gta[axis]})
    active = gt.get("active_mask")
    if gt.get("case") in ("shm_h", "shm_v") or gt["dx"].any() or gt["dy"].any():
        if active is None or not np.asarray(active).any():
            active = np.ones(frames.shape[1:3], bool)
    masks = {"active": active if (active is not None and np.asarray(active).any()) else None}
    m_all = np.ones(frames.shape[1:3], bool)
    masks["static"] = (~np.asarray(active)) if (active is not None and np.asarray(active).any()) else m_all
    rec["regions"] = region_metrics(frames, out, args.fps, args.f_lo, args.f_hi, masks)
    rec["w_stats"] = wmap_stats(diag.get("w_map"), active)
    # Gate 的原始量：相干族用**未截断**的 A（w 会饱和到 1），B2/平滑族用 w
    ev = diag.get("a_map")
    if ev is None:
        ev = diag.get("w_map")
        rec["ev_kind"] = "w" if ev is not None else None
    else:
        rec["ev_kind"] = "A"
    rec["ev_stats"] = wmap_stats(ev, active)
    rec["gate_diag"] = {k: {kk: vv for kk, vv in v.items() if kk.startswith("w_")}
                        for k, v in diag.get("per_band", {}).items()}
    return rec


CASES = {
    # name: (case, amp, noise, seed)
    "local_005":       ("local", 0.05, 0.0, 0),
    "local_025":       ("local", 0.25, 0.0, 0),
    "local_lowc_005":  ("local_lowc", 0.05, 0.0, 0),
    "shm_h_005":       ("shm_h", 0.05, 0.0, 0),
    "local_n002_005":  ("local", 0.05, 0.02, 7),
    "local_n002_025":  ("local", 0.25, 0.02, 7),
    "static_noise":    ("static_noise", 0.0, 0.02, 7),
    "brightness":      ("brightness", 0.0, 0.0, 0),
    "texture_change":  ("texture_change", 0.0, 0.0, 0),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="ablate",
                    choices=("baseline", "ablate", "alpha", "gates"))
    ap.add_argument("--methods", nargs="+", default=["G0", "G1", "G2", "G3", "G4", "G5"])
    ap.add_argument("--windows", type=int, nargs="+", default=[5, 9])
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument("--alphas", type=float, nargs="+", default=[5.0])
    ap.add_argument("--T", type=int, default=120)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--freq", type=float, default=0.5)
    ap.add_argument("--f-lo", type=float, default=0.3)
    ap.add_argument("--f-hi", type=float, default=1.0)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.02, help="σ_z 标定用的噪声强度")
    ap.add_argument("--tau", type=float, default=0.35)
    ap.add_argument("--ablation-json", default=None)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    if args.stage == "gates":
        return evaluate_gates(args)

    # ---- 标定（始终用独立片段，绝不用测试片段）----
    calib_fr = synthetic.make_sequence(kind="static_noise", T=args.T, H=args.size,
                                       W=args.size, fps=args.fps,
                                       noise_sigma=args.noise, seed=42,
                                       noise_seed=424242)
    cal = calibrate_noise(calib_fr[0], levels=args.levels, order=args.order)
    sigma_A = cal["sigma_A"]
    sigma_z_by_w = {}
    for w in sorted({int(x) for x in args.windows} | {5, 9}):
        sigma_z_by_w[w] = calibrate_sigma_z(w, args.T, args.size, args.fps,
                                            args.f_lo, args.f_hi, args.noise,
                                            args.levels, args.order)
    report = {"config": vars(args), "gates": GATES,
              "calibration": {"sigma_A": {f"{a}-{b}": v for (a, b), v in sigma_A.items()},
                              "sigma_z": {str(w): {f"{a}-{b}": v for (a, b), v in sz.items()}
                                          for w, sz in sigma_z_by_w.items()}},
              "runs": {}}

    for cname in args.cases:
        case, amp, noise, seed = CASES[cname]
        frames, gt = build_case(case, args.T, args.size, args.fps, args.freq,
                                amp, noise, seed)
        for method in args.methods:
            for w in ([0] if method == "G0" else args.windows):
                sz = sigma_z_by_w.get(int(w)) if method != "G0" else None
                cfg = build_cfg(method, w, sz, sigma_A, args.tau)
                for alpha in args.alphas:
                    key = f"{cname}|{method}|W={w}|a={alpha:g}"
                    rec = run_one(frames, gt, args, cfg, alpha)
                    rec.update({"case": cname, "method": method, "W": w,
                                "noise_sigma": noise, "seed": seed})
                    report["runs"][key] = rec
                    print(f"[{key:34s}] gain={rec.get('gain', float('nan')):6.3f} "
                          f"corr={rec.get('wave_corr', float('nan')):+.3f} "
                          f"ev={rec.get('ev_kind')} "
                          f"act={rec['ev_stats']['w_active']:.4f} "
                          f"sta={rec['ev_stats']['w_static']:.4f} "
                          f"contrast={rec['ev_stats']['contrast_active_bg']:.2f} "
                          f"iou={rec['ev_stats']['iou']:.3f} ({rec['seconds']:.1f}s)")

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                              default=str), encoding="utf-8")
    print(f"\n[save] {args.out_json}")
    return 0


def evaluate_gates(args) -> int:
    """G1/G2/G3 判定（定义与 Phase D 一致：比值 / 对比 / 混淆比）。"""
    rep = json.loads(Path(args.ablation_json).read_text())
    runs = rep["runs"]

    def w_of(case, method, w, alpha=5.0):
        r = runs.get(f"{case}|{method}|W={w}|a={alpha:g}") or {}
        return r.get("ev_stats") or r.get("w_stats") or {}

    out = {"gates": GATES, "measured": {}, "verdict": {}, "per_method": {}}
    for method in sorted({r["method"] for r in runs.values()}):
        ws = [r["W"] for r in runs.values() if r["method"] == method]
        for w in sorted(set(ws)):
            if method == "G0":
                continue
            a = w_of("local_n002_005", method, w)      # 0.05px + σ=0.02：运动
            s = w_of("static_noise", method, w)        # 静止 + σ=0.02
            br = w_of("brightness", method, w)         # 亮度变化：无活动区 ⇒ 用全帧均值
            tx = w_of("texture_change", method, w)     # 纹理变化：用 patch 内的均值
            m = w_of("local_005", method, w)           # 干净 0.05px
            m_ev = m.get("w_active")
            row = {
                "method": method, "W": w,
                "w_motion_noisy": a.get("w_active"), "w_static_noisy": s.get("w_static"),
                "G1": ((a.get("w_active") or 0) / (s.get("w_static") or 1e-12)),
                "G2": a.get("contrast_active_bg"),
                "w_motion_clean": m_ev,
                "G3_brightness": ((br.get("w_static") or 0)
                                  / max(m_ev or 1e-12, 1e-12)),
                "G3_texture": ((tx.get("w_active") or 0)
                               / max(m_ev or 1e-12, 1e-12)),
                "iou_noisy": a.get("iou"),
            }
            out["per_method"][f"{method}|W={w}"] = row
    # 取"运动保真度可接受"的最优方法：先看 Gate，再看 wave_corr/gain
    best = None
    for k, r in out["per_method"].items():
        ok = (r["G1"] >= GATES["G1_motion_over_static"]
              and r["G2"] >= GATES["G2_localization_contrast"]
              and max(r["G3_brightness"], r["G3_texture"]) < GATES["G3_confounder_over_motion"])
        r["pass"] = bool(ok)
        if ok and (best is None or r["G1"] > out["per_method"][best]["G1"]):
            best = k
    out["best"] = best
    out["decision"] = ("GO（可考虑重开 Phase E）" if best else
                       "NO-GO（保留 B2 为最终系统）")
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2,
                                              default=str), encoding="utf-8")
    print(json.dumps(out["per_method"], ensure_ascii=False, indent=2))
    print(f"\nbest={best}  decision={out['decision']}")
    print(f"[save] {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
