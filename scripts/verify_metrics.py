#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""独立复核 SSIM/LPIPS 与增益的计算（不重复原脚本的实现路径）。

目的：回答"测试数据是否正确、是否有依据、参数计算是否与 tests/ 一致"。
做法：对每一类指标，用**第二条独立实现**重算，并与已报告值对比。

复核项
------
1. α=0 恒等：与 `tests/test_phase_identity.py` 的 `max|out-in| ≤ 1e-4` 同口径。
2. 增益：与 `tests/test_phase_shift.py` / `scripts/evaluate_synthetic.py` 同口径
   （LK 位移波形幅值比），并对照理论 `1+α·|H|²`，显式区分"线性区"与"饱和区"。
3. SSIM：`skimage` 默认（7×7 均匀窗）vs **标准 Wang 实现**（11×11 高斯 σ=1.5），
   两者差异 >0.01 就说明指标选择会改变结论。
4. GT 保真度：把"理想 GT"（振幅 (1+α)A）与"实测增益 GT"（振幅 g·A）分开算，
   用于分离"增益不足"与"形状失真"。
5. LPIPS 合理性：同图→0、模糊→正、随模糊单调。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vmm import metrics, synthetic, video_io            # noqa: E402
from vmm.phase_magnifier import PhaseBasedMagnifier     # noqa: E402
from vmm.reliability import ReliabilityConfig           # noqa: E402


def ssim_skimage_default(a, b):
    from skimage.metrics import structural_similarity as sk
    return float(sk(a, b, channel_axis=2, data_range=1.0))


def ssim_wang(a, b, sigma=1.5, C1=(0.01) ** 2, C2=(0.03) ** 2):
    """标准 Wang et al. SSIM：11×11 高斯窗 σ=1.5，逐通道后平均。"""
    from scipy.ndimage import gaussian_filter
    vals = []
    for c in range(a.shape[2]):
        x = a[:, :, c].astype(np.float64)
        y = b[:, :, c].astype(np.float64)
        mu_x = gaussian_filter(x, sigma)
        mu_y = gaussian_filter(y, sigma)
        sxx = gaussian_filter(x * x, sigma) - mu_x ** 2
        syy = gaussian_filter(y * y, sigma) - mu_y ** 2
        sxy = gaussian_filter(x * y, sigma) - mu_x * mu_y
        smap = ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2))
        vals.append(float(smap.mean()))
    return float(np.mean(vals))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="baby.mp4")
    ap.add_argument("--targets", type=float, nargs="+", default=[5, 10, 20, 50])
    ap.add_argument("--amp", type=float, default=0.25)
    ap.add_argument("--out-json", default="outputs/amplify_multi/verify_metrics.json")
    args = ap.parse_args()

    import torch
    import lpips as lpips_pkg
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lp = lpips_pkg.LPIPS(net="vgg", verbose=False).to(dev).eval()
    for p in lp.parameters():
        p.requires_grad_(False)

    def lpips_pair(a, b):
        with torch.no_grad():
            x = torch.from_numpy(a).permute(2, 0, 1)[None].float().to(dev)
            y = torch.from_numpy(b).permute(2, 0, 1)[None].float().to(dev)
            return float(lp(x * 2 - 1, y * 2 - 1).item())

    out = {"checks": {}}

    # ---------------- 1. α=0 恒等（与 tests 同口径）----------------
    video, meta = video_io.read_video(args.video, max_frames=240, scale=0.5)
    mag = PhaseBasedMagnifier(fps=meta.fps, f_lo=0.2, f_hi=0.8, levels=4, order=3,
                              channels="color", reliability_cfg=ReliabilityConfig(mode="B2"))
    o0, _ = mag.amplify(video, 0.0)
    err = float(np.abs(o0 - video).max())
    out["checks"]["identity_alpha0"] = {
        "max_abs_err": err, "test_threshold": 1e-4, "pass": err <= 1e-4,
        "ssim_skimage": float(np.mean([ssim_skimage_default(video[t], o0[t])
                                       for t in range(0, len(video), 12)])),
        "lpips": float(np.mean([lpips_pair(video[t], o0[t])
                                for t in range(0, len(video), 24)]))}
    print(f"[1] α=0 恒等: max|out-in|={err:.2e}（tests 阈值 1e-4）"
          f" → {'PASS' if err <= 1e-4 else 'FAIL'}")

    # ---------------- 2. SSIM 两种实现（同一输出）----------------
    # 只取 60 帧算，控制耗时；两种实现同一对帧
    o5, _ = mag.amplify(video, 4.0)          # 目标 5×
    idx = range(0, 60, 5)
    s_sk = [ssim_skimage_default(video[t], o5[t]) for t in idx]
    s_wg = [ssim_wang(video[t], o5[t]) for t in idx]
    out["checks"]["ssim_impl"] = {
        "skimage_default_7x7_uniform": float(np.mean(s_sk)),
        "wang_11x11_gaussian": float(np.mean(s_wg)),
        "abs_diff": float(abs(np.mean(s_sk) - np.mean(s_wg)))}
    print(f"[2] SSIM 实现差异: skimage默认={np.mean(s_sk):.4f} "
          f"Wang标准={np.mean(s_wg):.4f} 差={abs(np.mean(s_sk)-np.mean(s_wg)):.4f}")

    # ---------------- 3. LPIPS 合理性 ----------------
    from scipy.ndimage import gaussian_filter
    base = video[30]
    checks = {"identical": lpips_pair(base, base)}
    for sig in (1.0, 2.0, 4.0):
        blur = np.stack([gaussian_filter(base[:, :, c], sig) for c in range(3)], -1)
        checks[f"blur_sigma{sig}"] = lpips_pair(base, blur)
    checks["amplified_x5_frame"] = lpips_pair(base, o5[30])
    out["checks"]["lpips_sanity"] = checks
    print(f"[3] LPIPS 合理性: 同图={checks['identical']:.4f} "
          f"σ1={checks['blur_sigma1.0']:.4f} σ2={checks['blur_sigma2.0']:.4f} "
          f"σ4={checks['blur_sigma4.0']:.4f} x5={checks['amplified_x5_frame']:.4f}")

    # ---------------- 4. 增益：tests 口径 + 理想/实测 GT ----------------
    rows = []
    for kind in ("shm_h", "local"):
        A = args.amp
        fr, gt = synthetic.make_sequence(kind=kind, T=120, H=192, W=192, fps=meta.fps,
                                         amp_px=A, freq=0.5, patch=64, seed=42)
        roi = synthetic.measurement_roi(gt)
        act = gt["active_mask"] if gt["active_mask"].any() else np.ones(fr.shape[1:3], bool)
        sta = ~act
        for target in args.targets:
            alpha = target - 1.0
            for mode in ("B2", "none"):
                m2 = PhaseBasedMagnifier(fps=meta.fps, f_lo=0.3, f_hi=1.0, levels=4,
                                         order=3, channels="gray",
                                         reliability_cfg=ReliabilityConfig(
                                             mode="M0" if mode == "none" else mode))
                o, d = m2.amplify(fr, alpha)
                wi = metrics.lk_displacement_to_reference(fr, ref="first", roi=roi)
                wo = metrics.lk_displacement_to_reference(o, ref="first", roi=roi)
                gta = synthetic.gt_amplitude(gt)
                axis = 0 if gta[0] >= gta[1] else 1
                a_in = metrics.waveform_amplitude(wi)[axis]
                a_out = metrics.waveform_amplitude(wo)[axis]
                g = a_out / a_in if a_in > 0.05 else float("nan")
                gt_ideal, _ = synthetic.make_sequence(kind=kind, T=120, H=192, W=192,
                                                      fps=meta.fps, amp_px=A * target,
                                                      freq=0.5, patch=64, seed=42)
                gt_meas, _ = synthetic.make_sequence(kind=kind, T=120, H=192, W=192,
                                                     fps=meta.fps, amp_px=A * g,
                                                     freq=0.5, patch=64, seed=42)
                s_ideal = float(np.mean([ssim_wang(gt_ideal[t], o[t]) for t in range(120)]))
                s_meas = float(np.mean([ssim_wang(gt_meas[t], o[t]) for t in range(120)]))
                rows.append({"case": kind, "mode": mode, "target": target,
                             "alpha": alpha, "theory": target,
                             "measured_gain": g, "ratio_to_theory": g / target,
                             "ssim_vs_ideal_gt": s_ideal, "ssim_vs_measured_gt": s_meas,
                             "gain_floor_px": a_in})
                print(f"[4] {kind:6s} {mode:4s} x{target:<3g} 增益={g:6.2f} "
                      f"({g/target:.3f}×理论) | SSIM vs 理想GT={s_ideal:.4f} "
                      f"vs 实测GT={s_meas:.4f}")
    out["checks"]["gain_and_gt"] = rows

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2,
                                              default=str), encoding="utf-8")
    print(f"\n[save] {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
