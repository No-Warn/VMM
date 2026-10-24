#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比不同放大强度（α）的实测效果。

两种用法：

1) **合成**（有真值）——报真实运动增益并与理论 ``1+α·|H(f)|²`` 对照::

    python scripts/compare_gains.py --synthetic shm_h --alphas 0 1 2 5 10

2) **真实视频**（无真值）——只能报**估计**增益与其**不确定度**（多估计器分歧），
   外加活动区/静止区的频带能量比::

    python scripts/compare_gains.py --video baby.mp4 --alphas 0 1 2 5 10 \
        --f-lo 0.2 --f-hi 0.8 --scale 0.5 --max-frames 240

注意：``α`` 是**额外**增益（0 = 不放大；线性区总量≈1+α），
与 FD4MM 数据集里的合成倍率 ``m`` 不是同一个量。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vmm import metrics, synthetic, video_io          # noqa: E402
from vmm.phase_magnifier import PhaseBasedMagnifier   # noqa: E402
from vmm.temporal_filter import bandpass_filter, bandpass_gain_at  # noqa: E402


def build_magnifier(args, fps):
    return PhaseBasedMagnifier(fps=fps, f_lo=args.f_lo, f_hi=args.f_hi,
                               levels=args.levels, order=args.order,
                               channels=args.channels, phase_clip=args.phase_clip,
                               amp_floor=args.amp_floor)


def synth_gain_table(video, args, fps, axis):
    """合成数据：主估计器用真值甄别（同 evaluate_synthetic 的口径）。"""
    allm = metrics.measure_all_estimators(video)
    waves = allm.pop("_waves")
    amps = {k: (v["amp_x"] if axis == 0 else v["amp_y"]) for k, v in allm.items()}
    return amps, waves


def main() -> int:
    ap = argparse.ArgumentParser(description="对比不同 α 的实测放大效果")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="真实视频路径")
    src.add_argument("--synthetic", choices=("shm_h", "shm_v", "local", "static",
                                             "static_noise"), help="合成输入类型")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0, 1, 2, 5, 10])
    ap.add_argument("--f-lo", type=float, default=0.3)
    ap.add_argument("--f-hi", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--amp-px", type=float, default=0.5)
    ap.add_argument("--freq", type=float, default=0.5)
    ap.add_argument("--noise-sigma", type=float, default=0.0)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--channels", default=None, choices=("gray", "color"),
                    help="默认：彩色输入用 color（否则 α=0 的 |out-in| 没有意义），灰度输入用 gray")
    ap.add_argument("--phase-clip", type=float, default=None)
    ap.add_argument("--amp-floor", type=float, default=0.0)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    gt = None
    if args.synthetic:
        video, gt = synthetic.make_sequence(kind=args.synthetic, T=args.frames,
                                            H=args.size, W=args.size, fps=args.fps,
                                            amp_px=args.amp_px, freq=args.freq,
                                            noise_sigma=args.noise_sigma,
                                            patch=args.size // 3, seed=42)
        fps, roi = args.fps, synthetic.measurement_roi(gt)
        gt_amp = synthetic.gt_amplitude(gt)
        axis = 0 if gt_amp[0] >= gt_amp[1] else 1
    else:
        video, meta = video_io.read_video(args.video, max_frames=args.max_frames,
                                          scale=args.scale)
        fps, roi = meta.fps, None
        axis = None
        gt_amp = (float("nan"), float("nan"))

    if args.channels is None:
        args.channels = "color" if video.ndim == 4 else "gray"
        print(f"[note] 未指定 --channels，按输入自动选 {args.channels}"
              f"{'（灰度输入时 α=0 的 |out-in| 才有恒等意义）' if args.channels == 'gray' else ''}")
    filt = bandpass_filter(fps, args.f_lo, args.f_hi, 2)
    g_theory = bandpass_gain_at(filt, args.freq, fps, passes=2)
    print(f"输入: {'合成 ' + args.synthetic if args.synthetic else args.video}  "
          f"{video.shape[0]} 帧 {video.shape[2]}x{video.shape[1]} @ {fps:.2f}fps  "
          f"带通 {args.f_lo}-{args.f_hi}Hz")
    if args.synthetic:
        print(f"真值振幅(dx,dy)=({gt_amp[0]:.3f},{gt_amp[1]:.3f}) px → 运动轴 "
              f"{'x' if axis == 0 else 'y'}")
    print(f"理论总增益 ≈ 1 + α·|H({args.freq}Hz)|² = 1 + {g_theory:.3f}·α\n")

    in_all = metrics.measure_all_estimators(video, roi=roi)
    in_waves = in_all.pop("_waves")
    in_amps = {k: (v["amp_x"] if (axis or 0) == 0 else v["amp_y"]) for k, v in in_all.items()}
    # 主估计器：合成数据上按真值甄别（±15%）；无真值（真实视频）时用 LK——
    # 它是 4 类合成输入里唯一在 3/4 情形下通过真值甄别的估计器。
    if gt is not None and axis is not None:
        gt_a = gt_amp[axis]
        cand = [k for k, v in in_amps.items()
                if v == v and gt_a > 0.05 and abs(v - gt_a) / gt_a <= 0.15]
        primary = (min(cand, key=lambda k: abs(in_amps[k] - gt_a)) if cand else "lk_first")
    else:
        primary = "lk_first"
    keep = [k for k, v in in_amps.items() if v == v and v > 1e-3]
    amp_med = float(np.median([in_amps[k] for k in keep])) if keep else float("nan")
    print(f"[note] 主估计器 = {primary}"
          f"{'（按真值甄别）' if gt is not None else '（无真值，默认用 LK）'}")
    if amp_med == amp_med and amp_med < 0.1:
        print(f"[warn] 输入振幅中位仅 {amp_med:.4f} px，接近估计器噪声底 —— "
              f"下面的**增益数值不可靠**，只能当作上界/参考，不能当作精确倍率。")
    print("输入振幅·各估计器（真实视频无真值，只能把分歧当不确定度）:")
    for k, v in in_all.items():
        a = v["amp_x"] if (axis or 0) == 0 else v["amp_y"]
        print(f"  {k:>16s} amp={a:8.4f} resp={v['resp']:.3f}")
    if keep:
        vals = [in_amps[k] for k in keep]
        print(f"  → 中位 {np.median(vals):.4f} px，范围 {min(vals):.4f}~{max(vals):.4f} px "
              f"（相对离散 {100*(max(vals)-min(vals))/max(np.median(vals),1e-9):.0f}%）")

    mag = build_magnifier(args, fps)
    rows = []
    print(f"\n{'α':>5} {'主估计器增益':>12} {'LK 增益':>9} {'中位增益':>9} {'理论':>6} "
          f"{'活动带内×':>10} {'静止带内×':>10} {'|out-in|max':>11}")
    for a in args.alphas:
        out, diag = mag.amplify(video, alpha=a)
        o_all = metrics.measure_all_estimators(out, roi=roi)
        o_waves = o_all.pop("_waves")
        o_amps = {k: (v["amp_x"] if (axis or 0) == 0 else v["amp_y"]) for k, v in o_all.items()}
        gains = [o_amps[k] / in_amps[k] for k in keep
                 if in_amps[k] > 1e-3 and o_amps[k] == o_amps[k]]
        g_med = float(np.median(gains)) if gains else float("nan")
        g_lk = (o_amps.get("lk_first", float("nan")) / in_amps.get("lk_first", float("nan"))
                if in_amps.get("lk_first", 0) > 1e-3 else float("nan"))
        g_main = (o_amps.get(primary, float("nan")) / in_amps.get(primary, float("nan"))
                  if in_amps.get(primary, 0) > 1e-3 else float("nan"))
        act_r = metrics.band_energy_ratio(video, out, np.ones(video.shape[1:3], bool),
                                          fps, args.f_lo, args.f_hi)
        m = metrics.masks_from_motion(video, gt_mask=gt["active_mask"] if gt else None)
        act_r = metrics.band_energy_ratio(video, out, m["active"], fps, args.f_lo, args.f_hi)
        stat_r = metrics.band_energy_ratio(video, out, m["static"], fps, args.f_lo, args.f_hi)
        err = metrics.recon_error(video, out)["max_abs"]
        theory = 1.0 + a * g_theory
        print(f"{a:>5g} {g_main:>12.2f} {g_lk:>9.2f} {g_med:>9.2f} {theory:>6.2f} "
              f"{act_r:>10.2f} {stat_r:>10.2f} {err:>11.2e}")
        rows.append({"alpha": a, "gain_primary": g_main, "gain_lk": g_lk,
                     "gain_median": g_med, "theory": theory,
                     "band_active_ratio": act_r, "band_static_ratio": stat_r,
                     "recon_err_max": err, "beta_max": max(
                         (v["beta_max"] for v in diag.get("per_band", {}).values()),
                         default=float("nan"))})
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(
            {"input": args.video or args.synthetic, "fps": fps,
             "band": [args.f_lo, args.f_hi], "gain_at_freq": g_theory,
             "gt_amp": list(gt_amp), "input_estimators": in_all, "rows": rows},
            ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"\n[save] {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())