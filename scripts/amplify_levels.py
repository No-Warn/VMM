#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按"总放大倍数"输出多个视频（默认 B2 门控）。

约定（**关键，别混淆**）：本仓库的 ``alpha`` 是**额外增益**，总增益 ≈ ``1 + alpha``。
所以"5×" ⇒ ``alpha=4``、"50×" ⇒ ``alpha=49``。本脚本按目标倍数换算，并把两列都写进 JSON。

输出：
* ``<out-dir>/baby_x5.mp4``        放大结果
* ``<out-dir>/baby_x5_sbs.mp4``    左=原视频、右=放大（同一帧率/尺寸）
* ``<out-dir>/metrics.json``       每个倍数的：分区目标带能量 in/out/比值、
                                   5 个独立位移估计器读数与分歧、伪影统计
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vmm import metrics, video_io                     # noqa: E402
from vmm.phase_magnifier import PhaseBasedMagnifier    # noqa: E402
from vmm.reliability import ReliabilityConfig          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="baby.mp4")
    ap.add_argument("--targets", type=float, nargs="+", default=[5, 10, 20, 50],
                    help="总放大倍数；内部换算 alpha = target − 1")
    ap.add_argument("--f-lo", type=float, default=0.2)
    ap.add_argument("--f-hi", type=float, default=0.8)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=240)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--channels", default="color")
    ap.add_argument("--mode", default="B2", choices=("B2", "B1", "none"),
                    help="B2=推荐；none=纯阶段 1 基线（E0）")
    ap.add_argument("--phase-clip", type=float, default=None)
    ap.add_argument("--out-dir", default="outputs/amplify_multi")
    args = ap.parse_args()

    video, meta = video_io.read_video(args.video, max_frames=args.max_frames,
                                     scale=args.scale)
    m = metrics.masks_from_motion(video, active_pct=90, static_pct=40)
    masks = {"active": m["active"], "static": m["static"]}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"config": vars(args), "fps": meta.fps,
              "frames": int(video.shape[0]), "size": list(video.shape[1:3]),
              "note": "alpha = target_gain − 1；实测'总增益'应接近 target_gain",
              "runs": {}}
    for target in args.targets:
        alpha = float(target) - 1.0
        cfg = ReliabilityConfig(mode="M0" if args.mode == "none" else args.mode)
        mag = PhaseBasedMagnifier(fps=meta.fps, f_lo=args.f_lo, f_hi=args.f_hi,
                                  levels=args.levels, order=args.order,
                                  channels=args.channels,
                                  phase_clip=args.phase_clip,
                                  reliability_cfg=cfg)
        amp, diag = mag.amplify(video, alpha)
        tag = f"x{target:g}"
        p_amp = out_dir / f"baby_{tag}.mp4"
        p_sbs = out_dir / f"baby_{tag}_sbs.mp4"
        video_io.write_video(amp, p_amp, fps=meta.fps)
        video_io.write_side_by_side(video, amp, p_sbs, fps=meta.fps)

        regions = {}
        e_in = metrics.band_energy_map(video, meta.fps, args.f_lo, args.f_hi)
        e_out = metrics.band_energy_map(amp, meta.fps, args.f_lo, args.f_hi)
        for name, mk in masks.items():
            if mk is None or not mk.any():
                continue
            bi, bo = float(e_in[mk].mean()), float(e_out[mk].mean())
            regions[name] = {"in": bi, "out": bo,
                             "ratio": (bo / bi) if bi > metrics.ENERGY_FLOOR else None}
        amps = metrics.measure_all_estimators(amp, roi=None)
        amps.pop("_waves", None)
        est = {k: float(v["amp_x"]) for k, v in amps.items()}
        art = metrics.artifact_report(amp)
        report["runs"][tag] = {"target_gain": target, "alpha": alpha,
                               "regions": regions, "estimators_px": est,
                               "estimator_spread_px": [min(est.values()), max(est.values())],
                               "artifacts": art, "beta_max": float(diag.get("beta_max", 0.0))
                               if "beta_max" in diag else None,
                               "output": str(p_amp), "side_by_side": str(p_sbs)}
        print(f"[{tag:>4s}] alpha={alpha:>5g} | 活动带内×{regions['active']['ratio']:.2f} "
              f"静止带内×{regions['static']['ratio']:.2f} "
              f"| 位移估计 {min(est.values()):.3f}~{max(est.values()):.3f} px "
              f"| 饱和 {art['saturation_frac']:.3f} 闪烁 {art['flicker_std']:.2e} "
              f"| {p_amp.name}")

    (out_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[save] {out_dir/'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
