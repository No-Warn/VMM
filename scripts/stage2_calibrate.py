#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 2：静止背景噪声标定（逐尺度、逐方向），并给出逐带增益上限的**建议**。

关键：σ 必须在**同一金字塔带幅值单位**上标定（静止视频/静止 ROI 上 A(x,t) 的时间
标准差 + 稳健分位），而不是把像素噪声方差加到归一化幅值上。

用法::

    # 合成：独立噪声实现（seed 与测试视频不同）
    python scripts/stage2_calibrate.py --synthetic static_noise --noise-sigma 0.02 \
        --out configs/vmm_stage2/sigma_syn002.json

    # 真实视频：只用**静止 ROI**（低能量 40% 像素）
    python scripts/stage2_calibrate.py --video baby.mp4 --scale 0.5 --max-frames 160 \
        --static-roi --out configs/vmm_stage2/sigma_baby.json
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
from vmm.reliability import (calibrate_noise, cap_suggestions,   # noqa: E402
                             save_calibration)


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", choices=("static", "static_noise"))
    src.add_argument("--video")
    ap.add_argument("--noise-sigma", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=1234, help="合成标定视频的随机种子（与测试不同）")
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=160)
    ap.add_argument("--static-roi", action="store_true",
                    help="真实视频只用静止 ROI（按时间能量低 40% 选）")
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--alpha-max", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mask = None
    if args.synthetic:
        video, gt = synthetic.make_sequence(kind=args.synthetic, T=args.T, H=args.size,
                                            W=args.size, fps=args.fps,
                                            noise_sigma=args.noise_sigma if
                                            args.synthetic == "static_noise" else 0.0,
                                            seed=args.seed)
        tag = f"{args.synthetic}(seed={args.seed},σ={args.noise_sigma})"
    else:
        video, meta = video_io.read_video(args.video, max_frames=args.max_frames,
                                          scale=args.scale)
        args.fps = meta.fps
        if args.static_roi:
            m = metrics.masks_from_motion(video, active_pct=90, static_pct=40)
            mask = m["static"]
        tag = f"{args.video}(static_roi={args.static_roi})"
    print(f"[calib] {tag}  T={video.shape[0]} {video.shape[2]}x{video.shape[1]} "
          f"levels={args.levels} order={args.order}")

    calib = calibrate_noise(video, levels=args.levels, order=args.order, mask=mask)
    save_calibration(calib, args.out)
    print(f"[save] {args.out}\n")
    print(f"{'band':>6} {'shape':>10} {'A_median':>10} {'σ_A':>10} {'σ_A/A_med':>10} "
          f"{'σ_φ(rad)':>10} {'|Δφ|mean':>10} {'建议β':>8}")
    sugg = cap_suggestions(calib["band_stats"], alpha_max=args.alpha_max)
    for k in sorted(calib["sigma_A"], key=lambda x: (x[0], x[1])):
        st = calib["band_stats"][k]
        print(f"{f'{k[0]}-{k[1]}':>6} {str(st['coeff_shape']):>10} {st['A_median']:>10.4f} "
              f"{st['sigma_A']:>10.4f} {st['noise_over_signal_A']:>10.3f} "
              f"{st['sigma_phi']:>10.4f} {st['dphi_abs_mean']:>10.4f} {sugg[k]:>8.2f}")
    print("\n[sugg] 建议上限 β 由 统计比值 dphi_abs_mean/(σ_φ) 推出，"
          "仅作消融设计起点，**不会自动生效**；数值必须经消融验证。")
    sugg_path = Path(args.out).with_name(Path(args.out).stem + "_cap_suggestions.json")
    sugg_path.write_text(json.dumps({f"{a}-{b}": v for (a, b), v in sugg.items()},
                                    indent=2), encoding="utf-8")
    print(f"[save] {sugg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())