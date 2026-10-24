#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI：对任意连续视频做相位式微运动放大。

    python scripts/run_vmm.py --video baby.mp4 --out outputs/vmm2/baby_a5.mp4 \\
        --alpha 5 --f-lo 0.3 --f-hi 1.0 --levels 4 --order 3 --channels color \\
        --scale 0.5 --metrics --side-by-side

注意：``alpha`` 是**额外**增益（0 = 不放大；理论总量约 ``1+alpha``），
与 FD4MM 数据集里的合成倍率 ``m`` **不是同一个量**，不要把 ``m`` 传进来。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vmm import metrics, video_io                      # noqa: E402
from vmm.phase_magnifier import PhaseBasedMagnifier     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="相位式 Eulerian 微运动放大（阶段 1）")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, required=True, help="额外运动增益（0=不放大）")
    ap.add_argument("--f-lo", type=float, required=True, help="目标频带下限 Hz")
    ap.add_argument("--f-hi", type=float, required=True, help="目标频带上限 Hz")
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--channels", default="gray", choices=("gray", "color"))
    ap.add_argument("--max-gain", type=float, default=None)
    ap.add_argument("--phase-clip", type=float, default=None,
                    help="相位平移逐带上限（弧度）；建议真实视频给 1.2，缺省不限")
    ap.add_argument("--amp-floor", type=float, default=0.0)
    ap.add_argument("--filter-order", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--chunk-frames", type=int, default=0,
                    help=">0 时用 overlap-save 分块流式处理长视频（含接缝自检）")
    ap.add_argument("--phi-dtype", default="float32", choices=("float32", "float16"),
                    help="局部相位存储精度；长视频可用 float16 省一半内存")
    ap.add_argument("--side-by-side", action="store_true")
    ap.add_argument("--metrics", action="store_true")
    ap.add_argument("--save-diagnostics", default=None, help="诊断 JSON 路径")
    args = ap.parse_args()

    video, meta = video_io.read_video(args.video, max_frames=args.max_frames,
                                      scale=args.scale)
    print(f"[in ] {args.video}: {meta.n_frames} 帧 {meta.width}x{meta.height} "
          f"@ {meta.fps:.2f} fps（scale={args.scale}）")

    mag = PhaseBasedMagnifier(fps=meta.fps, f_lo=args.f_lo, f_hi=args.f_hi,
                              levels=args.levels, order=args.order,
                              max_gain=args.max_gain, channels=args.channels,
                              filter_order=args.filter_order,
                              phase_clip=args.phase_clip,
                              amp_floor=args.amp_floor,
                              phi_dtype=args.phi_dtype)
    if args.chunk_frames > 0:
        out, diag = mag.amplify_chunked(video, alpha=args.alpha,
                                        chunk_frames=args.chunk_frames)
        print(f"[chunk] chunked={diag['chunked']} "
              f"margin={diag.get('margin_frames')} {diag.get('note', '')}")
        if diag.get("chunked") and diag.get("chunk_boundaries"):
            score, med_all, med_b = metrics.seam_score(out, diag["chunk_boundaries"])
            print(f"[seam] 边界/全片 帧间差分比 = {score:.2f} "
                  f"(全片 {med_all:.2e}，边界 {med_b:.2e})；≈1 表示无接缝")
            report_seam = {"seam_score": score}
        else:
            report_seam = {}
    else:
        out, diag = mag.amplify(video, alpha=args.alpha)
        report_seam = {}
    video_io.write_video(out, args.out, meta.fps)
    print(f"[out] {args.out}")
    if args.side_by_side:
        sbs = str(Path(args.out).with_name(Path(args.out).stem + "_sbs.mp4"))
        video_io.write_side_by_side(video, out, sbs, meta.fps)
        print(f"[out] {sbs}  (左=原视频 右=放大)")

    checks = video_io.assert_consistent(
        video_io.VideoMeta(str(args.video), meta.width, meta.height, meta.fps, out.shape[0]),
        out, meta.fps)
    print(f"[chk] 帧数/分辨率/顺序一致性: {checks}")
    print(f"[art] {metrics.artifact_report(out)}")

    report = {"input": str(args.video), "output": str(args.out),
              "alpha": args.alpha, "channels": args.channels,
              "meta": meta.__dict__, "consistency": checks,
              "diagnostics": {k: v for k, v in diag.items() if k != "per_band"},
              "artifacts": metrics.artifact_report(out)}
    if args.metrics:
        em_in = metrics.masks_from_motion(video, gt_mask=None)
        for name, m in (("active", em_in["active"]), ("static", em_in["static"])):
            e_in = metrics.region_energy(video, m)
            e_out = metrics.region_energy(out, m)
            b_in = float(metrics.band_energy_map(video, meta.fps, args.f_lo, args.f_hi)[m].mean())
            b_out = float(metrics.band_energy_map(out, meta.fps, args.f_lo, args.f_hi)[m].mean())
            print(f"[nrg] {name:6s} 时间能量 in={e_in:.5f} out={e_out:.5f} "
                  f"(×{e_out/max(e_in,1e-12):.2f}) | 目标带内 in={b_in:.5f} out={b_out:.5f} "
                  f"(×{b_out/max(b_in,1e-12):.2f})")
            report[f"energy_{name}"] = {"temporal_in": e_in, "temporal_out": e_out,
                                        "band_in": b_in, "band_out": b_out}
        print("[note] SSIM/PSNR 不用于判断放大的正确性；真实视频的原始帧不是放大结果的真值。")
    if args.save_diagnostics:
        Path(args.save_diagnostics).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_diagnostics).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"[save] {args.save_diagnostics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())