#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 2 交付物：并排视频与局部 ROI 对比（任务书 §9）。

对**同一输入**分别生成 E0（阶段 1 基线）与 B1（最佳配置）的放大结果，并写出：

* ``<tag>_<variant>_a<α>.mp4``            放大结果
* ``<tag>_sbs_<variant>_a<α>.mp4``        原图 | 放大 并排（左原右放）
* ``<tag>_roi_<variant>_a<α>.mp4``        活动 ROI 放大裁切（可直接看局部运动）
* ``<tag>_sbs_roi_E0_vs_B1_a<α>.mp4``     E0 放大 | B1 放大 的 ROI 并排（比较两种门控）

真实视频用阶段 1 的同一输入/ROI/频带；合成用局部运动案例（有真值）。
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
from vmm.phase_magnifier import PhaseBasedMagnifier      # noqa: E402
from vmm.reliability import ReliabilityConfig, calibrate_noise   # noqa: E402


def build(mode, calib, fps, f_lo, f_hi, levels, order, **kw):
    cfg = ReliabilityConfig(mode=mode, **kw)
    if mode in ("M1", "M2", "M3", "M4", "B1"):
        cfg.sigma_A = calib["sigma_A"]
        cfg.sigma_phi = calib["sigma_phi"]
    return PhaseBasedMagnifier(fps=fps, f_lo=f_lo, f_hi=f_hi, levels=levels,
                               order=order, channels="gray",
                               reliability_cfg=cfg.resolved())


def crop(video, mask, pad=16):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return video
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, video.shape[1])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, video.shape[2])
    return video[:, y0:y1, x0:x1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="baby.mp4")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=240)
    ap.add_argument("--f-lo", type=float, default=0.2)
    ap.add_argument("--f-hi", type=float, default=0.8)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--alphas", type=float, nargs="+", default=[5.0])
    ap.add_argument("--variants", nargs="+", default=["E0", "B1"])
    ap.add_argument("--synthetic", action="store_true",
                    help="额外生成局部运动合成案例的产物（有真值，便于核对）")
    ap.add_argument("--out-dir", default="outputs/vmm_stage2/artifacts")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    made = []

    def run(tag, video, fps, f_lo, f_hi, mask, gt=None):
        calib = calibrate_noise(video, levels=args.levels, order=args.order,
                               mask=mask if mask is not None and mask.any() else None)
        results = {}
        for name in args.variants:
            mode = {"E0": "M0", "B1": "B1"}.get(name, name)
            for a in args.alphas:
                mag = build(mode, calib, fps, f_lo, f_hi, args.levels, args.order)
                amp, _ = mag.amplify(video, alpha=a)
                p = out / f"{tag}_{name}_a{a:g}.mp4"
                video_io.write_video(amp, p, fps)
                made.append(str(p))
                sbs = out / f"{tag}_sbs_{name}_a{a:g}.mp4"
                video_io.write_side_by_side(video, amp, sbs, fps)
                made.append(str(sbs))
                if mask is not None and mask.any():
                    roi = crop(amp, mask)
                    rp = out / f"{tag}_roi_{name}_a{a:g}.mp4"
                    video_io.write_video(roi, rp, fps)
                    made.append(str(rp))
                results[name] = {"amp": amp, "metrics": {
                    "identity_err_max": metrics.recon_error(video, amp)["max_abs"],
                    "artifacts": metrics.artifact_report(amp)}}
        if len(args.variants) >= 2 and mask is not None and mask.any():
            for a in args.alphas:
                r0 = crop(results[args.variants[0]]["amp"], mask)
                r1 = crop(results[args.variants[1]]["amp"], mask)
                p = out / f"{tag}_sbs_roi_{args.variants[0]}_vs_{args.variants[1]}_a{a:g}.mp4"
                video_io.write_side_by_side(r0, r1, p, fps)
                made.append(str(p))
        return results

    video, meta = video_io.read_video(args.video, max_frames=args.max_frames,
                                      scale=args.scale)
    m = metrics.masks_from_motion(video, active_pct=90, static_pct=40)
    print(f"[real] {args.video} {video.shape} fps={meta.fps} 带={args.f_lo}-{args.f_hi}Hz "
          f"活动={m['active'].sum()} 静止={m['static'].sum()}")
    run("real", video, meta.fps, args.f_lo, args.f_hi, m["active"])

    if args.synthetic:
        fr, gt = synthetic.make_sequence(kind="local", T=96, H=192, W=192, fps=30.0,
                                         amp_px=0.5, freq=0.5, noise_sigma=0.02, seed=42)
        act = gt["active_mask"]
        calib = calibrate_noise(fr, levels=args.levels, order=args.order)
        for name in args.variants:
            mode = {"E0": "M0", "B1": "B1"}.get(name, name)
            for a in args.alphas:
                mag = build(mode, calib, 30.0, 0.3, 1.0, args.levels, args.order)
                amp, _ = mag.amplify(fr, alpha=a)
                p = out / f"syn_local_{name}_a{a:g}.mp4"
                video_io.write_video(amp, p, 30.0)
                made.append(str(p))
                sbs = out / f"syn_local_sbs_{name}_a{a:g}.mp4"
                video_io.write_side_by_side(fr, amp, sbs, 30.0)
                made.append(str(sbs))
                roi = crop(amp, act)
                rp = out / f"syn_local_roi_{name}_a{a:g}.mp4"
                video_io.write_video(roi, rp, 30.0)
                made.append(str(rp))

    (out / "artifacts_manifest.json").write_text(
        json.dumps({"files": made, "variants": args.variants,
                    "alphas": args.alphas, "video": args.video,
                    "band_hz": [args.f_lo, args.f_hi]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] 生成 {len(made)} 个视频 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())