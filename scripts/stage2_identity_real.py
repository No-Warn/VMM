#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 2 专项：**真实视频 α=0 恒等**（同通道模式比较）。

为什么需要单独做：`scripts/stage2_ablation.py --real` 为了与合成统一，用的是
``channels="gray"``；此时输出是灰度，而输入 `baby.mp4` 是彩色，
``metrics.recon_error(video, out)`` 拿彩色和灰度直接比会得到 ~0.43 的"恒等误差"——
**这是度量伪影，不是算法问题**（证据：E0 与 E4 的值完全相同 4.317e-01，
说明与阶段 2 的门控无关；而分区**能量比**不受影响，因为 `band_energy_map` 内部
统一按 Rec.709 转灰度，这正解释了 E0 能逐位复现阶段 1 的 1.91/1.42、3.15/2.13、4.60/3.21）。

本脚本用 ``channels=None``（自动：RGB 输入 ⇒ 彩色路径）做同模式比较，输出 JSON 供
`scripts/stage2_accept.py` 读取。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vmm import metrics, video_io                       # noqa: E402
from vmm.phase_magnifier import PhaseBasedMagnifier      # noqa: E402
from vmm.reliability import ReliabilityConfig, calibrate_noise   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="baby.mp4")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=240)
    ap.add_argument("--f-lo", type=float, default=0.2)
    ap.add_argument("--f-hi", type=float, default=0.8)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--out-json", default="outputs/vmm_stage2/identity_real.json")
    args = ap.parse_args()

    video, meta = video_io.read_video(args.video, max_frames=args.max_frames,
                                      scale=args.scale)
    ch = "color" if video.ndim == 4 else "gray"      # 自动：RGB 输入 ⇒ 彩色路径
    m = metrics.masks_from_motion(video, active_pct=90, static_pct=40)
    calib = calibrate_noise(video, levels=args.levels, order=args.order, mask=m["static"])
    out = {"video": args.video, "shape": list(video.shape), "fps": meta.fps,
           "channels": ch, "alpha": 0.0, "variants": {}}
    for name, mode in (("E0", "M0"), ("B1", "B1")):
        cfg = ReliabilityConfig(mode=mode)
        if mode != "M0":
            cfg.sigma_A = calib["sigma_A"]
            cfg.sigma_phi = calib["sigma_phi"]
        mag = PhaseBasedMagnifier(fps=meta.fps, f_lo=args.f_lo, f_hi=args.f_hi,
                                  levels=args.levels, order=args.order, channels=ch,
                                  reliability_cfg=cfg.resolved())
        amp, _ = mag.amplify(video, alpha=0.0)
        err = metrics.recon_error(video, amp)
        out["variants"][name] = {"identity_err_max": err["max_abs"],
                                 "identity_err_mean": err.get("mean_abs"),
                                 "artifacts": metrics.artifact_report(amp),
                                 "out_shape": list(amp.shape)}
        print(f"  {name}: α=0 恒等 max|Δ|={err['max_abs']:.3e}  "
              f"out_shape={amp.shape}  与输入同通道模式 ✓")
    p = Path(args.out_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[save] {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())