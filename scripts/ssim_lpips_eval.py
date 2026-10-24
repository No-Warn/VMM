#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""放大结果的 SSIM / LPIPS 性能测试（Phase H：质量指标）。

⚠️ 先说清"跟谁比"（本仓库的硬性纪律）：
真实视频的**原始帧不是放大结果的真值**——放大后本来就该不同。
所以本脚本给三类互不替代的指标：

1. **对原视频的偏离**（`ssim_orig` / `lpips_orig`）：衡量"改了多少"，
   不是"好不好"；α=0 时必须 ≈1 / ≈0（恒等控制）。
2. **分区**：静止区（算法**不应**改动）与活动区（算法**应该**改动）分开算
   —— 静止区掉分 = 背景被破坏；活动区不动 = 没放大。
3. **时间一致性**：相邻输出帧之间的 SSIM/LPIPS，与输入相邻帧比
   —— 抓闪烁/抖动。
4. **合成真值**（唯一能算"保真度"的地方）：已知位移 A 的序列，
   理想放大 GT = 用 `(1+α)·A` 重新生成的序列；比较 输出 vs GT。

用法::

    python scripts/ssim_lpips_eval.py --video baby.mp4 \
        --out-json outputs/amplify_multi/ssim_lpips.json
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

from vmm import metrics, synthetic, video_io            # noqa: E402
from vmm.phase_magnifier import PhaseBasedMagnifier     # noqa: E402
from vmm.reliability import ReliabilityConfig           # noqa: E402


# --------------------------------------------------------------------------- #
def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐像素 SSIM 图（a/b 为 HWC [0,1]）。"""
    from skimage.metrics import structural_similarity as sk_ssim
    return sk_ssim(a, b, channel_axis=2, data_range=1.0, full=True)[1]


def lpips_batch(fn, a: np.ndarray, b: np.ndarray, device, batch: int = 16) -> list:
    """逐帧 LPIPS（输入 THWC [0,1] → 内部 [-1,1]），分批上 GPU。"""
    import torch
    out = []
    for i in range(0, len(a), batch):
        x = torch.from_numpy(np.ascontiguousarray(a[i:i + batch])).permute(0, 3, 1, 2).float()
        y = torch.from_numpy(np.ascontiguousarray(b[i:i + batch])).permute(0, 3, 1, 2).float()
        with torch.no_grad():
            d = fn((x * 2 - 1).to(device), (y * 2 - 1).to(device))
        out += [float(v) for v in d.flatten().cpu()]
    return out


def crop_bbox(video: np.ndarray, mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return video, mask
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return video[:, y0:y1, x0:x1], mask[y0:y1, x0:x1]


def eval_pair(orig: np.ndarray, out: np.ndarray, active: np.ndarray, static: np.ndarray,
              lpips_fn, device) -> dict:
    """orig/out: THWC float [0,1]。返回 SSIM/LPIPS 的分区与时间一致性指标。"""
    T = len(orig)
    s_all, s_act, s_sta = [], [], []
    for t in range(T):
        m = ssim_map(orig[t], out[t])
        s_all.append(float(m.mean()))
        if active.any():
            s_act.append(float(m[active].mean()))
        if static.any():
            s_sta.append(float(m[static].mean()))
    # LPIPS：全图 + 静止区 bbox（网络需要足够大的输入）
    lp_all = lpips_batch(lpips_fn, orig, out, device)
    if static.any():
        o_st, _ = crop_bbox(orig, static)
        u_st, _ = crop_bbox(out, static)
        if o_st.shape[1] >= 64 and o_st.shape[2] >= 64:
            lp_sta = lpips_batch(lpips_fn, o_st, u_st, device)
        else:
            lp_sta = [float("nan")] * T
    else:
        lp_sta = [float("nan")] * T          # 无静止区（如全局运动）：不报
    # 时间一致性：相邻帧
    t_ssim_o = float(np.mean([float(ssim_map(orig[t], orig[t + 1]).mean())
                              for t in range(T - 1)]))
    t_ssim_u = float(np.mean([float(ssim_map(out[t], out[t + 1]).mean())
                              for t in range(T - 1)]))
    t_lp_o = float(np.mean(lpips_batch(lpips_fn, orig[:-1], orig[1:], device)))
    t_lp_u = float(np.mean(lpips_batch(lpips_fn, out[:-1], out[1:], device)))
    return {"ssim_all": float(np.mean(s_all)),
            "ssim_active": float(np.mean(s_act)) if s_act else None,
            "ssim_static": float(np.mean(s_sta)) if s_sta else None,
            "ssim_all_std": float(np.std(s_all)),
            "lpips_all": float(np.mean(lp_all)),
            "lpips_static": float(np.nanmean(lp_sta)),
            "temporal_ssim_in": t_ssim_o, "temporal_ssim_out": t_ssim_u,
            "temporal_lpips_in": t_lp_o, "temporal_lpips_out": t_lp_u}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="baby.mp4")
    ap.add_argument("--targets", type=float, nargs="+", default=[5, 10, 20, 50])
    ap.add_argument("--f-lo", type=float, default=0.2)
    ap.add_argument("--f-hi", type=float, default=0.8)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=240)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--channels", default="color")
    ap.add_argument("--mode", default="B2")
    ap.add_argument("--lpips-net", default="vgg", choices=("vgg", "alex"))
    ap.add_argument("--synthetic-amp", type=float, default=0.25)
    ap.add_argument("--mp4-dir", default="outputs/amplify_multi",
                    help="已生成的 mp4 目录（可选；给出则额外报告编码后的指标）")
    ap.add_argument("--save-lossless", default=None,
                    help="给出目录则把每档输出写成 FFV1 无损 avi（评估/肉眼核对用）")
    ap.add_argument("--out-json", default="outputs/amplify_multi/ssim_lpips.json")
    args = ap.parse_args()

    import torch
    import lpips as lpips_pkg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = lpips_pkg.LPIPS(net=args.lpips_net, verbose=False).to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    t_start = time.time()
    video, meta = video_io.read_video(args.video, max_frames=args.max_frames,
                                     scale=args.scale)
    m = metrics.masks_from_motion(video, active_pct=90, static_pct=40)
    active, static = m["active"], m["static"]
    print(f"[in] {args.video} {video.shape} fps={meta.fps} | "
          f"active={active.mean():.2%} static={static.mean():.2%}")

    report = {"config": vars(args), "video_shape": list(video.shape), "fps": meta.fps,
              "note": ("ssim/lpips vs 原视频 = '偏离'，不是质量；"
                       "合成真值段才是真保真度"),
              "real_video": {}, "mp4_encoded": {}, "synthetic_gt": {}}

    def cfg():
        return ReliabilityConfig(mode="M0" if args.mode == "none" else args.mode)

    # ---------- 内存中的算法输出（避免 mp4 编码污染）----------
    lossless_dir = Path(args.save_lossless) if args.save_lossless else None
    if lossless_dir:
        lossless_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    for target in [0.0] + list(args.targets):
        alpha = max(float(target) - 1.0, 0.0)
        mag = PhaseBasedMagnifier(fps=meta.fps, f_lo=args.f_lo, f_hi=args.f_hi,
                                  levels=args.levels, order=args.order,
                                  channels=args.channels, reliability_cfg=cfg())
        out, diag = mag.amplify(video, alpha)
        cache[target] = out
        if lossless_dir is not None and target > 0:
            import cv2
            p = lossless_dir / f"baby_x{target:g}_lossless.avi"
            T, H, W, _ = out.shape
            vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"FFV1"),
                                 float(meta.fps), (W, H))
            if vw.isOpened():
                for t in range(T):
                    img = (np.clip(out[t], 0, 1) * 255).round().astype(np.uint8)
                    vw.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                vw.release()
                print(f"[lossless] {p} ({p.stat().st_size/2**20:.1f} MB)")
            else:
                print(f"[lossless] 后端不支持 FFV1：{p}")
        r = eval_pair(video, out, active, static, lpips_fn, device)
        amps = metrics.measure_all_estimators(out, roi=None); amps.pop("_waves", None)
        est = [float(v["amp_x"]) for v in amps.values()]
        r.update({"alpha": alpha, "est_px": est,
                  "est_mean_px": float(np.mean(est)),
                  "seconds": float(diag.get("elapsed_s", float("nan")))})
        report["real_video"][f"x{target:g}"] = r
        print(f"[x{target:g}] alpha={alpha:g} SSIM={r['ssim_all']:.4f} "
              f"(静止 {r['ssim_static']:.4f}) LPIPS={r['lpips_all']:.4f} "
              f"| 位移均值 {r['est_mean_px']:.3f} px | {r['seconds']:.0f}s")
        del out
    del cache

    # ---------- mp4 编码后的指标（交付文件，含编码损失）----------
    mp4_dir = Path(args.mp4_dir)
    for target in args.targets:
        p = mp4_dir / f"baby_x{target:g}.mp4"
        if not p.exists():
            report["mp4_encoded"][f"x{target:g}"] = {"skipped": f"missing {p}"}
            continue
        dec, _ = video_io.read_video(p, max_frames=None, scale=1.0)
        n = min(len(dec), len(video))
        r = eval_pair(video[:n], dec[:n], active, static, lpips_fn, device)
        report["mp4_encoded"][f"x{target:g}"] = {"ssim_all": r["ssim_all"],
                                                 "lpips_all": r["lpips_all"],
                                                 "path": str(p)}
        print(f"[mp4 x{target:g}] SSIM={r['ssim_all']:.4f} LPIPS={r['lpips_all']:.4f}")

    # ---------- 合成真值：唯一能算"保真度"的部分 ----------
    A = float(args.synthetic_amp)
    for kind in ("shm_h", "local"):
        fr, gt = synthetic.make_sequence(kind=kind, T=120, H=192, W=192, fps=meta.fps,
                                         amp_px=A, freq=0.5, patch=64, seed=42)
        roi = synthetic.measurement_roi(gt)
        for target in [1.0] + list(args.targets):
            alpha = max(float(target) - 1.0, 0.0)
            gt_gain = 1.0 + alpha
            gt_video, _ = synthetic.make_sequence(kind=kind, T=120, H=192, W=192,
                                                  fps=meta.fps, amp_px=A * gt_gain,
                                                  freq=0.5, patch=64, seed=42)
            mag = PhaseBasedMagnifier(fps=meta.fps, f_lo=args.f_lo, f_hi=args.f_hi,
                                      levels=args.levels, order=args.order,
                                      channels=args.channels, reliability_cfg=cfg())
            out, _ = mag.amplify(fr, alpha)
            act = gt["active_mask"] if gt["active_mask"].any() else np.ones(fr.shape[1:3], bool)
            sta = ~act if act.all() else (~act) & np.ones_like(act)
            if sta.sum() == 0:
                sta = np.zeros_like(act)
            r = eval_pair(gt_video, out, act, sta, lpips_fn, device)
            wi = metrics.lk_displacement_to_reference(fr, ref="first", roi=roi)
            wo = metrics.lk_displacement_to_reference(out, ref="first", roi=roi)
            gta = synthetic.gt_amplitude(gt)
            axis = 0 if gta[0] >= gta[1] else 1
            a_in = metrics.waveform_amplitude(wi)[axis]
            a_out = metrics.waveform_amplitude(wo)[axis]
            r.update({"alpha": alpha, "gt_gain": gt_gain, "gt_amp_px": A * gt_gain,
                      "measured_gain": a_out / a_in if a_in > 0.05 else None})
            report["synthetic_gt"][f"{kind}_x{target:g}"] = r
            print(f"[syn {kind} x{target:g}] vs GT: SSIM={r['ssim_all']:.4f} "
                  f"LPIPS={r['lpips_all']:.4f} | 实测增益 "
                  f"{r['measured_gain'] if r['measured_gain'] else float('nan'):.2f} "
                  f"(请求 {gt_gain:g})")

    report["elapsed_s"] = time.time() - t_start
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                              default=str), encoding="utf-8")
    print(f"\n[save] {args.out_json}  ({report['elapsed_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
