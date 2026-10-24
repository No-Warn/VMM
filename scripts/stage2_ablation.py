#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 2 消融实验：门控（M0–M3）× 逐带增益上限 × 放大中去噪，对 α∈{0,1,2,5,10}。

同一输入、同一频带、同一 ROI、同一指标定义（任务书 §6 的硬性要求）。
所有测量沿用阶段 1 已验收的 `vmm/metrics.py`（5 估计器 + 真值甄别）。

指标（任务书 §7）：
* 运动保留：输入/输出振幅、增益、波形相关、与理论 1+α 的偏差（合成，有真值）。
* 静止区噪声：同频带时间能量 in/out/比值，以及**带外**（guard 带）能量比 ⇒ 是否引入新频率成分。
* 空间渗漏：活动区 / 边界环 / 远背景 的分区带内能量比（局部运动案例）。
* 重建质量：α=0 恒等误差、NaN/Inf/黑帧/饱和、耗时。
* 门控诊断：逐带 w 均值、被抑制比例（w<0.1）、α_eff、β 上限。

用法::

    python scripts/stage2_ablation.py --out-json outputs/vmm_stage2/ablation_A.json \
        --variants E0 E1 M2 M3 --cases shm_h local local_noise static_noise --alphas 0 1 5
    python scripts/stage2_ablation.py --real --variants E0 E1 --alphas 0 2 5 \
        --out-json outputs/vmm_stage2/ablation_real.json
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

from vmm import metrics, synthetic                        # noqa: E402
from vmm.phase_magnifier import PhaseBasedMagnifier        # noqa: E402
from vmm.reliability import (ReliabilityConfig, calibrate_noise,   # noqa: E402
                             cap_suggestions, load_calibration)

CASES = {
    # name: (kind, noise_sigma, low_contrast)
    "shm_h":            ("shm_h", 0.0, False),
    "shm_v":            ("shm_v", 0.0, False),
    "local":            ("local", 0.0, False),
    "local_lowc":       ("local", 0.0, True),      # 低对比度真实运动
    "local_noise":      ("local", 0.02, False),    # 不同噪声强度
    "local_noise_hi":   ("local", 0.05, False),
    "static":           ("static", 0.0, False),
    "static_noise":     ("static_noise", 0.02, False),
}


def build_calibration(kind, noise, size, T, fps, levels, order, tag, base_seed=42):
    """独立标定：**同一底图**（同 base_seed）+ **不同噪声实现**（noise_seed）。

    ⚠️ 实测教训：若标定视频的底图与测试视频不同，σ_A 的幅值尺度会对不上，
    导致 M1（纯幅值门控）完全失效（相位证据 σ_φ 不受影响，因为相位是尺度无关的）。
    """
    cal_kind = "static_noise" if noise > 0 else "static"
    fr, _ = synthetic.make_sequence(kind=cal_kind, T=T, H=size, W=size, fps=fps,
                                    noise_sigma=noise, seed=base_seed,
                                    noise_seed=99991 + (abs(hash(tag)) % 1000))
    return calibrate_noise(fr, levels=levels, order=order)


def region_metrics(video_in, video_out, fps, f_lo, f_hi, masks: dict, guard=(3.0, 8.0)):
    """分区带内能量比 + 带外（guard）能量比。"""
    out = {}
    for name, m in masks.items():
        if m is None or not m.any():
            out[name] = {"band_in": float("nan"), "band_out": float("nan"), "ratio": float("nan")}
            continue
        b_in = float(metrics.band_energy_map(video_in, fps, f_lo, f_hi)[m].mean())
        b_out = float(metrics.band_energy_map(video_out, fps, f_lo, f_hi)[m].mean())
        out[name] = {"band_in": b_in, "band_out": b_out,
                     "ratio": (b_out / b_in) if b_in > metrics.ENERGY_FLOOR else float("nan")}
    if "static" in masks and masks["static"] is not None and masks["static"].any():
        m = masks["static"]
        g_in = float(metrics.band_energy_map(video_in, fps, *guard)[m].mean())
        g_out = float(metrics.band_energy_map(video_out, fps, *guard)[m].mean())
        out["guard_out_of_band"] = {"band_in": g_in, "band_out": g_out,
                                    "ratio": (g_out / g_in) if g_in > metrics.ENERGY_FLOOR
                                    else float("nan")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+",
                    default=["E0", "E1", "M2", "M3", "E2", "E3", "E4"])
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument("--alphas", type=float, nargs="+", default=[0, 1, 2, 5, 10])
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--amp-px", type=float, default=0.5)
    ap.add_argument("--freq", type=float, default=0.5)
    ap.add_argument("--f-lo", type=float, default=0.3)
    ap.add_argument("--f-hi", type=float, default=1.0)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--caps-json", default=None, help="逐带上限 β（JSON）；给出则 E2/E3/E4 使用")
    ap.add_argument("--caps-strong-json", default=None,
                    help="第二组（更激进）上限，用于 C1/C2 消融点；必须也是从统计/消融得来")
    ap.add_argument("--real", action="store_true", help="改用真实视频（阶段1 同输入/ROI/频带）")
    ap.add_argument("--video", default="baby.mp4")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--max-frames", type=int, default=240)
    ap.add_argument("--real-f-lo", type=float, default=0.2)
    ap.add_argument("--real-f-hi", type=float, default=0.8)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    caps = None
    if args.caps_json:
        caps = {tuple(int(x) for x in k.split("-")): float(v)
                for k, v in json.loads(Path(args.caps_json).read_text()).items()}
    caps_strong = None
    if args.caps_strong_json:
        caps_strong = {tuple(int(x) for x in k.split("-")): float(v)
                       for k, v in json.loads(Path(args.caps_strong_json).read_text()).items()}

    def make_variant(name, calib, fps, f_lo, f_hi):
        base = dict(fps=fps, f_lo=f_lo, f_hi=f_hi, levels=args.levels, order=args.order,
                    channels="gray")
        cfg = ReliabilityConfig(mode="M0")
        D = {"E0": dict(mode="M0"), "E1": dict(mode="M1"), "M2": dict(mode="M2"),
             "M3": dict(mode="M3"), "E2": dict(mode="M0", caps=True),
             "E3": dict(mode="M1", caps=True),
             "E4": dict(mode="B1", caps=True, temporal_robust=5),
             "C1": dict(mode="M0", caps_strong=True),
             "C2": dict(mode="M2", caps_strong=True),
             # 第二轮新增：长时窗带内谱集中度证据（B1/M4）与 §2.3/§4 的旋钮消融
             "B1": dict(mode="B1"),
             # 第三轮新增：**归一化**谱集中度（未闭环 #1 的修复候选）
             "B2": dict(mode="B2"),                        # scalar 归一化（bp 域）
             "B2r": dict(mode="B2", band_domain="raw"),    # scalar 归一化（raw 域）
             "B3": dict(mode="B3"),                        # 逐像素归一化（bp 域）
             "B3r": dict(mode="B3", band_domain="raw"),    # 逐像素归一化（raw 域）
             "B2a": dict(mode="B2", amp=True),             # 额外叠加幅值证据 w_A
             "M4": dict(mode="M4"),
             "M4s1": dict(mode="M4", spatial_smooth=1.0),
             "M4s2": dict(mode="M4", spatial_smooth=2.0),
             "M4t5": dict(mode="M4", temporal_smooth=5),
             "B1f01": dict(mode="B1", amp_floor=0.1),
             "B1f03": dict(mode="B1", amp_floor=0.3),
             "B1f10": dict(mode="B1", amp_floor=1.0),
             "B1s1": dict(mode="B1", spatial_smooth=1.0),
             # §5 D0–D3：确定性去噪对比（D0=E0 无去噪；D1=仅门控；D2=+稳健时间抑噪；D3=+局部空间一致性）
             "D0": dict(mode="M0"),
             "D1": dict(mode="B1"),
             "D2": dict(mode="B1", temporal_robust=5),
             "D2m": dict(mode="B1", temporal_robust=9),
             "D3": dict(mode="B1", temporal_robust=5, spatial_smooth=1.0),
             "D3b": dict(mode="B1", temporal_robust=5, spatial_smooth=2.0),
             "D1s": dict(mode="B1", spatial_smooth=1.0)}[name]
        cfg.mode = D["mode"]
        if cfg.mode in ("M1", "M2", "M3", "M4", "B1"):
            cfg.sigma_A = calib["sigma_A"]
            cfg.sigma_phi = calib["sigma_phi"]
        if D.get("amp"):
            cfg.sigma_A = calib["sigma_A"]
        if D.get("band_domain"):
            cfg.band_domain = str(D["band_domain"])
        if D.get("spatial_smooth"):
            cfg.spatial_smooth = float(D["spatial_smooth"])
        if D.get("temporal_smooth"):
            cfg.temporal_smooth = int(D["temporal_smooth"])
        if D.get("amp_floor"):
            cfg.amp_floor = float(D["amp_floor"])
        if D.get("temporal_robust"):
            cfg.temporal_robust = int(D["temporal_robust"])
        if D.get("caps"):
            cfg.per_band_caps = caps
        if D.get("caps_strong"):
            cfg.per_band_caps = caps_strong
        if D.get("temporal_smooth"):
            cfg.temporal_smooth = D["temporal_smooth"]
        return ReliabilityConfig(**cfg.__dict__).resolved(), base

    report = {"config": vars(args), "variants": {}, }
    t_start = time.time()
    # §5 要求：所有新增滤波必须记录频率响应与边界处理（稳健时间抑噪的等效频响）
    from vmm.reliability import robust_temporal_gain
    probe_freqs = [0.1, 0.2, 0.5, 0.8, 1.5, 3.0, 8.0, 12.0]
    report["robust_temporal_frequency_response"] = {
        f"median_k{k}": robust_temporal_gain(30.0, k, probe_freqs, mode="median")
        for k in (5, 9)}
    report["boundary_handling"] = {
        "robust_temporal_smooth": "scipy.ndimage.median_filter(size=(k,1,1), mode='nearest')"
                                  " —— 两端复制，无边界振荡；非线性，非 LTI",
        "bandpass_time": "现有阶段 1 零相位 Butterworth（filtfilt），未改动、未重复叠加",
        "smooth_weight": "空间高斯 + 时间对称均值，均 mode='reflect'/'nearest'"}
    print("[freq] 稳健时间抑噪等效频响（正弦中段幅度比）:")
    for k, v in report["robust_temporal_frequency_response"].items():
        print("   " + k + ": " + ", ".join(f"{f}Hz→{g:.3f}" for f, g in v.items()))

    # ================= 真实视频 =================
    if args.real:
        from vmm import video_io
        video, meta = video_io.read_video(args.video, max_frames=args.max_frames,
                                          scale=args.scale)
        fps, f_lo, f_hi = meta.fps, args.real_f_lo, args.real_f_hi
        m = metrics.masks_from_motion(video, active_pct=90, static_pct=40)
        masks = {"active": m["active"], "static": m["static"]}
        calib = calibrate_noise(video, levels=args.levels, order=args.order,
                                mask=m["static"])
        print(f"[real] {args.video} {video.shape[0]} 帧 {video.shape[2]}x{video.shape[1]} "
              f"band={f_lo}-{f_hi}Hz levels={args.levels}")
        print("[real] 静止 ROI 标定 σ_A: " +
              ", ".join(f"{a}-{b}:{v:.4f}" for (a, b), v in sorted(calib["sigma_A"].items())))
        for name in args.variants:
            cfg, base = make_variant(name, calib, fps, f_lo, f_hi)
            rows = []
            for a in args.alphas:
                mag = PhaseBasedMagnifier(reliability_cfg=cfg, **base)
                t0 = time.time()
                out, diag = mag.amplify(video, alpha=a)
                dt = time.time() - t0
                rm = region_metrics(video, out, fps, f_lo, f_hi, masks)
                art = metrics.artifact_report(out)
                err = metrics.recon_error(video, out)["max_abs"]
                # 真实视频无真值：只报分区能量比 + 估计器分歧（不声称精确增益）
                amps = metrics.measure_all_estimators(out, roi=None)
                amps.pop("_waves")
                est = {k: v["amp_x"] for k, v in amps.items()}
                rows.append({"alpha": a, "regions": rm, "artifacts": art,
                             "identity_err_max": err, "seconds": dt,
                             "est_disp_px": est,
                             "est_spread": [float(min(est.values())), float(max(est.values()))],
                             "gate": diag.get("per_band")})
                print(f"  {name:>3} α={a:>4g}: 活动带内×{rm['active']['ratio']:.2f} "
                      f"静止带内×{rm['static']['ratio']:.2f} "
                      f"带外×{rm['guard_out_of_band']['ratio']:.2f} "
                      f"| 位移估计分歧 {min(est.values()):.4f}~{max(est.values()):.4f}px "
                      f"| {dt:.1f}s")
            report["variants"][name] = {"rows": rows, "config": cfg.to_dict()}
        report["calibration"] = {"sigma_A": {f"{a}-{b}": v for (a, b), v in calib["sigma_A"].items()},
                                 "band_stats": {f"{a}-{b}": v for (a, b), v in calib["band_stats"].items()}}
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                  default=str), encoding="utf-8")
        print(f"[save] {args.out_json}  ({time.time()-t_start:.0f}s)")
        return 0

    # ================= 合成 =================
    for case in args.cases:
        kind, noise, lowc = CASES[case]
        fr, gt = synthetic.make_sequence(kind=kind, T=args.T, H=args.size, W=args.size,
                                         fps=args.fps, amp_px=args.amp_px,
                                         freq=args.freq, noise_sigma=noise,
                                         patch=args.size // 3, seed=42)
        if lowc:
            fr = np.clip(0.5 + 0.15 * (fr - 0.5), 0, 1)     # 低对比度（运动不变）
        roi = synthetic.measurement_roi(gt)
        active = gt["active_mask"]
        masks = {"active": active if active.any() else None,
                 "static": ~active if active.any() else np.ones_like(active)}
        if active.any():
            ring, far = metrics.ring_and_far_masks(active, ring_px=10, far_px=30)
            masks["ring"], masks["far"] = ring, far
        calib = build_calibration(kind, noise, args.size, args.T, args.fps,
                                  args.levels, args.order, case)
        gt_amp = synthetic.gt_amplitude(gt)
        axis = 0 if gt_amp[0] >= gt_amp[1] else 1
        in_wave = metrics.lk_displacement_to_reference(fr, ref="first", roi=roi)
        in_amp_vec = metrics.waveform_amplitude(in_wave)
        in_amp = in_amp_vec[axis]
        print(f"\n===== case={case} kind={kind} noise={noise} lowc={lowc} "
              f"真值振幅={gt_amp[axis]:.3f}px 输入估计={in_amp:.4f}px =====")
        report["variants"].setdefault("_cases", {})[case] = {
            "gt_amp_px": gt_amp[axis], "input_amp_px": in_amp,
            "sigma_A": {f"{a}-{b}": v for (a, b), v in calib["sigma_A"].items()},
            "suggested_caps": {f"{a}-{b}": v for (a, b), v in
                               cap_suggestions(calib["band_stats"], alpha_max=10.0).items()},
            "results": {}}
        for name in args.variants:
            cfg, base = make_variant(name, calib, args.fps, args.f_lo, args.f_hi)
            rows = []
            for a in args.alphas:
                mag = PhaseBasedMagnifier(reliability_cfg=cfg, **base)
                t0 = time.time()
                out, diag = mag.amplify(fr, alpha=a)
                dt = time.time() - t0
                o_wave = metrics.lk_displacement_to_reference(out, ref="first", roi=roi)
                o_amp_vec = metrics.waveform_amplitude(o_wave)
                o_amp = o_amp_vec[axis]
                corr = metrics.waveform_correlation(in_wave, o_wave)
                c = corr[axis]
                rm = region_metrics(fr, out, args.fps, args.f_lo, args.f_hi, masks)
                rows.append({
                    "alpha": a, "out_amp_px": o_amp,
                    "gain": (o_amp / in_amp) if in_amp > 0.05 else float("nan"),
                    "gain_over_theory": (o_amp / in_amp) / (1 + a) if (a > 0 and in_amp > 0.05) else float("nan"),
                    "wave_corr": c, "regions": rm,
                    "identity_err_max": metrics.recon_error(fr, out)["max_abs"],
                    "false_disp_px": (o_amp if not gt["has_motion"] else None),
                    "artifacts": metrics.artifact_report(out), "seconds": dt,
                    "gate": diag.get("per_band")})
                r = rows[-1]
                extra = ""
                if "ring" in rm:
                    extra = (f" 环×{rm['ring']['ratio']:.2f} 远背景×{rm['far']['ratio']:.2f}")
                print(f"  {name:>3} α={a:>4g}: 振幅={o_amp:.4f} 增益={r['gain']:6.2f} "
                      f"增益/(1+α)={r['gain_over_theory']:5.2f} 相关={c:+.3f} | "
                      f"活动带内×{rm['active']['ratio']:.2f} 静止带内×{rm['static']['ratio']:.2f}"
                      f"{extra} | 带外×{rm.get('guard_out_of_band', {}).get('ratio', float('nan')):.2f}"
                      f" | 恒等err={r['identity_err_max']:.2e} {dt:.1f}s")
            report["variants"]["_cases"][case]["results"][name] = {
                "rows": rows, "config": cfg.to_dict()}

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                              default=str), encoding="utf-8")
    print(f"\n[save] {args.out_json}  (总耗时 {time.time()-t_start:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())