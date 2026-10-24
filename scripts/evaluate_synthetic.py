#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 1 验证套件：α=0 恒等 / 合成亚像素位移的真实运动增益 / 噪声与静止不误放大。

判定纪律（任务书 §9）：
* 阈值写死在 :data:`THRESHOLDS` 里，**运行前**确定，不允许看到结果再改；
* 运动振幅必须**独立测量**（不读算法内部相位），并用**多估计器**交叉验证：
  实测发现没有单一赢家（干净带限序列上单层相位相关准、金字塔粗到细低估 3 倍；
  带噪序列上反过来），因此这里用**真值甄别**可信估计器（合成数据的特权），
  并把全体估计器的分歧作为**不确定度**一并上报；
* 校准相位相关符号后再检验"方向不反转"，且只用**运动轴**的相关（早期版本误用
  x 轴去判竖直运动，会把放大引入的杂散分量当成分数）；
* 无运动输入（纯静止、无噪声）必须输出=输入；接近零运动时**只报绝对误差**；
* 纯噪声输入如实报告噪声被放大的倍数（阶段 1 无可靠性门控，这是阶段 2 的目标）。

用法::

    python scripts/evaluate_synthetic.py --out-json outputs/vmm2/synthetic_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vmm import metrics, synthetic                        # noqa: E402
from vmm.phase_magnifier import PhaseBasedMagnifier       # noqa: E402
from vmm.temporal_filter import bandpass_filter, bandpass_gain_at  # noqa: E402

# ---- 运行前确定的阈值（不得事后修改） ----
THRESHOLDS = {
    "identity_max_abs": 1e-4,      # α=0 输出的最大逐像素偏差
    "static_max_abs": 1e-4,        # 纯静止（无噪声）输入的最大逐像素偏差
    "estimator_rel_err": 0.15,     # 合成数据上判定"可信估计器"的相对误差上限
    "gain_monotone_tol": 0.02,     # 增益随 α 单调（允许的回落容差）
    "gain_at_alpha10_min": 2.0,    # α=10 时实测增益下界
    "waveform_corr_min": 0.8,      # 运动轴上 输出/输入 位移波形相关下界
}


def noise_for(kind: str, args) -> float:
    if kind == "static_noise":
        return float(args.noise_sigma)
    if kind == "shm_h_noisy":
        return float(args.noise_sigma)
    return 0.0


def calibrate(frames, gt, args):
    """用真值甄别可信估计器、选主估计器，并记录全体分歧（=不确定度）。"""
    roi = synthetic.measurement_roi(gt)     # 局部运动用内缩窗口，避开裁切边界伪静止
    allm = metrics.measure_all_estimators(frames, roi=roi)
    waves = allm.pop("_waves")
    gt_amp = synthetic.gt_amplitude(gt)
    axis = 0 if gt_amp[0] >= gt_amp[1] else 1
    gt_a = gt_amp[axis]
    rows = {}
    for k, v in allm.items():
        a = v["amp_x"] if axis == 0 else v["amp_y"]
        # 真值接近零（无运动）时不能算比例误差，只算绝对误差
        err = abs(a - gt_a) / gt_a if gt_a > 0.05 else float("nan")
        abs_err = abs(a - gt_a)
        rows[k] = {"amp": a, "resp": v["resp"], "rel_err_vs_gt": err,
                   "abs_err_vs_gt": abs_err,
                   "trustworthy": bool(err == err and err <= THRESHOLDS["estimator_rel_err"])}
    trust = [k for k, v in rows.items() if v["trustworthy"]]
    if trust:
        primary = min(trust, key=lambda k: rows[k]["rel_err_vs_gt"])
    else:
        primary = min(rows, key=lambda k: rows[k]["abs_err_vs_gt"])
    amps = np.array([v["amp"] for v in rows.values() if v["amp"] == v["amp"]])
    return {"axis": axis, "axis_name": "x" if axis == 0 else "y",
            "gt_amp": gt_amp, "all": rows, "primary": primary,
            "n_trustworthy": len(trust),
            "spread_min": float(amps.min()) if amps.size else float("nan"),
            "spread_max": float(amps.max()) if amps.size else float("nan"),
            "waves": waves, "roi": roi}


def primary_wave(video, roi, primary):
    """按主估计器的键名取位移波形（同时支持 ``pc_<method>_<ref>`` 与 ``lk_first``）。"""
    if primary.startswith("pc_"):
        _, meth, ref = primary.split("_")
        return metrics.displacement_to_reference(video, roi=roi, ref=ref, method=meth)
    if primary == "lk_first":
        # ⚠️ 必须把 roi 传下去：否则 LK 会在整帧上跟踪占多数的静止背景，
        # 局部运动被报成 0.0001 px（这个 bug 让 local 档一度"不通过"）。
        return metrics.lk_displacement_to_reference(video, ref="first", roi=roi)
    raise ValueError(f"未知估计器键：{primary}")


def out_wave(out, roi, primary, axis):
    """输出侧用两条不同的方法测同一个量：主估计器配置 + LK（交叉验证）。

    返回 ``(w_main, w_lk, amp_main_vec, amp_lk_vec)``，振幅都是 ``(ax, ay, resp)``。
    """
    w_main = primary_wave(out, roi, primary)
    w_lk = metrics.lk_displacement_to_reference(out, ref="first", roi=roi)
    amp_main_vec = metrics.waveform_amplitude(w_main)
    amp_lk_vec = metrics.waveform_amplitude(w_lk)
    return w_main, w_lk, amp_main_vec, amp_lk_vec


def run_kind(kind, alphas, args, f_lo, f_hi):
    base_kind = "shm_h" if kind == "shm_h_noisy" else kind
    frames, gt = synthetic.make_sequence(
        kind=base_kind, T=args.frames, H=args.size, W=args.size, fps=args.fps,
        amp_px=args.amp_px, freq=args.freq, noise_sigma=noise_for(kind, args),
        patch=args.size // 3, seed=42)
    cal = calibrate(frames, gt, args)
    axis = cal["axis"]
    in_amp = cal["all"][cal["primary"]]["amp"]
    print(f"\n===== kind={kind}  真值振幅(dx,dy)=({cal['gt_amp'][0]:.3f},{cal['gt_amp'][1]:.3f}) px "
          f"freq={gt['freq']}Hz noise={gt['noise_sigma']} =====")
    print("  输入振幅·各独立估计器（真值甄别，±15%）:")
    for k, v in cal["all"].items():
        mark = " ★主" if k == cal["primary"] else ("  ✓" if v["trustworthy"] else "  ✗")
        print(f"    {k:>16s}: amp_{cal['axis_name']}={v['amp']:.4f} resp={v['resp']:.3f} "
              f"相对误差={v['rel_err_vs_gt']:+.1%}{mark}")
    print(f"    分歧范围（不确定度）: {cal['spread_min']:.4f} ~ {cal['spread_max']:.4f} px，"
          f"可信估计器数={cal['n_trustworthy']}")
    # 无运动输入的"静止区"用整幅；有真值活动区时用补集
    active = gt["active_mask"]
    static = ~active if active.any() else np.ones_like(active)
    rows = []
    for a in alphas:
        mag = PhaseBasedMagnifier(fps=args.fps, f_lo=f_lo, f_hi=f_hi, levels=args.levels,
                                  order=args.order, channels="gray",
                                  phase_clip=args.phase_clip, amp_floor=args.amp_floor)
        out, diag = mag.amplify(frames, alpha=a)
        w_main, w_lk, amp_main_vec, amp_lk_vec = out_wave(out, cal["roi"], cal["primary"], axis)
        ox = amp_main_vec[axis]
        resp_out = amp_main_vec[2]
        gain = ox / in_amp if in_amp > 0.05 else float("nan")
        gain_lk = (amp_lk_vec[axis] / in_amp) if in_amp > 0.05 else float("nan")
        cx, cy = metrics.waveform_correlation(cal["waves"][cal["primary"]], w_main)
        corr_axis = cx if axis == 0 else cy
        e_act = metrics.energy_ratio(frames, out, active)
        e_stat = metrics.energy_ratio(frames, out, static)
        e_stat_abs = metrics.energy_absolute(frames, out, static)
        b_stat = metrics.band_energy_ratio(frames, out, static, args.fps, f_lo, f_hi)
        b_stat_abs = metrics.band_energy_absolute(frames, out, static, args.fps, f_lo, f_hi)
        rows.append({
            "alpha": a, "amp_axis": ox, "amp_main_x": amp_main_vec[0],
            "amp_main_y": amp_main_vec[1], "amp_lk_axis": amp_lk_vec[axis],
            "gain": gain, "gain_lk": gain_lk,
            "gain_over_theory": gain / (1.0 + a) if a > 0 else float("nan"),
            "wave_corr_x": cx, "wave_corr_y": cy, "wave_corr_axis": corr_axis,
            "resp_out": resp_out,
            "recon_err_max": metrics.recon_error(frames, out)["max_abs"],
            "beta_max": _beta_max(diag),
            "energy_active_ratio": e_act, "energy_static_ratio": e_stat,
            "energy_static_abs_in": e_stat_abs[0], "energy_static_abs_out": e_stat_abs[1],
            "band_static_ratio": b_stat,
            "band_static_abs_in": b_stat_abs[0], "band_static_abs_out": b_stat_abs[1],
            "artifact": metrics.artifact_report(out),
        })
        r = rows[-1]
        print(f"  alpha={a:>4g}: amp_{cal['axis_name']}={ox:.4f} (LK={amp_lk_vec[axis]:.4f}) "
              f"增益={gain:6.2f} LK增益={gain_lk:6.2f} 增益/(1+a)={r['gain_over_theory']:5.2f} | "
              f"运动轴相关={corr_axis:+.3f} resp_out={resp_out:.3f} | 能量比 活动={e_act:5.2f} "
              f"静止={e_stat:5.2f} 带内静止={b_stat:5.2f}(绝对 {b_stat_abs[0]:.2e}->{b_stat_abs[1]:.2e}) | "
              f"|out-in|max={r['recon_err_max']:.2e} beta_max={r['beta_max']:.1f}")
    return {"kind": kind, "noise_sigma": gt["noise_sigma"], "gt_amp": cal["gt_amp"],
            "axis": cal["axis"], "axis_name": cal["axis_name"],
            "primary_estimator": cal["primary"], "input_estimators": cal["all"],
            "input_amp_spread": [cal["spread_min"], cal["spread_max"]],
            "n_trustworthy_estimators": cal["n_trustworthy"], "rows": rows}


def _beta_max(diag):
    pb = diag.get("per_band")
    if not pb:
        return float("nan")
    return max(v["beta_max"] for v in pb.values())


def check_direction_convention(args):
    frames, gt = synthetic.make_sequence(kind="shm_h", T=8, H=args.size, W=args.size,
                                         fps=args.fps, amp_px=0.5, freq=args.freq, seed=7)
    s = metrics.displacement_series(frames)
    mid = len(s) // 2
    return {"shift_dx_estimate_at_rising": float(s[mid, 0]),
            "gt_dx_increment_same_pair": float(gt["dx"][mid + 1] - gt["dx"][mid])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--amp-px", type=float, default=0.5)
    ap.add_argument("--freq", type=float, default=0.5)
    ap.add_argument("--noise-sigma", type=float, default=0.02)
    ap.add_argument("--f-lo", type=float, default=0.3)
    ap.add_argument("--f-hi", type=float, default=1.0)
    ap.add_argument("--levels", type=int, default=4)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--phase-clip", type=float, default=None)
    ap.add_argument("--amp-floor", type=float, default=0.0)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0, 1, 2, 5, 10])
    ap.add_argument("--kinds", nargs="+",
                    default=["static", "static_noise", "shm_h", "shm_v", "local",
                             "shm_h_noisy"])
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    print(f"配置: frames={args.frames} size={args.size} fps={args.fps} "
          f"amp={args.amp_px}px freq={args.freq}Hz band={args.f_lo}-{args.f_hi}Hz "
          f"levels={args.levels} order={args.order} phase_clip={args.phase_clip}")
    filt = bandpass_filter(args.fps, args.f_lo, args.f_hi, 2)
    g_mid = bandpass_gain_at(filt, args.freq, args.fps)
    print(f"滤波器在 {args.freq}Hz 的增益 |H(f)| = {g_mid:.3f} → 理论增益 = 1+alpha*{g_mid:.3f}")
    conv = check_direction_convention(args)
    print(f"符号校准: 运动上升沿相邻帧估计={conv['shift_dx_estimate_at_rising']:+.4f}，"
          f"真值同对增量={conv['gt_dx_increment_same_pair']:+.4f}（同号即一致）")

    report = {"config": {k: v for k, v in vars(args).items()},
              "filter_gain_at_freq": g_mid, "sign_convention": conv,
              "thresholds": THRESHOLDS, "results": {}}
    for kind in args.kinds:
        report["results"][kind] = run_kind(kind, args.alphas, args, args.f_lo, args.f_hi)

    print("\n================ 验收判定（阈值运行前已固定）================")
    verdict: dict = {}
    motion_kinds = [k for k in ("shm_h", "shm_v", "local") if k in report["results"]]
    v1 = True
    for kind in motion_kinds:
        r0 = [r for r in report["results"][kind]["rows"] if r["alpha"] == 0]
        if r0:
            ok = r0[0]["recon_err_max"] <= THRESHOLDS["identity_max_abs"]
            v1 &= ok
            print(f"  ① α=0 恒等[{kind}]: {r0[0]['recon_err_max']:.2e} "
                  f"(阈值 {THRESHOLDS['identity_max_abs']:.0e}) {'PASS' if ok else 'FAIL'}")
    verdict["alpha0_identity"] = bool(v1)
    if "static" in report["results"]:
        worst = max(r["recon_err_max"] for r in report["results"]["static"]["rows"])
        ok = worst <= THRESHOLDS["static_max_abs"]
        print(f"  ③ 纯静止(无噪声) 全部 α: |out-in|max={worst:.2e} {'PASS' if ok else 'FAIL'}")
        verdict["static_no_false_amplification"] = bool(ok)
    for kind in motion_kinds:
        rows = sorted(report["results"][kind]["rows"], key=lambda r: r["alpha"])
        gains = [r["gain"] for r in rows]
        mono = all(gains[i + 1] >= gains[i] - THRESHOLDS["gain_monotone_tol"]
                   for i in range(len(gains) - 1))
        g10 = [r["gain"] for r in rows if r["alpha"] == 10]
        big = bool(g10 and g10[0] >= THRESHOLDS["gain_at_alpha10_min"])
        corrs = [r["wave_corr_axis"] for r in rows if r["alpha"] > 0]
        no_rev = len(corrs) > 0 and all(c >= THRESHOLDS["waveform_corr_min"]
                                        for c in corrs if c == c)
        ok = mono and big and no_rev
        print(f"  ② 增益[{kind}](轴={report['results'][kind]['axis_name']}, "
              f"主估计器={report['results'][kind]['primary_estimator']}): "
              f"{['%.2f' % g for g in gains]} 单调={'PASS' if mono else 'FAIL'}"
              f" | α=10={gains[-1]:.2f}(≥{THRESHOLDS['gain_at_alpha10_min']})"
              f"{'PASS' if big else 'FAIL'}"
              f" | 运动轴相关={['%+.2f' % c for c in corrs]}"
              f"（≥{THRESHOLDS['waveform_corr_min']}）{'PASS' if no_rev else 'FAIL'}")
        verdict[f"gain_and_direction_{kind}"] = bool(ok)
    bad = []
    for kind, res in report["results"].items():
        for r in res["rows"]:
            a = r["artifact"]
            if a["has_nan"] or a["has_inf"] or a["black_frame_frac"] > 0:
                bad.append((kind, r["alpha"]))
    print(f"  ⑤ 数值伪影(NaN/Inf/黑帧): {'PASS' if not bad else 'FAIL ' + str(bad)}")
    verdict["no_numeric_artifacts"] = not bad
    verdict["all_pass"] = all(verdict.values())

    print("\n  【附加·适用域边界】增益≥0.7·(1+α) 且 运动轴相关≥0.8 的 α（不参与判定）：")
    valid = {}
    for kind in motion_kinds:
        good = [r["alpha"] for r in sorted(report["results"][kind]["rows"],
                                           key=lambda r: r["alpha"])
                if r["alpha"] > 0 and r["gain"] == r["gain"] and r["gain"] / (1 + r["alpha"]) >= 0.7
                and r["wave_corr_axis"] == r["wave_corr_axis"] and r["wave_corr_axis"] >= 0.8]
        valid[kind] = good
        print(f"    {kind}: {good if good else '无'}")
    report["validity_range"] = valid

    print("\n  【附加·消融：相位上限对 α=10 的影响】（shm_h）")
    frames, gt = synthetic.make_sequence(kind="shm_h", T=args.frames, H=args.size,
                                         W=args.size, fps=args.fps, amp_px=args.amp_px,
                                         freq=args.freq, seed=42)
    cal = calibrate(frames, gt, args)
    in_amp = cal["all"][cal["primary"]]["amp"]
    base_wave = cal["waves"][cal["primary"]]
    ablation = {}
    for clip in (None, 1.0, 0.5, 0.25):
        mag = PhaseBasedMagnifier(fps=args.fps, f_lo=args.f_lo, f_hi=args.f_hi,
                                  levels=args.levels, order=args.order, channels="gray",
                                  phase_clip=clip)
        out, diag = mag.amplify(frames, alpha=10.0)
        w = primary_wave(out, cal["roi"], cal["primary"])
        amp_vec = metrics.waveform_amplitude(w)
        o = amp_vec[cal["axis"]]
        resp = amp_vec[2]
        c, _ = metrics.waveform_correlation(base_wave, w)
        err = metrics.recon_error(frames, out)["max_abs"]
        ablation[str(clip)] = {"amp": o, "gain": o / in_amp, "corr": c,
                               "recon_err_max": err, "resp_out": resp,
                               "beta_max": _beta_max(diag)}
        print(f"    phase_clip={str(clip):>5s}: 增益={o/in_amp:6.2f} 运动轴相关={c:+.3f} "
              f"resp_out={resp:.3f} |out-in|max={err:.3f} beta_max={_beta_max(diag):.1f}")
    report["clip_ablation_alpha10"] = ablation
    report["verdict"] = verdict
    print(f"\n总结: {verdict}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        def enc(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            return str(o)
        Path(args.out_json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=enc), encoding="utf-8")
        print(f"[save] {args.out_json}")
    return 0 if verdict.get("all_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())