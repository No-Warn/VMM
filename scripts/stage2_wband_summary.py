#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 `wband_fix_*.json`：B1 的 T 依赖 / B2 归一化是否消除它。"""
import json
import sys
from pathlib import Path

import numpy as np


def rows(path, tag):
    d = json.loads(Path(path).read_text())
    cfg = d["config"]
    out = []
    for case, c in d["variants"].get("_cases", {}).items():
        for v, res in c["results"].items():
            for r in res["rows"]:
                g = r.get("gate") or {}
                wb = [vv.get("w_band_mean") for vv in g.values()
                      if isinstance(vv, dict) and vv.get("w_band_mean") is not None]
                out.append({
                    "tag": tag, "case": case, "variant": v, "alpha": r["alpha"],
                    "T": cfg.get("T"), "gain": r.get("gain"),
                    "gain_norm": r.get("gain_over_theory"),
                    "corr": r.get("wave_corr"),
                    "w_band": float(np.mean(wb)) if wb else None,
                    "active": (r.get("regions", {}).get("active") or {}).get("ratio"),
                    "static": (r.get("regions", {}).get("static") or {}).get("ratio"),
                    "ring": (r.get("regions", {}).get("ring") or {}).get("ratio"),
                    "far": (r.get("regions", {}).get("far") or {}).get("ratio"),
                    "guard": (r.get("regions", {}).get("guard_out_of_band") or {}).get("ratio"),
                })
    return out


def fmt(x, n=3):
    return "   nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:{n}.{n}f}".rjust(6)


def main():
    files = [("outputs/vmm_stage2/wband_fix_T64.json", "T64"),
             ("outputs/vmm_stage2/wband_fix_T120.json", "T120"),
             ("outputs/vmm_stage2/wband_fix_T240.json", "T240")]
    allrows = []
    for p, tag in files:
        if Path(p).exists():
            allrows += rows(p, tag)
    print("== 合成：增益 / w_band（α=5）==")
    print(f"{'case':<12}{'var':<5}{'T':>5}{'gain':>7}{'g/(1+a)':>8}{'corr':>7}{'w_band':>8}"
          f"{'active':>7}{'static':>7}{'ring':>7}{'far':>7}{'guard':>7}")
    cases = sorted({r["case"] for r in allrows})
    for case in cases:
        for tag in ("T64", "T120", "T240"):
            for v in ("E0", "B1", "B2", "B2r", "B3"):
                sel = [r for r in allrows if r["case"] == case and r["tag"] == tag and r["variant"] == v]
                for r in sel:
                    print(f"{case:<12}{v:<5}{r['T']:>5}{fmt(r['gain'])}{fmt(r['gain_norm'])}"
                          f"{fmt(r['corr'])}{fmt(r['w_band'])}{fmt(r['active'])}{fmt(r['static'])}"
                          f"{fmt(r['ring'])}{fmt(r['far'])}{fmt(r['guard'])}")
    # 真实视频
    p = Path("outputs/vmm_stage2/wband_fix_real.json")
    if p.exists():
        d = json.loads(p.read_text())
        print("\n== 真实视频（baby.mp4）：活动/静止带内能量比 ==")
        print(f"{'alpha':>6}{'var':>5}{'active':>8}{'static':>8}{'guard':>7}{'select':>8}{'sec':>6}")
        for v, res in d["variants"].items():
            for r in res["rows"]:
                a = r["regions"]["active"]["ratio"]; s = r["regions"]["static"]["ratio"]
                print(f"{r['alpha']:>6g}{v:>5}{fmt(a)}{fmt(s)}"
                      f"{fmt(r['regions']['guard_out_of_band']['ratio'])}"
                      f"{fmt(a / s if s else float('nan'))}{fmt(r.get('seconds'))}")


if __name__ == "__main__":
    main()
