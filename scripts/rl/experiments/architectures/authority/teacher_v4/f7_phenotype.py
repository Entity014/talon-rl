#!/usr/bin/env python3
"""F7 dynamic-stability phenotype map, descriptive (docs/contracts/teacher_v4/teacher-v4-f7-dynamic-stability-semantics-contract.md).

Offline, on existing F6 traces. Per behavior, a vector of physical
dynamic-stability observables (never a weighted sum), computed on 32-step
windows of the steady window (steps 33-128) of surviving traced envs:
rotational RMS |w_xy|, vertical RMS |v_z|, vertical excursion (peak-to-peak of
the detrended integral of v_z; body-frame v_z, approximate), flight fraction
(all four feet off the ground). Tilt is reported as an orientation guardrail,
not a D component. No raw reward term defines any column.
"""
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[6]
TR = REPO / "runs/teacher_v4_f6-2026-09-28/traces"
OUT = REPO / "runs/teacher_v4_f7-2026-09-28"
DT = 0.02
BEHAVIORS = {"s73102 C (L)": ("v4c_g1_2_s73102", "C"), "s73102 T+ (L)": ("v4c_g1_2_s73102", "Tp"),
             "F4 s74102 T+ (H)": ("f4_s74102", "Tp"), "F4 s74103 T+ (H)": ("f4_s74103", "Tp"),
             "M0": ("v4c_g1_2_s73102", "M0"), "standing s73101 C": ("v4c_g1_2_s73101", "C"),
             "F4 s74102 C": ("f4_s74102", "C"), "F4 s74103 C": ("f4_s74103", "C")}


def main():
    out = {}
    for name, (d, k) in BEHAVIORS.items():
        z = np.load(TR / d / "substrate_traces.npz"); c = list(z["cols"])
        X = z[k][32:128][:, z["ok"]]                                        # [96, envs, F]
        W = X.reshape(3, 32, X.shape[1], X.shape[2])                       # 3 windows
        g = lambda col: W[..., c.index(col)]  # noqa: E731
        w, vz, tilt = g("w_xy"), g("v_z"), g("tilt_deg")
        contact = np.stack([g(f) for f in ("c_FL", "c_FR", "c_RL", "c_RR")], -1) > 0.5
        h = np.cumsum(vz * DT, axis=1); t = np.arange(32)[None, :, None]
        slope = ((t - t.mean()) * (h - h.mean(1, keepdims=True))).sum(1, keepdims=True) / ((t - t.mean()) ** 2).sum()
        hd = h - h.mean(1, keepdims=True) - slope * (t - t.mean())
        per_win = {"rot_rms_w_xy": np.sqrt((w ** 2).mean(1)), "vert_rms_v_z": np.sqrt((vz ** 2).mean(1)),
                   "vert_excursion_m": hd.max(1) - hd.min(1), "flight_fraction": (~contact).all(-1).mean(1),
                   "guardrail_tilt_deg": tilt.mean(1)}
        out[name] = {m: {"median": float(np.median(v)), "q10": float(np.quantile(v, .1)), "q90": float(np.quantile(v, .9))} for m, v in per_win.items()}
        out[name]["tl"] = float(X[..., c.index("track_lin_vel_xy_exp")].mean())
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"schema": "teacher_v4_f7_phenotype_v1", "behaviors": out}, open(OUT / "f7_phenotype.json", "w"), indent=1)
    cols = ("tl", "rot_rms_w_xy", "vert_rms_v_z", "vert_excursion_m", "flight_fraction", "guardrail_tilt_deg")
    print("behavior".ljust(20), *[x[:14].rjust(14) for x in cols])
    for n, v in out.items():
        print(n.ljust(20), *[(f"{v[x]:.3f}" if x == "tl" else f"{v[x]['median']:.3f}").rjust(14) for x in cols])


if __name__ == "__main__":
    main()
