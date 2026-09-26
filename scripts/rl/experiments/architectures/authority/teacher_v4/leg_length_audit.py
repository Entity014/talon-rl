#!/usr/bin/env python3
"""V4-B1 audit of the e_t leg_length channel: USD variants, per-env spawn, legScale lookup and physical leg geometry.

V4-B saw e_t[3] == 1.0 on all 64 envs. This traces the chain end to end:
what each of the 5 USD variants contains, which variant each env references,
where legScale lives on the live stage versus where leg_length_extrinsic
looks, and what PhysX actually simulates (calf mass, thigh-to-foot distance).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # scripts/

import torch

from rl.core.diagnostics.isaac_audit import REPO, IsaacAudit

VARIANT_DIR = REPO / "talon_rl/assets/data/Robots/unitree_a1"
SCALES = (0.85, 0.925, 1.0, 1.075, 1.15)


def _usd_variant(path: Path) -> dict:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    root = stage.GetDefaultPrim()
    attr = root.GetAttribute("legScale")
    out = {"default_prim": str(root.GetPath()), "legScale": attr.Get() if attr.IsValid() else None}
    for prim in stage.Traverse():
        if prim.GetName() == "FR_calf":
            s = prim.GetAttribute("xformOp:scale")
            out["FR_calf_scale"] = list(s.Get()) if s.IsValid() and s.Get() is not None else None
            m = UsdPhysics.MassAPI(prim).GetMassAttr()
            out["FR_calf_mass"] = m.Get() if m else None
            out["FR_calf_children"] = [c.GetName() for c in prim.GetChildren()]
        j = UsdPhysics.Joint(prim)
        if j and prim.GetName() in ("FR_calf_joint", "FR_foot_fixed", "FR_thigh_joint"):
            out[f"{prim.GetName()}_parent"] = str(prim.GetParent().GetPath())
            out[f"{prim.GetName()}_localPos0"] = list(j.GetLocalPos0Attr().Get() or ())
            out[f"{prim.GetName()}_localPos1"] = list(j.GetLocalPos1Attr().Get() or ())
    return out


class LegLengthAudit(IsaacAudit):
    """V4-B1 leg_length channel audit."""
    task = "Isaac-Talon-A1-v0"
    run = "teacher_v4_b1_leg_length_audit-2026-09-26"
    report = "report.json"
    num_envs = 64
    require_diversity = True  # canonical env has replicate_physics=False since 2026-09-26

    def build_env(self):
        import gymnasium as gym
        import talon_rl.tasks.locomotion.a1_env  # noqa: F401
        from talon_rl.tasks.locomotion.a1_env.a1_env_cfg import IsaacLabTalonEnvCfg

        cfg = IsaacLabTalonEnvCfg()
        cfg.scene.num_envs = self.num_envs
        cfg.seed = self.seed
        self.cfg = cfg
        env = gym.make(self.task, cfg=cfg).unwrapped
        return env, env.reset()

    def rollout(self, env, tr) -> dict:
        import carb
        from talon_rl.tasks.locomotion.a1_env import mdp

        out: dict = {"variants": {str(s): _usd_variant(VARIANT_DIR / f"unitree_a1_leg_scale_{s}.usd") for s in SCALES}}
        out["scene_replicate_physics"] = bool(env.scene.cfg.replicate_physics)
        out["carb_multi_assets_flag"] = carb.settings.get_settings().get("/isaaclab/spawn/multi_assets")

        asset = env.scene["robot"]
        stage = env.sim.stage
        names = asset.body_names
        out["body_names"] = names
        rows = []
        masses = asset.root_physx_view.get_masses().cpu()
        pos = asset.data.body_pos_w.cpu()
        calf, thigh = names.index("FR_calf"), names.index("FR_thigh")
        foot = names.index("FR_foot") if "FR_foot" in names else None
        reported = mdp.leg_length_extrinsic(env).squeeze(-1).cpu()
        for i in range(env.num_envs):
            root_path = asset._root_physx_view.prim_paths[i]  # noqa: SLF001
            # walk up from the articulation root until legScale is found
            found_at, found = None, None
            p = stage.GetPrimAtPath(root_path)
            while p and p.IsValid() and str(p.GetPath()) != "/":
                a = p.GetAttribute("legScale")
                if a.IsValid() and a.Get() is not None:
                    found_at, found = str(p.GetPath()), float(a.Get()); break
                p = p.GetParent()
            robot = stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot")
            refs = sorted({Path(spec.layer.identifier).name for spec in robot.GetPrimStack()}) if robot.IsValid() else []
            rows.append({
                "env": i, "root_physx_path": root_path, "legScale_found_at": found_at, "legScale": found,
                "referenced_layers": refs, "reported_e_t_leg_length": float(reported[i]),
                "FR_calf_mass_physx": float(masses[i, calf]),
                "FR_thigh_to_calf_m": float((pos[i, calf] - pos[i, thigh]).norm()),
                "FR_calf_to_foot_m": float((pos[i, foot] - pos[i, calf]).norm()) if foot is not None else None,
            })
        out["envs"] = rows
        out["summary"] = {
            "reported_values": dict(Counter(round(r["reported_e_t_leg_length"], 4) for r in rows)),
            "legScale_on_stage": dict(Counter(r["legScale"] for r in rows)),
            "legScale_found_at_example": rows[0]["legScale_found_at"],
            "root_physx_path_example": rows[0]["root_physx_path"],
            "referenced_variant_files": dict(Counter(tuple(x for x in r["referenced_layers"] if "leg_scale" in x) for r in rows).most_common()),
            "FR_calf_mass_physx": dict(Counter(round(r["FR_calf_mass_physx"], 4) for r in rows)),
            "FR_thigh_to_calf_m": dict(Counter(round(r["FR_thigh_to_calf_m"], 4) for r in rows)),
            "FR_calf_to_foot_m": dict(Counter(round(r["FR_calf_to_foot_m"], 4) for r in rows if r["FR_calf_to_foot_m"] is not None)),
        }
        out["summary"]["referenced_variant_files"] = {str(k): v for k, v in out["summary"]["referenced_variant_files"].items()}

        def file_scale(r):
            files = [x for x in r["referenced_layers"] if "leg_scale" in x]
            return float(files[0].removeprefix("unitree_a1_leg_scale_").removesuffix(".usd")) if len(files) == 1 else None

        # Regression gate: e_t must report the variant PhysX actually simulates, per env.
        checks = {
            "one_variant_file_per_env": all(file_scale(r) is not None for r in rows),
            "reported_matches_stage_legScale": all(r["legScale"] is not None and abs(r["reported_e_t_leg_length"] - r["legScale"]) < 1e-5 for r in rows),
            "reported_matches_variant_file": all(file_scale(r) is not None and abs(r["reported_e_t_leg_length"] - file_scale(r)) < 1e-5 for r in rows),
            "physx_thigh_length_matches_reported": all(abs(r["FR_thigh_to_calf_m"] - 0.2 * r["reported_e_t_leg_length"]) < 1e-3 for r in rows),
        }
        # Diversity needs replicate_physics=False; reported, gated only when asked.
        out["unique_variants"] = len({round(r["reported_e_t_leg_length"], 4) for r in rows})
        if self.require_diversity:
            checks["more_than_one_variant"] = out["unique_variants"] > 1
        out["checks"] = checks
        out["pass"] = all(checks.values())
        self.write(out)
        print(out["summary"], checks, "PASS" if out["pass"] else "FAIL", flush=True)
        if not out["pass"]:
            raise SystemExit(1)
        return out


if __name__ == "__main__":
    LegLengthAudit.main()
