#!/usr/bin/env python3
# scripts/rl/assets/generate_a1_leg_length_variants.py
"""Offline utility — generates N leg-length USD variants of the vendored
A1 asset for Isaac Lab's MultiUsdFileCfg spawn-time selection (leg-length
is fixed per env at spawn, not randomized at reset — see
docs/superpowers/specs/2026-09-15-adaptation-module-phase1-design.md for
why Isaac Lab blocks runtime scale-randomization on an Articulation).

Run once, on a machine with Isaac Sim installed:
    python scripts/rl/assets/generate_a1_leg_length_variants.py

For each scale factor s, every leg link (thigh, calf — hip stays fixed,
it's the mount point) gets:
  - geometry scaled by s (xformOp:scale)
  - mass scaled by s^3, diagonal inertia scaled by s^5 (uniform-density
    assumption, standard geometric similarity — NOT ported from any
    specific paper, see the spec's corrected note on the URMA citation)
  - the child joint's local position offset scaled by s (keeps the
    kinematic chain attached at the right point)
  - the chosen scale factor written as a custom "legScale" float attribute
    on the root prim, so an observation function can read back which
    variant a given env got (Isaac Lab doesn't expose "which multi-asset
    variant this env received" as a queryable scene property)
"""

import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402 — after SimulationApp, see repo convention

SCALE_FACTORS = [0.85, 0.925, 1.0, 1.075, 1.15]  # [TBD] placeholder range, not tuned — see spec's Open Questions
LEG_LINK_NAMES = [
    f"{side}_{seg}" for side in ("FR", "FL", "RR", "RL") for seg in ("thigh", "calf")
]

# Vendored source asset (talon_rl/assets/unitree_a1/a1.py:36) — the brief's
# original "unitree_a1.usd" name doesn't exist on disk, the vendored file is "a1.usd".
BASE_USD = "talon_rl/assets/data/Robots/unitree_a1/a1.usd"
OUT_DIR = "talon_rl/assets/data/Robots/unitree_a1"


def generate_variant(scale: float) -> str:
    out_path = os.path.join(OUT_DIR, f"unitree_a1_leg_scale_{scale}.usd")
    stage = Usd.Stage.Open(BASE_USD)
    stage.Export(out_path)  # start from a copy, edit the copy
    stage = Usd.Stage.Open(out_path)

    root_prim = stage.GetDefaultPrim()
    legscale_attr = root_prim.CreateAttribute("legScale", Sdf.ValueTypeNames.Float)
    legscale_attr.Set(scale)

    for link_name in LEG_LINK_NAMES:
        for prim in stage.Traverse():
            if prim.GetName() != link_name:
                continue

            xform = UsdGeom.Xformable(prim)
            # ClearXformOpOrder() only clears the op-order list, not the underlying
            # xformOp:scale attribute spec the vendored asset already has (typed
            # double3) — AddScaleOp()'s default float precision then collides with
            # that existing typeName (pxr.Tf.ErrorException). Match it explicitly.
            existing_scale_attr = prim.GetAttribute("xformOp:scale")
            precision = (
                UsdGeom.XformOp.PrecisionDouble
                if existing_scale_attr and existing_scale_attr.GetTypeName() == Sdf.ValueTypeNames.Double3
                else UsdGeom.XformOp.PrecisionFloat
            )
            xform.ClearXformOpOrder()
            xform.AddScaleOp(precision=precision).Set((scale, scale, scale))

            if not UsdPhysics.MassAPI(prim):
                UsdPhysics.MassAPI.Apply(prim)
            mass_api = UsdPhysics.MassAPI(prim)
            current_mass = mass_api.GetMassAttr().Get() or 1.0
            current_inertia = mass_api.GetDiagonalInertiaAttr().Get()
            mass_api.CreateMassAttr().Set(current_mass * scale**3)
            if current_inertia is not None:
                mass_api.CreateDiagonalInertiaAttr().Set(tuple(v * scale**5 for v in current_inertia))

            for child in prim.GetChildren():
                joint = UsdPhysics.Joint(child)
                if not joint:
                    continue
                local_pos1 = joint.GetLocalPos1Attr().Get()
                if local_pos1 is not None:
                    joint.CreateLocalPos1Attr().Set(tuple(v * scale for v in local_pos1))

    stage.GetRootLayer().Save()
    return out_path


def main() -> None:
    paths = [generate_variant(s) for s in SCALE_FACTORS]
    for p in paths:
        # flush=True: simulation_app.close() can tear the process down before
        # Python's stdout buffer flushes when stdout isn't a tty (redirected to
        # a file/pipe), silently swallowing these lines otherwise.
        print(f"generated: {p}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
