#!/usr/bin/env python3
"""Every plant parameter the MuJoCo A1 scene declares, as one record.

Read-only. This is the MuJoCo half of the plant-equivalence comparison, so it
reports what the model says rather than what a rollout does.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.offline_audit import REPO, OfflineAudit

SCENE = REPO / "talon_rl/assets/data/Robots/unitree_a1/mujoco/scene.xml"


class MujocoPlantSnapshot(OfflineAudit):
    """Every plant parameter the MuJoCo A1 scene declares."""

    run = "phase3_plant_equivalence_audit"
    report = "mujoco_snapshot.json"
    schema = "phase3_plant_mujoco_snapshot_v1"

    def analyze(self):
        import mujoco

        m = mujoco.MjModel.from_xml_path(str(SCENE))
        name = lambda kind, i: mujoco.mj_id2name(m, kind, i)  # noqa: E731

        bodies = [(name(mujoco.mjtObj.mjOBJ_BODY, i), i) for i in range(1, m.nbody)]
        # the free joint carries no damping/armature/limits, so it is skipped
        joints = [(name(mujoco.mjtObj.mjOBJ_JOINT, j), j) for j in range(m.njnt)
                  if int(m.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE)]
        geoms = [{"name": name(mujoco.mjtObj.mjOBJ_GEOM, g),
                  "body": name(mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])),
                  "friction": m.geom_friction[g].tolist(),
                  "solref": m.geom_solref[g].tolist(),
                  "solimp": m.geom_solimp[g].tolist(),
                  "condim": int(m.geom_condim[g])}
                 for g in range(m.ngeom)]

        return {
            "schema": self.schema,
            "scene": str(SCENE.relative_to(REPO)),
            "sim_dt": float(m.opt.timestep),
            "integrator": int(m.opt.integrator),
            "solver": int(m.opt.solver),
            "iterations": int(m.opt.iterations),
            "ls_iterations": int(m.opt.ls_iterations),
            "body_names": [n for n, _ in bodies],
            "mass": [float(m.body_mass[i]) for _, i in bodies],
            "body_ipos": [m.body_ipos[i].tolist() for _, i in bodies],
            "body_iquat": [m.body_iquat[i].tolist() for _, i in bodies],
            "body_inertia_diagonal": [m.body_inertia[i].tolist() for _, i in bodies],
            "joint_names": [n for n, _ in joints],
            "joint_damping": [float(m.dof_damping[int(m.jnt_dofadr[j])]) for _, j in joints],
            "joint_armature": [float(m.dof_armature[int(m.jnt_dofadr[j])]) for _, j in joints],
            "joint_frictionloss": [float(m.dof_frictionloss[int(m.jnt_dofadr[j])]) for _, j in joints],
            "joint_range": [m.jnt_range[j].tolist() for _, j in joints],
            "geoms": geoms,
        }

    def summarize(self, report):
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    MujocoPlantSnapshot.main()
