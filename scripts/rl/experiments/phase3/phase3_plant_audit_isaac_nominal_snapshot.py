#!/usr/bin/env python3
"""Every plant parameter the Isaac A1 scene resolves to, as one record.

The Isaac half of the plant-equivalence comparison, to be read beside the
MuJoCo snapshot. Base-mass randomisation is switched off and a single
environment is built, so what it reports is the nominal model rather than one
sampled instance of it.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.core.isaac_audit import IsaacAudit


class IsaacPlantSnapshot(IsaacAudit):
    """Every plant parameter the Isaac A1 scene resolves to."""

    run = "phase3_plant_equivalence_audit"
    report = "isaac_nominal_snapshot.json"
    schema = "phase3_plant_isaac_nominal_snapshot_v1"
    num_envs = 1
    reset_seed = 1
    set_cfg_seed = False      # the original never set it, and the snapshot is nominal

    def configure(self, cfg):
        cfg.events.add_base_mass = None

    def rollout(self, env, obs):
        u = env.unwrapped
        d = u.scene["robot"].data
        spawn = self.cfg.scene.robot.spawn
        rep = {
            "schema": self.schema,
            "step_dt": float(u.step_dt),
            "sim_dt": float(u.cfg.sim.dt),
            "decimation": int(u.cfg.decimation),
            "body_names": list(d.body_names),
            "mass": d.default_mass[0].detach().cpu().tolist(),
            "inertia": d.default_inertia[0].detach().cpu().tolist(),
            "body_com_pos_b": d.body_com_pos_b[0].detach().cpu().tolist(),
            "body_com_quat_b": d.body_com_quat_b[0].detach().cpu().tolist(),
            "joint_names": list(d.joint_names),
            "joint_stiffness": d.default_joint_stiffness[0].detach().cpu().tolist(),
            "joint_damping": d.default_joint_damping[0].detach().cpu().tolist(),
            "joint_armature": d.default_joint_armature[0].detach().cpu().tolist(),
            "joint_friction_coeff": d.default_joint_friction_coeff[0].detach().cpu().tolist(),
            "solver_position_iterations":
                int(spawn.articulation_props.solver_position_iteration_count),
            "solver_velocity_iterations":
                int(spawn.articulation_props.solver_velocity_iteration_count),
            "rigid_linear_damping": float(spawn.rigid_props.linear_damping),
            "rigid_angular_damping": float(spawn.rigid_props.angular_damping),
            "max_depenetration_velocity": float(spawn.rigid_props.max_depenetration_velocity),
        }
        # the terrain may not expose a material; record why rather than omitting it
        try:
            rep["ground_physics_material"] = str(self.cfg.scene.terrain.physics_material)
        except Exception as e:
            rep["ground_physics_material_error"] = str(e)
        self.write(rep)
        print(json.dumps(rep, indent=2), flush=True)
        return rep


if __name__ == "__main__":
    IsaacPlantSnapshot.main()
