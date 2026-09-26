from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch

class Phase5PlantEnsembleWrapper:
    """Transparent Gym wrapper that changes only physical plant parameters."""
    def __init__(self, env, manifest_path: str | Path, tuple_log_path: str | Path):
        import gymnasium as gym
        self._wrapper = gym.Wrapper(env)
        self.env = env
        self.unwrapped = env.unwrapped
        self.manifest = json.load(open(manifest_path))
        self.log_path = Path(tuple_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        existing_draws = 0
        if self.log_path.exists():
            existing_draws = sum(1 for line in self.log_path.open() if line.strip())
        else:
            self.log_path.write_text("")
        self.sobol = torch.quasirandom.SobolEngine(
            3, scramble=True, seed=int(self.manifest["sampler"]["seed"])
        )
        if existing_draws:
            self.sobol.fast_forward(existing_draws)
        self.robot = self.unwrapped.scene["robot"]
        self.names = list(self.robot.data.joint_names)
        self.bodies = list(self.robot.data.body_names)
        self.trunk = self.bodies.index("trunk")
        self.hips = [i for i,n in enumerate(self.names) if "_hip_joint" in n]
        self.flex = [i for i,n in enumerate(self.names) if "_thigh_joint" in n or "_calf_joint" in n]
        self.base_mass = float(self.robot.data.default_mass[0,self.trunk].cpu())
        self.base_inertia = self.robot.data.default_inertia[0,self.trunk].cpu().clone()
        self.heldout = np.array([
            [x["mass_delta_kg"],x["passive_blend"],x["contact_blend"]]
            for x in self.manifest["held_out_plants"]
        ], dtype=np.float64)
        self.draw_index = existing_draws

    @property
    def action_space(self): return self.env.action_space
    @property
    def observation_space(self): return self.env.observation_space
    @property
    def metadata(self): return getattr(self.env, "metadata", {})
    def close(self): return self.env.close()

    def _draw(self, n):
        out=[]
        while len(out)<n:
            u=self.sobol.draw(1).cpu().numpy()[0]
            x=np.array([-1.5+4.5*u[0],u[1],u[2]],dtype=np.float64)
            if len(self.heldout) and np.min(np.max(np.abs(self.heldout-x[None,:]),axis=1)) <= 1e-9:
                continue
            out.append(x)
        return np.asarray(out)

    def _apply(self, env_ids):
        ids=np.asarray(env_ids,dtype=np.int64)
        if len(ids)==0:return
        x=self._draw(len(ids))
        tid=torch.as_tensor(ids,dtype=torch.int32,device="cpu")
        masses=self.robot.root_physx_view.get_masses()
        inertias=self.robot.root_physx_view.get_inertias()
        damp=self.robot.root_physx_view.get_dof_dampings()
        arm=self.robot.root_physx_view.get_dof_armatures()
        mats=self.robot.root_physx_view.get_material_properties()
        for k,e in enumerate(ids):
            dm,lp,lc=[float(v) for v in x[k]]
            nm=self.base_mass+dm
            masses[e,self.trunk]=nm
            inertias[e,self.trunk]=self.base_inertia*(nm/self.base_mass)
            damp[e]=0.0;arm[e]=0.0
            damp[e,self.hips]=lp
            damp[e,self.flex]=2.0*lp
            arm[e,:]=0.01*lp
            mats[e,:,0]=0.8
            mats[e,:,1]=0.6+0.2*lc
            mats[e,:,2]=0.0
        self.robot.root_physx_view.set_masses(masses,tid)
        self.robot.root_physx_view.set_inertias(inertias,tid)
        self.robot.root_physx_view.set_dof_dampings(damp,indices=tid)
        self.robot.root_physx_view.set_dof_armatures(arm,indices=tid)
        self.robot.root_physx_view.set_material_properties(mats,tid)
        with self.log_path.open("a") as f:
            for k,e in enumerate(ids):
                dm,lp,lc=[float(v) for v in x[k]]
                f.write(json.dumps({"draw":self.draw_index,"env":int(e),
                    "mass_delta_kg":dm,"passive_blend":lp,"contact_blend":lc})+"\n")
                self.draw_index+=1

    def _obs_after_apply(self):
        self.unwrapped.scene.write_data_to_sim()
        self.unwrapped.sim.forward()
        return self.unwrapped.observation_manager.compute(update_history=True)

    def reset(self, *args, **kwargs):
        _,info=self.env.reset(*args,**kwargs)
        self._apply(np.arange(self.unwrapped.num_envs))
        return self._obs_after_apply(),info

    def step(self, action):
        obs,reward,terminated,truncated,info=self.env.step(action)
        done=(terminated|truncated).detach().cpu().numpy()
        ids=np.flatnonzero(done)
        if len(ids):
            self._apply(ids)
            obs=self._obs_after_apply()
        return obs,reward,terminated,truncated,info

    def __getattr__(self,name):
        if name in {"env","unwrapped","manifest","log_path","sobol","robot","names","bodies",
                    "trunk","hips","flex","base_mass","base_inertia","heldout","draw_index","_wrapper"}:
            raise AttributeError(name)
        return getattr(self.env,name)
