"""Phase-1 deployment runtime and MuJoCo canonicalization helpers."""
from __future__ import annotations

"""Frozen Phase-1 deployment boundary.

This module contains no training logic. It exposes the deterministic actor
used by the canonical authority-isolated Phase-1 controller.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

OBS_DIM=48
ACTION_DIM=12
PREF_DIM=4
ACTION_CLIP=1.0

JOINT_NAMES=(
    "FL_hip_joint","FR_hip_joint","RL_hip_joint","RR_hip_joint",
    "FL_thigh_joint","FR_thigh_joint","RL_thigh_joint","RR_thigh_joint",
    "FL_calf_joint","FR_calf_joint","RL_calf_joint","RR_calf_joint",
)
ACTION_OFFSET=(0.1,-0.1,0.1,-0.1,0.8,0.8,1.0,1.0,-1.5,-1.5,-1.5,-1.5)
ACTION_SCALE=0.25
class Phase1DeploymentActor(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.action_clip=1.0
        self.actor_body=source.actor_body
        self.actor_mean=source.actor_mean
        self.family_hyper=source.family_hyper
        self.family_w1_base=nn.Parameter(source.family_w1_base.detach().clone())
        self.family_b1_base=nn.Parameter(source.family_b1_base.detach().clone())
        self.family_w2_base=nn.Parameter(source.family_w2_base.detach().clone())
        self.family_b2_base=nn.Parameter(source.family_b2_base.detach().clone())
        self.family_B_w1=nn.Parameter(source.family_B_w1.detach().clone())
        self.family_B_b1=nn.Parameter(source.family_B_b1.detach().clone())
        self.family_B_w2=nn.Parameter(source.family_B_w2.detach().clone())
        self.family_B_b2=nn.Parameter(source.family_B_b2.detach().clone())

    @torch.jit.export
    def pre_tanh(self,obs:Tensor,w:Tensor)->Tensor:
        h=self.actor_body(obs)
        base=self.actor_mean(h)
        c=self.family_hyper(w)
        z0=F.linear(h,self.family_w1_base,self.family_b1_base)
        zb=torch.einsum("bi,koi->bko",h,self.family_B_w1)
        zd=torch.einsum("bk,bko->bo",c,zb)+torch.einsum("bk,ko->bo",c,self.family_B_b1)
        z=F.elu(z0+zd)
        y0=F.linear(z,self.family_w2_base,self.family_b2_base)
        yb=torch.einsum("bi,koi->bko",z,self.family_B_w2)
        yd=torch.einsum("bk,bko->bo",c,yb)+torch.einsum("bk,ko->bo",c,self.family_B_b2)
        return base+y0+yd

    def forward(self,obs:Tensor,w:Tensor)->Tensor:
        return torch.tanh(self.pre_tanh(obs,w))*self.action_clip

class Phase1StandaloneActor(nn.Module):
    def __init__(self):
        super().__init__();self.action_clip=1.0
        self.actor_body=nn.Sequential(nn.Linear(OBS_DIM,128),nn.ELU(),nn.Linear(128,128),nn.ELU(),nn.Linear(128,128),nn.ELU())
        self.actor_mean=nn.Linear(128,ACTION_DIM)
        self.family_hyper=nn.Sequential(nn.Linear(PREF_DIM,32),nn.ELU(),nn.Linear(32,8))
        self.family_w1_base=nn.Parameter(torch.empty(128,128));self.family_b1_base=nn.Parameter(torch.empty(128))
        self.family_w2_base=nn.Parameter(torch.empty(ACTION_DIM,128));self.family_b2_base=nn.Parameter(torch.empty(ACTION_DIM))
        self.family_B_w1=nn.Parameter(torch.empty(8,128,128));self.family_B_b1=nn.Parameter(torch.empty(8,128))
        self.family_B_w2=nn.Parameter(torch.empty(8,ACTION_DIM,128));self.family_B_b2=nn.Parameter(torch.empty(8,ACTION_DIM))
    def pre_tanh(self,obs:Tensor,w:Tensor)->Tensor:
        h=self.actor_body(obs);base=self.actor_mean(h);c=self.family_hyper(w)
        z0=F.linear(h,self.family_w1_base,self.family_b1_base);zb=torch.einsum("bi,koi->bko",h,self.family_B_w1)
        zd=torch.einsum("bk,bko->bo",c,zb)+torch.einsum("bk,ko->bo",c,self.family_B_b1);z=F.elu(z0+zd)
        y0=F.linear(z,self.family_w2_base,self.family_b2_base);yb=torch.einsum("bi,koi->bko",z,self.family_B_w2)
        yd=torch.einsum("bk,bko->bo",c,yb)+torch.einsum("bk,ko->bo",c,self.family_B_b2)
        return base+y0+yd
    def forward(self,obs:Tensor,w:Tensor)->Tensor:return torch.tanh(self.pre_tanh(obs,w))*self.action_clip

@dataclass(frozen=True)
class Phase1RuntimeConfig:
    stale_hold_ms: float=40.0
    stale_warn_ms: float=20.0

class Phase1DeploymentRuntime:
    def __init__(self,policy_path:str,config:Phase1RuntimeConfig=Phase1RuntimeConfig(),device:str="cpu"):
        self.config=config
        self.device=torch.device(device)
        self.policy=torch.jit.load(policy_path,map_location=self.device)
        self.policy.eval()
        self.last_action=np.zeros((1,ACTION_DIM),dtype=np.float32)
        self.estop_latched=False

    @staticmethod
    def _validate(obs:np.ndarray,w:np.ndarray)->tuple[np.ndarray,np.ndarray]:
        obs=np.asarray(obs,dtype=np.float32);w=np.asarray(w,dtype=np.float32)
        if obs.ndim!=2 or obs.shape[1]!=OBS_DIM:
            raise ValueError(f"observation must be [batch,{OBS_DIM}]")
        if w.ndim!=2 or w.shape!=(obs.shape[0],PREF_DIM):
            raise ValueError(f"preference must be [batch,{PREF_DIM}]")
        if not np.isfinite(obs).all():raise ValueError("non-finite observation")
        if not np.isfinite(w).all():raise ValueError("non-finite preference")
        if (w<0).any():raise ValueError("negative preference weight")
        if not np.allclose(w.sum(axis=1),1.0,atol=1e-6):
            raise ValueError("preference must lie on simplex")
        return obs,w
    def reset_estop(self)->None:
        self.estop_latched=False

    def estop(self)->None:
        self.estop_latched=True

    def act(self,obs:np.ndarray,w:np.ndarray,age_ms:float=0.0)->np.ndarray:
        obs,w=self._validate(obs,w)
        if self.estop_latched:
            return np.zeros((obs.shape[0],ACTION_DIM),dtype=np.float32)
        if not np.isfinite(age_ms) or age_ms<0:
            raise ValueError("invalid observation age")
        if age_ms>self.config.stale_hold_ms:
            self.estop_latched=True
            return np.zeros((obs.shape[0],ACTION_DIM),dtype=np.float32)
        if age_ms>self.config.stale_warn_ms:
            if self.last_action.shape[0]==obs.shape[0]:return self.last_action.copy()
            return np.zeros((obs.shape[0],ACTION_DIM),dtype=np.float32)
        with torch.inference_mode():
            a=self.policy(torch.from_numpy(obs).to(self.device),torch.from_numpy(w).to(self.device))
        if a.ndim!=2 or tuple(a.shape)!=(obs.shape[0],ACTION_DIM):
            raise ValueError("invalid policy action shape")
        out=a.detach().cpu().numpy()
        if not np.isfinite(out).all():raise ValueError("non-finite policy action")
        if np.max(np.abs(out))>ACTION_CLIP+1e-6:
            raise ValueError("policy action outside normalized contract")
        self.last_action=out.astype(np.float32,copy=True)
        return self.last_action.copy()

class Phase1EagerStateRuntime(Phase1DeploymentRuntime):
    def __init__(self,policy_path:str,config:Phase1RuntimeConfig=Phase1RuntimeConfig(),device:str="cpu"):
        self.config=config;self.device=torch.device(device)
        bundle=torch.load(policy_path,map_location=self.device,weights_only=False)
        state=bundle.get("actor_state",bundle)
        self.policy=Phase1StandaloneActor().to(self.device)
        self.policy.load_state_dict(state);self.policy.eval()
        self.last_action=np.zeros((1,ACTION_DIM),dtype=np.float32)
        self.estop_latched=False

def action_to_joint_target(action:np.ndarray)->np.ndarray:
    a=np.asarray(action,dtype=np.float32)
    if a.ndim!=2 or a.shape[1]!=ACTION_DIM:raise ValueError("invalid action shape")
    if not np.isfinite(a).all():raise ValueError("non-finite action")
    if np.max(np.abs(a))>ACTION_CLIP+1e-6:raise ValueError("action outside normalized contract")
    return np.asarray(ACTION_OFFSET,dtype=np.float32)[None,:]+ACTION_SCALE*a


# ---- MuJoCo adapter helpers ----
"""Canonical Phase-1 MuJoCo interface adapter primitives.

No MuJoCo import is required here. Raw-engine extraction should feed these
functions only after D3-A live-runtime conventions are verified.
"""
import numpy as np

OBS_DIM=48
ACTION_DIM=12
CANONICAL_JOINT_ORDER=(
    "FL_hip_joint","FR_hip_joint","RL_hip_joint","RR_hip_joint",
    "FL_thigh_joint","FR_thigh_joint","RL_thigh_joint","RR_thigh_joint",
    "FL_calf_joint","FR_calf_joint","RL_calf_joint","RR_calf_joint",
)
CANONICAL_DEFAULT_Q=np.asarray(
    [0.1,-0.1,0.1,-0.1,0.8,0.8,1.0,1.0,-1.5,-1.5,-1.5,-1.5],
    dtype=np.float32,
)
ACTION_SCALE=np.float32(0.25)

def quat_wxyz_to_rot(q):
    q=np.asarray(q,dtype=np.float64)
    if q.shape[-1]!=4:raise ValueError("quaternion must be wxyz")
    w,x,y,z=np.moveaxis(q,-1,0)
    return np.array([
      [1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
      [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
      [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],dtype=np.float64)

def canonical_base_kinematics(free_quat_wxyz,free_qvel6):
    qv=np.asarray(free_qvel6,dtype=np.float64)
    if qv.shape[-1]!=6:raise ValueError("free qvel must have 6 elements")
    R=quat_wxyz_to_rot(free_quat_wxyz)
    lin_b=R.T@qv[:3]
    ang_b=qv[3:6].copy()
    gravity_b=R.T@np.array([0.0,0.0,-1.0],dtype=np.float64)
    return lin_b.astype(np.float32),ang_b.astype(np.float32),gravity_b.astype(np.float32)

def quat_rotate_inverse_wxyz(quat_wxyz,vec_world):
    q=np.asarray(quat_wxyz,np.float64);v=np.asarray(vec_world,np.float64)
    q=q/np.linalg.norm(q);w=q[0];qv=q[1:4]
    a=v*(2.0*w*w-1.0)
    b=np.cross(qv,v)*w*2.0
    c=qv*(qv@v)*2.0
    return (a-b+c).astype(np.float64)

def root_com_velocity_b_from_freejoint(quat_wxyz,lin_vel_world,ang_vel_body,com_offset_body):
    q=np.asarray(quat_wxyz,np.float64);q=q/np.linalg.norm(q)
    w,x,y,z=q
    R=np.array([
        [1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],
    ],dtype=np.float64)
    v_origin_b=R.T@np.asarray(lin_vel_world,np.float64)
    return v_origin_b+np.cross(np.asarray(ang_vel_body,np.float64),np.asarray(com_offset_body,np.float64))

def permutation(source_names, target_names):
    src=list(source_names)
    if len(src)!=len(set(src)):raise ValueError("duplicate source joint names")
    return np.asarray([src.index(name) for name in target_names],dtype=np.int64)

def reorder(values, source_names, target_names=CANONICAL_JOINT_ORDER):
    a=np.asarray(values)
    if a.shape[-1]!=len(source_names):raise ValueError("joint vector width mismatch")
    return a[...,permutation(source_names,target_names)]

def build_canonical_obs(base_lin_vel_b,base_ang_vel_b,projected_gravity_b,
                        command,joint_pos_abs,joint_vel_abs,prev_action,
                        joint_names=CANONICAL_JOINT_ORDER):
    q=reorder(joint_pos_abs,joint_names)
    qd=reorder(joint_vel_abs,joint_names)
    pa=reorder(prev_action,joint_names)
    parts=[
        np.asarray(base_lin_vel_b,np.float32),
        np.asarray(base_ang_vel_b,np.float32),
        np.asarray(projected_gravity_b,np.float32),
        np.asarray(command,np.float32),
        q.astype(np.float32)-CANONICAL_DEFAULT_Q,
        qd.astype(np.float32),
        pa.astype(np.float32),
    ]
    out=np.concatenate(parts,axis=-1).astype(np.float32)
    if out.shape[-1]!=OBS_DIM:raise ValueError(f"expected {OBS_DIM}-D observation")
    if not np.isfinite(out).all():raise ValueError("non-finite canonical observation")
    return out

def canonical_joint_target(action):
    a=np.asarray(action,np.float32)
    if a.shape[-1]!=ACTION_DIM:raise ValueError("action width mismatch")
    if not np.isfinite(a).all():raise ValueError("non-finite action")
    if np.max(np.abs(a))>1.0+1e-6:raise ValueError("normalized action outside [-1,1]")
    return CANONICAL_DEFAULT_Q+ACTION_SCALE*a

def source_dcmotor_torque(q_target,q,qdot,kp=25.0,kd=0.5,effort_limit=33.5,saturation_effort=33.5,velocity_limit=21.0):
    q_target=np.asarray(q_target,dtype=np.float64)
    q=np.asarray(q,dtype=np.float64)
    qdot=np.asarray(qdot,dtype=np.float64)
    tau_raw=kp*(q_target-q)-kd*qdot
    vel_at_effort_lim=velocity_limit*(1.0+effort_limit/saturation_effort)
    qdot_clip=np.clip(qdot,-vel_at_effort_lim,vel_at_effort_lim)
    tau_top=saturation_effort*(1.0-qdot_clip/velocity_limit)
    tau_bottom=saturation_effort*(-1.0-qdot_clip/velocity_limit)
    tau_max=np.minimum(tau_top,effort_limit)
    tau_min=np.maximum(tau_bottom,-effort_limit)
    return np.clip(tau_raw,tau_min,tau_max)

def target_for_actuator_order(action,actuator_joint_names):
    q=canonical_joint_target(action)
    return reorder(q,CANONICAL_JOINT_ORDER,actuator_joint_names)
