#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
DIR=ROOT/"runs/phase3_t1_matched_trace"
I=json.load(open(DIR/"isaac_trace.json"))
M=json.load(open(DIR/"mujoco_trace.json"))
CMDS=("forward","turn_left","turn_right","lateral")

def arr(rows,key):
    if key in ("action","q","qd"):
        return np.asarray([x[key] for x in rows],float)
    if key=="state":
        return np.asarray([[x["vx"],x["vy"],x["vz"],x["wx"],x["wy"],x["wz"],*x["g"],*x["q"],*x["qd"]] for x in rows],float)
    return np.asarray([x[key] for x in rows],float)

def engine_metrics(rep,cmd):
    T=rep["traces"][f"{cmd}:T"];C=rep["traces"][f"{cmd}:C"]
    aT=arr(T,"action");aC=arr(C,"action");da=aT-aC
    st=arr(T,"state");sc=arr(C,"state")
    trT=arr(T,"tracking_error");trC=arr(C,"tracking_error")
    tObj=arr(T,"T_obj");cObj=arr(C,"T_obj")
    out={
      "t0_action_sep":float(np.linalg.norm(da[0])),
      "mean_action_sep_h8":float(np.mean(np.linalg.norm(da[:8],axis=1))),
      "mean_action_sep_h16":float(np.mean(np.linalg.norm(da[:16],axis=1))),
      "mean_action_sep_h64":float(np.mean(np.linalg.norm(da,axis=1))),
      "mean_state_sep_h8":float(np.mean(np.linalg.norm(st[:8]-sc[:8],axis=1))),
      "mean_state_sep_h16":float(np.mean(np.linalg.norm(st[:16]-sc[:16],axis=1))),
      "mean_state_sep_h64":float(np.mean(np.linalg.norm(st-sc,axis=1))),
      "tracking_delta_h8":float(np.mean(trT[:8]-trC[:8])),
      "tracking_delta_h16":float(np.mean(trT[:16]-trC[:16])),
      "tracking_delta_h64":float(np.mean(trT-trC)),
      "cum_T_margin_h8":float(np.sum(tObj[:8]-cObj[:8])),
      "cum_T_margin_h16":float(np.sum(tObj[:16]-cObj[:16])),
      "cum_T_margin_h64":float(np.sum(tObj-cObj)),
      "T_tracking_mean":float(np.mean(trT)),
      "C_tracking_mean":float(np.mean(trC)),
      "T_height_mean":float(np.mean(arr(T,"height"))),
      "C_height_mean":float(np.mean(arr(C,"height"))),
      "T_sat_mean":float(np.mean(arr(T,"sat_frac"))),
      "C_sat_mean":float(np.mean(arr(C,"sat_frac"))),
      "T_contacts_mean":float(np.mean(arr(T,"contacts"))),
      "C_contacts_mean":float(np.mean(arr(C,"contacts"))),
      "mean_joint_sep":float(np.mean(np.linalg.norm(arr(T,"q")-arr(C,"q"),axis=1))),
    }
    return out,da

rows={};cross={}
for cmd in CMDS:
    im,ida=engine_metrics(I,cmd);mm,mda=engine_metrics(M,cmd)
    # Same initial observation should imply same T-vs-C action vector.
    t0_vec_err=float(np.max(np.abs(ida[0]-mda[0])))
    # cosine of preference response after divergence.
    cos=[]
    for x,y in zip(ida,mda):
        nx=np.linalg.norm(x);ny=np.linalg.norm(y)
        cos.append(float(np.dot(x,y)/(nx*ny+1e-12)))
    rows[cmd]={"isaac":im,"mujoco":mm}
    cross[cmd]={
      "t0_preference_action_vector_max_error":t0_vec_err,
      "mean_action_response_cosine_h8":float(np.mean(cos[:8])),
      "mean_action_response_cosine_h16":float(np.mean(cos[:16])),
      "mean_action_response_cosine_h64":float(np.mean(cos)),
      "action_sep_ratio_mj_over_isaac_h64":float(mm["mean_action_sep_h64"]/(im["mean_action_sep_h64"]+1e-12)),
      "center_tracking_shift_mj_minus_isaac":float(mm["C_tracking_mean"]-im["C_tracking_mean"]),
      "center_height_shift_mj_minus_isaac":float(mm["C_height_mean"]-im["C_height_mean"]),
      "center_saturation_shift":float(mm["C_sat_mean"]-im["C_sat_mean"]),
      "tracking_margin_sign_flip":bool(np.sign(im["tracking_delta_h64"])!=np.sign(mm["tracking_delta_h64"])),
      "T_return_margin_sign_flip":bool(np.sign(im["cum_T_margin_h64"])!=np.sign(mm["cum_T_margin_h64"])),
    }

agg={
 "isaac_T_semantic_correct_commands":int(sum(rows[c]["isaac"]["tracking_delta_h64"]<0 and rows[c]["isaac"]["cum_T_margin_h64"]>0 for c in CMDS)),
 "mujoco_T_semantic_correct_commands":int(sum(rows[c]["mujoco"]["tracking_delta_h64"]<0 and rows[c]["mujoco"]["cum_T_margin_h64"]>0 for c in CMDS)),
 "t0_exact_commands":int(sum(cross[c]["t0_preference_action_vector_max_error"]<=1e-6 for c in CMDS)),
 "tracking_sign_flip_commands":int(sum(cross[c]["tracking_margin_sign_flip"] for c in CMDS)),
 "return_sign_flip_commands":int(sum(cross[c]["T_return_margin_sign_flip"] for c in CMDS)),
 "mean_action_sep_ratio":float(np.mean([cross[c]["action_sep_ratio_mj_over_isaac_h64"] for c in CMDS])),
 "mean_action_cosine_h8":float(np.mean([cross[c]["mean_action_response_cosine_h8"] for c in CMDS])),
 "mean_action_cosine_h64":float(np.mean([cross[c]["mean_action_response_cosine_h64"] for c in CMDS])),
 "mean_center_tracking_shift":float(np.mean([cross[c]["center_tracking_shift_mj_minus_isaac"] for c in CMDS])),
 "mean_center_height_shift":float(np.mean([cross[c]["center_height_shift_mj_minus_isaac"] for c in CMDS])),
 "mean_center_saturation_shift":float(np.mean([cross[c]["center_saturation_shift"] for c in CMDS])),
}

# Evidence-oriented mechanism flags, not adaptation decisions.
mechanism={
 "initial_policy_function_changed":bool(agg["t0_exact_commands"]<4),
 "closed_loop_policy_response_diverges":bool(agg["mean_action_cosine_h64"]<0.9 or abs(np.log(max(agg["mean_action_sep_ratio"],1e-12)))>0.2),
 "dynamics_effect_sign_reversal":bool(agg["tracking_sign_flip_commands"]>=3 and agg["return_sign_flip_commands"]>=3),
 "center_baseline_shift_material":bool(abs(agg["mean_center_tracking_shift"])>0.1),
 "operating_regime_shift":bool(abs(agg["mean_center_height_shift"])>0.05 or agg["mean_center_saturation_shift"]>0.2),
}
rep={"schema":"phase3_t1_compare_v1","commands":rows,"cross_engine":cross,"aggregate":agg,"mechanism_flags":mechanism}
(DIR/"comparison_report.json").write_text(json.dumps(rep,indent=2)+"\n")
print(json.dumps({"aggregate":agg,"mechanism_flags":mechanism,"cross_engine":cross},indent=2))
