#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
SRC=ROOT/"runs/phase3_t1b_source_anchor/mujoco_source_replay.json"
OUT=ROOT/"runs/phase3_t1d_temporal_reversal";OUT.mkdir(parents=True,exist_ok=True)
H=(1,4,8,16,32,64)

def main():
 d=json.load(open(SRC));tr=d["traces"];reports=[]
 for suite in range(4):
  envs=range(8);rows=[]
  # tensors [env,time]
  TT=[];CT=[];TP=[];CP=[];TH=[];CH=[];TS=[];CS=[];AD=[]
  for e in envs:
   T=tr[f"s{suite}:e{e}:T"];C=tr[f"s{suite}:e{e}:C"]
   TT.append([x["T_obj"] for x in T]);CT.append([x["T_obj"] for x in C])
   TP.append([x["tracking"] for x in T]);CP.append([x["tracking"] for x in C])
   TH.append([x["height"] for x in T]);CH.append([x["height"] for x in C])
   TS.append([x["sat_frac"] for x in T]);CS.append([x["sat_frac"] for x in C])
   AD.append([np.linalg.norm(np.asarray(x["action"])-np.asarray(y["action"])) for x,y in zip(T,C)])
  TT=np.asarray(TT);CT=np.asarray(CT);TP=np.asarray(TP);CP=np.asarray(CP)
  TH=np.asarray(TH);CH=np.asarray(CH);TS=np.asarray(TS);CS=np.asarray(CS);AD=np.asarray(AD)
  first_phys_wrong=None;first_obj_wrong=None
  for h in H:
   physical=float(np.mean(TP[:,:h]-CP[:,:h]))
   obj=float(np.mean(np.sum(TT[:,:h]-CT[:,:h],axis=1)))
   row={"horizon":h,"tracking_delta_T_minus_C":physical,"cum_T_margin":obj,
        "action_sep_mean":float(np.mean(AD[:,:h])),
        "height_delta":float(np.mean(TH[:,:h]-CH[:,:h])),
        "T_sat":float(np.mean(TS[:,:h])),"C_sat":float(np.mean(CS[:,:h]))}
   rows.append(row)
   if first_phys_wrong is None and physical>0:first_phys_wrong=h
   if first_obj_wrong is None and obj<0:first_obj_wrong=h
  reports.append({"suite":suite,"ladder":rows,"first_physical_wrong_h":first_phys_wrong,"first_objective_wrong_h":first_obj_wrong})
 agg={
  "early_H1_physical_correct_suites":int(sum(r["ladder"][0]["tracking_delta_T_minus_C"]<0 for r in reports)),
  "H64_physical_correct_suites":int(sum(r["ladder"][-1]["tracking_delta_T_minus_C"]<0 for r in reports)),
  "early_H1_objective_correct_suites":int(sum(r["ladder"][0]["cum_T_margin"]>0 for r in reports)),
  "H64_objective_correct_suites":int(sum(r["ladder"][-1]["cum_T_margin"]>0 for r in reports)),
  "first_physical_wrong_h":[r["first_physical_wrong_h"] for r in reports],
  "first_objective_wrong_h":[r["first_objective_wrong_h"] for r in reports],
 }
 rep={"schema":"phase3_t1d_temporal_reversal_v1","suites":reports,"aggregate":agg}
 (OUT/"temporal_reversal.json").write_text(json.dumps(rep,indent=2)+"\n")
 print(json.dumps(rep,indent=2),flush=True)

if __name__=="__main__":main()
