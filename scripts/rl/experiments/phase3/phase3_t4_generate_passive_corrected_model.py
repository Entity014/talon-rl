#!/usr/bin/env python3
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[4]
D=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco"
SRC=D/"a1_t3_dcmotor.xml";SCENE_SRC=D/"scene_t3_dcmotor.xml"
A1_OUT=D/"a1_t4_passive_corrected.xml";SCENE_OUT=D/"scene_t4_passive_corrected.xml"
OUT=ROOT/"runs/phase3_t4_passive_correction";OUT.mkdir(parents=True,exist_ok=True)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    s=SRC.read_text()
    repl=[
      ('<joint axis="0 1 0" damping="2" armature="0.01" frictionloss="0.2"/>',
       '<joint axis="0 1 0" damping="0" armature="0" frictionloss="0"/>'),
      ('<joint axis="1 0 0" damping="1" range="-0.802851 0.802851"/>',
       '<joint axis="1 0 0" damping="0" range="-0.802851 0.802851"/>'),
    ]
    counts=[]
    for old,new in repl:
        n=s.count(old);counts.append(n)
        if n!=1: raise RuntimeError(f"expected one replacement, got {n}: {old}")
        s=s.replace(old,new)
    A1_OUT.write_text(s)
    sc=SCENE_SRC.read_text()
    token='<include file="a1_t3_dcmotor.xml"/>'
    if sc.count(token)!=1: raise RuntimeError("scene include mismatch")
    SCENE_OUT.write_text(sc.replace(token,'<include file="a1_t4_passive_corrected.xml"/>'))
    rep={"schema":"phase3_t4_model_generation_v1",
         "t3_a1_sha256":sha(SRC),"t4_a1_sha256":sha(A1_OUT),
         "t3_scene_sha256":sha(SCENE_SRC),"t4_scene_sha256":sha(SCENE_OUT),
         "intervention":{"joint_damping":0.0,"frictionloss":0.0,"armature":0.0}}
    (OUT/"model_generation.json").write_text(json.dumps(rep,indent=2)+"\n")
    print(json.dumps(rep,indent=2),flush=True)

if __name__=="__main__":main()
