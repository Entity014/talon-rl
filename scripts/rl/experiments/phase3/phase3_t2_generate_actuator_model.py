#!/usr/bin/env python3
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[4]
D=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco"
A1=D/"a1.xml";SCENE=D/"scene.xml"
A1_OUT=D/"a1_t2_actuator_calibrated.xml"
SCENE_OUT=D/"scene_t2_actuator.xml"
OUT=ROOT/"runs/phase3_t2_actuator_calibration";OUT.mkdir(parents=True,exist_ok=True)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    src=A1.read_text()
    repl=[
      ('<joint axis="0 1 0" damping="2" armature="0.01" frictionloss="0.2"/>',
       '<joint axis="0 1 0" damping="0.5" armature="0.01" frictionloss="0"/>'),
      ('<position kp="100" forcerange="-33.5 33.5"/>',
       '<position kp="25" forcerange="-33.5 33.5"/>'),
      ('<joint axis="1 0 0" damping="1" range="-0.802851 0.802851"/>',
       '<joint axis="1 0 0" damping="0.5" range="-0.802851 0.802851"/>'),
    ]
    counts=[]
    for old,new in repl:
        n=src.count(old);counts.append(n)
        if n!=1:raise RuntimeError(f"expected exactly one replacement, got {n}: {old}")
        src=src.replace(old,new)
    A1_OUT.write_text(src)
    scene=SCENE.read_text()
    old='<include file="a1.xml"/>'
    if scene.count(old)!=1:raise RuntimeError("scene include mismatch")
    SCENE_OUT.write_text(scene.replace(old,'<include file="a1_t2_actuator_calibrated.xml"/>'))
    rep={
      "schema":"phase3_t2_model_generation_v1",
      "baseline_a1_sha256":sha(A1),"calibrated_a1_sha256":sha(A1_OUT),
      "baseline_scene_sha256":sha(SCENE),"calibrated_scene_sha256":sha(SCENE_OUT),
      "replacements":counts,
      "intervention":{"kp":25.0,"damping":0.5,"frictionloss":0.0,"forcerange":[-33.5,33.5]},
    }
    (OUT/"model_generation.json").write_text(json.dumps(rep,indent=2)+"\n")
    print(json.dumps(rep,indent=2),flush=True)

if __name__=="__main__":main()
