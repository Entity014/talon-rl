#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,re

ROOT=Path(__file__).resolve().parents[4]
D=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco"
A1=D/"a1.xml";SCENE=D/"scene.xml"
A1_OUT=D/"a1_t3_dcmotor.xml"
SCENE_OUT=D/"scene_t3_dcmotor.xml"
OUT=ROOT/"runs/phase3_t3_dcmotor";OUT.mkdir(parents=True,exist_ok=True)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    src=A1.read_text()
    # Replace only the concrete actuator block. Keep baseline joint damping/friction/inertia/contact untouched.
    block=re.search(r"<actuator>.*?</actuator>",src,re.S)
    if not block: raise RuntimeError("actuator block not found")
    names=[
      ("FR_hip","FR_hip_joint"),("FR_thigh","FR_thigh_joint"),("FR_calf","FR_calf_joint"),
      ("FL_hip","FL_hip_joint"),("FL_thigh","FL_thigh_joint"),("FL_calf","FL_calf_joint"),
      ("RR_hip","RR_hip_joint"),("RR_thigh","RR_thigh_joint"),("RR_calf","RR_calf_joint"),
      ("RL_hip","RL_hip_joint"),("RL_thigh","RL_thigh_joint"),("RL_calf","RL_calf_joint"),
    ]
    motor_lines=["  <actuator>"]
    for name,joint in names:
        motor_lines.append(f'    <motor name="{name}" joint="{joint}" gear="1" ctrllimited="false"/>')
    motor_lines.append("  </actuator>")
    repl="\n".join(motor_lines)
    out=src[:block.start()]+repl+src[block.end():]
    # Home keyframe ctrl is no longer a position target. Make it zero torque.
    out=re.sub(r'\n\s+ctrl="[^"]*"/>', '\n      ctrl="0 0 0 0 0 0 0 0 0 0 0 0"/>', out, count=1)
    A1_OUT.write_text(out)

    scene=SCENE.read_text()
    token='<include file="a1.xml"/>'
    if scene.count(token)!=1: raise RuntimeError("scene include mismatch")
    SCENE_OUT.write_text(scene.replace(token,'<include file="a1_t3_dcmotor.xml"/>'))

    rep={
      "schema":"phase3_t3_model_generation_v1",
      "baseline_a1_sha256":sha(A1),
      "t3_a1_sha256":sha(A1_OUT),
      "baseline_scene_sha256":sha(SCENE),
      "t3_scene_sha256":sha(SCENE_OUT),
      "actuator_type":"motor",
      "gear":1.0,
      "ctrllimited":False,
      "baseline_joint_dynamics_preserved":True,
    }
    (OUT/"model_generation.json").write_text(json.dumps(rep,indent=2)+"\n")
    print(json.dumps(rep,indent=2),flush=True)

if __name__=="__main__": main()
