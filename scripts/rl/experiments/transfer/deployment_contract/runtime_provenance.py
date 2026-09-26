#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,platform,sys
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/"runs/phase1_d3_runtime_provenance"
OUT.mkdir(parents=True,exist_ok=True)

def sha(p:Path)->str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    scene=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/scene.xml"
    a1=ROOT/"talon_rl/assets/data/Robots/unitree_a1/mujoco/a1.xml"
    m=mujoco.MjModel.from_xml_path(str(scene))
    rep={
      "schema":"phase1_d3_mujoco_runtime_provenance_v1",
      "python":sys.version.split()[0],
      "python_executable":sys.executable,
      "mujoco":mujoco.__version__,
      "numpy":np.__version__,
      "platform":platform.platform(),
      "scene_xml_sha256":sha(scene),
      "a1_xml_sha256":sha(a1),
      "timestep":float(m.opt.timestep),
      "integrator":int(m.opt.integrator),
      "solver":int(m.opt.solver),
      "iterations":int(m.opt.iterations),
      "ls_iterations":int(m.opt.ls_iterations),
      "gravity":m.opt.gravity.tolist(),
      "nq":int(m.nq),"nv":int(m.nv),"nu":int(m.nu),
      "ctrlrange":m.actuator_ctrlrange.tolist(),
      "forcerange":m.actuator_forcerange.tolist(),
    }
    (OUT/"runtime_provenance.json").write_text(json.dumps(rep,indent=2)+"\n")
    print(json.dumps(rep,indent=2),flush=True)

if __name__=="__main__":main()
