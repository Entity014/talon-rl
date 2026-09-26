import json, numpy as np
from pathlib import Path
p=Path("runs/post_v2_t0_trajectory_credit-2026-09-23/audit.json")
d=json.loads(p.read_text())
weights={"P":np.array([.8,.1,.1]),"B":np.array([.1,.8,.1]),"E":np.array([.1,.1,.8])}
cross={}
for kind in ("D1","V2"):
    cross[kind]={}
    for H in ("8","16","32"):
        rows=d["summary"][kind][H]["suite_rows"]
        per=[]
        for r in rows:
            mat=np.asarray(r["objective_matrix"],float)
            wins={}; vals={}; diag={}
            for i,lab in enumerate(("P","B","E")):
                sc=mat@weights[lab]
                vals[lab]=sc.tolist()
                win=int(np.argmax(sc))
                wins[lab]=win
                diag[lab]=bool(win==i)
            per.append({"suite":r["suite"],"scalarized_values_by_target_pref":vals,"winners_by_target_pref":wins,"diagonal":diag})
        cross[kind][H]={
            "diagonal_fraction":{lab:float(np.mean([x["diagonal"][lab] for x in per])) for lab in ("P","B","E")},
            "all_three_fraction":float(np.mean([all(x["diagonal"].values()) for x in per])),
            "suite_rows":per
        }
d["cross_preference_scalarized_ranking"]=cross
p.write_text(json.dumps(d,indent=2)+"\n")
print(json.dumps(cross,indent=2))
