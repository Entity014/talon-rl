#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,torch,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
from talon_rl.t4_actor_critic import T4SharedActorCritic,initialize_from_rsl_m01

def source_actor(obs,state):
    x=obs
    for i in (0,2,4):
        x=torch.nn.functional.elu(torch.nn.functional.linear(x,state[f"actor.{i}.weight"],state[f"actor.{i}.bias"]))
    return torch.tanh(torch.nn.functional.linear(x,state["actor.6.weight"],state["actor.6.bias"]))*T4SharedActorCritic.ACTION_CLIP
def source_value(obs,state):
    x=obs
    for i in (0,2,4):
        x=torch.nn.functional.elu(torch.nn.functional.linear(x,state[f"critic.{i}.weight"],state[f"critic.{i}.bias"]))
    return torch.nn.functional.linear(x,state["critic.6.weight"],state["critic.6.bias"]).squeeze(-1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path,default=Path("runs/m0_1_seed0_2026-09-22/model_299.pt"));ap.add_argument("--output",type=Path,required=True);args=ap.parse_args()
    p=torch.load(args.checkpoint,map_location="cpu",weights_only=False);s=p.get("model_state_dict",p.get("model",p))
    torch.manual_seed(44);obs=torch.randn(64,48);m=T4SharedActorCritic(48,12);initialize_from_rsl_m01(m,args.checkpoint)
    src_mu=source_actor(obs,s);src_v=source_value(obs,s)
    prefs=torch.tensor([[.7,.1,.1,.1],[.1,.7,.1,.1],[.1,.1,.7,.1],[.1,.1,.1,.7],[.25,.25,.25,.25]])
    rows=[]
    for w0 in prefs:
        w=w0.repeat(len(obs),1)
        mu=m.act_inference_with_preference(obs,w);v=m.value_with_preference(obs,w)
        rows.append({"w":w0.tolist(),"action_max_abs_diff":float((mu-src_mu).abs().max()),"value_max_abs_diff":float((v-src_v[:,None]).abs().max())})
    logstd_diff=float((m.log_std-s["std"].log()).abs().max())
    report={"schema":"t4_init_smoke_v1","rows":rows,"logstd_max_abs_diff":logstd_diff,
            "pass":all(r["action_max_abs_diff"]<=1e-6 and r["value_max_abs_diff"]<=1e-5 for r in rows) and logstd_diff<=1e-7}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
