#!/usr/bin/env python3
from pathlib import Path
p=Path(__file__).with_name("post_v2_t5_c25_angular_credit_confirm.py")
s=p.read_text()
s=s.replace('A_snap_10.pt','A_snap_0.pt')
s=s.replace('angular_credit_confirm.json','angular_u0_control.json')
s=s.replace('3010000+i*211','3210000+i*211')
s=s.replace('3110000+ss*149','3310000+ss*149')
exec(compile(s,str(p),'exec'))
