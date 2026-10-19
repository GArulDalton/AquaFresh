import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aquafresh
from aquafresh import MODELS,RESULTS,FIGURES
import json,numpy as np,glob
from evaluate import es_policy
from env import Ocean
S=list(range(100,106))
def ev(f,kw,**x):
    A=[];E=[];R=[]
    for sd in S:
        e=Ocean(seed=sd,**{**kw,**x}); e.reset(seed=sd); d=False; r0=0
        while not d: _,r,d=e.step(f(e)); r0+=r
        A.append(np.mean(e.hist)); E.append((e.stats['Eac']+e.stats['Eopt'])/e.N); R.append(r0)
    return np.mean(A),np.mean(E),np.mean(R)
out={}
for tag in ['aquafresh','no_attn']:
    ps=[es_policy(p) for p in sorted(glob.glob(os.path.join(MODELS,f'{tag}_s*.pkl')))[:2]]
    for x in [dict(K=5),dict(N=100)]:
        r=np.mean([ev(f,kw,**x) for f,kw in ps],0); out[f'{tag}_{list(x.items())[0]}']=[float(v) for v in r]; print(tag,x,np.round(r,1),flush=True)
json.dump(out,open(os.path.join(RESULTS,'scal.json'),'w'))
