import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aquafresh
from aquafresh import MODELS,RESULTS,FIGURES
import numpy as np, json
from evaluate import run_policy, summ
from baselines import greedy_nn_acoustic, TSPOptical, aoi_greedy_hybrid, lyapunov_maxweight
S=list(range(100,110)); out={}
for n,p in [('Greedy-NN (acoustic)',greedy_nn_acoustic),('TSP tour (optical)',TSPOptical),('AoI-greedy hybrid',aoi_greedy_hybrid),('Lyapunov max-weight',lyapunov_maxweight)]:
    out[n]=summ(run_policy(p,S)); print(n,{k:round(v,2) for k,v in out[n].items()})
json.dump(out,open(os.path.join(RESULTS,'baselines.json'),'w'),indent=1)
