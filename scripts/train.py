import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aquafresh
from aquafresh import MODELS,RESULTS,FIGURES
import argparse
from es import es_train, CFG
p=argparse.ArgumentParser(description="Train AquaFresh (or an ablation) with antithetic evolution strategies")
p.add_argument('config',choices=sorted(CFG)); p.add_argument('--seeds',default='0'); p.add_argument('--gens',type=int,default=50)
a=p.parse_args()
for sd in [int(x) for x in a.seeds.split(',')]: es_train(a.config,sd,gens=a.gens,**CFG[a.config])
