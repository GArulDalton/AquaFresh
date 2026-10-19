import numpy as np,pickle,sys
from env import Ocean
from mappo import AquaFresh
from baselines import *
def learned(path,greedy=True):
    d=pickle.load(open(path,'rb')); cfg=d['cfg']; ag=AquaFresh(attn=cfg['attn'],prior=cfg.get('prior',True)); ag.p=d['p']; rng=np.random.default_rng(0)
    envkw={k:v for k,v in cfg.items() if k in ('hybrid','cur_aware')}
    return (lambda e: list(ag.act(e.obs(),rng,greedy)[0])),envkw
def run_policy(pol,seeds,envkw={},**kw):
    out=[]
    for sd in seeds:
        e=Ocean(seed=sd,**{**envkw,**kw}); e.reset(seed=sd); P=pol(e) if isinstance(pol,type) else pol; done=False
        while not done: _,_,done=e.step(P(e))
        st=e.stats; out.append(dict(aoi=np.mean(e.hist),paoi=np.mean(st['peak']) if st['peak'] else np.nan,alive=e.alive.mean(),
            deliv=st['deliv'],ac=st['ac'],opt=st['opt'],fail=st['fail'],optfrac=st['opt']/max(st['deliv'],1),
            Epp=(st['Eac']+st['Eopt'])/max(st['deliv'],1),Etot=(st['Eac']+st['Eopt'])/e.N,dist=st['dist']/1000,hist=np.array(e.hist)))
    return out
def summ(out): return {k:float(np.nanmean([o[k] for o in out])) for k in out[0] if k!='hist'}
if __name__=='__main__':
    S=list(range(100,110))
    for n,p in [('aoi-greedy',aoi_greedy_hybrid),('lyap',lyapunov_maxweight)]: print(n,{k:round(v,2) for k,v in summ(run_policy(p,S)).items()})
    for path in sys.argv[1:]:
        f,kw=learned(path); print(path,{k:round(v,2) for k,v in summ(run_policy(f,S,kw)).items()})
def es_policy(path):
    import es
    d=pickle.load(open(path,'rb')); cfg=d['cfg']; pol=es.Policy(cfg['attn'],cfg['prior']); th=d['theta']
    envkw={k:v for k,v in cfg.items() if k in ('hybrid','cur_aware','harv_aware')}
    return (lambda e: list(pol.act(th,e.obs()))),envkw
