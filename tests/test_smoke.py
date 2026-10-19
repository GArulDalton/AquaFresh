"""Fast smoke tests: channel models, environment step, baselines, policy loading and one ES update."""
import os,sys,glob,json
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, aquafresh
from env import Ocean, thorp_db_per_km, JERLOV
from baselines import lyapunov_maxweight, aoi_greedy_hybrid

def test_channel_models():
    assert 1.0 < thorp_db_per_km(12.0) < 2.5                  # ~1.6 dB/km at 12 kHz
    e=Ocean(); assert e.ac_snr_db(100) > e.ac_snr_db(1000)    # SNR decreases with range
    r={w:Ocean(water=w).opt_range() for w in JERLOV}
    assert r['clear'] > r['coastal'] > r['turbid']            # optical range shrinks with turbidity

def test_episode_runs():
    e=Ocean(seed=1); e.reset(seed=1); d=False; n=0
    while not d: _,r,d=e.step(lyapunov_maxweight(e)); n+=1
    assert n==e.T and np.isfinite(r) and e.stats['deliv']>0

def test_pretrained_policy_loads():
    from evaluate import es_policy, run_policy, summ
    f,kw=es_policy(sorted(glob.glob(os.path.join(aquafresh.MODELS,'aquafresh_s*.pkl')))[0])
    s=summ(run_policy(f,[100],kw)); assert s['aoi']>0

def test_es_one_generation(tmp_path=None):
    import es
    os.environ['AQUAFRESH_MODELS']=os.path.join(aquafresh.ROOT,'results','_tmp')
    es.MODELS=os.environ['AQUAFRESH_MODELS']; os.makedirs(es.MODELS,exist_ok=True)
    es.es_train('smoke',0,gens=1,pairs=1,neps=1)
    assert os.path.exists(os.path.join(es.MODELS,'smoke_s0.pkl'))

if __name__=='__main__':
    for t in [test_channel_models,test_episode_runs,test_pretrained_policy_loads,test_es_one_generation]: t(); print('PASS',t.__name__)
