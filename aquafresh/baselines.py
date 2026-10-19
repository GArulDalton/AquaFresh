import numpy as np
from env import Ocean
def _claimed(e,k): return {e.target[q] for q in range(e.K) if q!=k}
def greedy_nn_acoustic(e):
    acts=[]; taken=set()
    for k in range(e.K):
        d=np.linalg.norm(e.sp-e.ap[k],axis=1); d[~e.alive]=1e9; d[e.aoi<e.Ts]=1e9
        for i in taken|_claimed(e,k):
            if i>=0: d[i]=1e9
        i=int(np.argmin(d)); taken.add(i); acts.append((i,'ac'))
    return acts
class TSPOptical:
    def __init__(s,e):
        r=np.random.default_rng(0); C=e.sp[r.choice(e.N,e.K,replace=False)]
        for _ in range(20):
            lab=np.argmin(((e.sp[:,None]-C[None])**2).sum(-1),1)
            C=np.array([e.sp[lab==k].mean(0) if (lab==k).any() else C[k] for k in range(e.K)])
        s.tours=[]
        for k in range(e.K):
            idx=list(np.where(lab==k)[0]); tour=[]; p=C[k]
            while idx:
                j=min(idx,key=lambda i:np.linalg.norm(e.sp[i]-p)); tour.append(j); idx.remove(j); p=e.sp[j]
            imp=True
            while imp and len(tour)>3:
                imp=False
                for a in range(len(tour)-2):
                    for b in range(a+2,len(tour)):
                        P=lambda i:e.sp[tour[i%len(tour)]]
                        if np.linalg.norm(P(a)-P(b))+np.linalg.norm(P(a+1)-P(b+1))<np.linalg.norm(P(a)-P(a+1))+np.linalg.norm(P(b)-P(b+1))-1e-6:
                            tour[a+1:b+1]=tour[a+1:b+1][::-1]; imp=True
            s.tours.append(tour)
        s.pos=[0]*e.K
    def __call__(s,e):
        acts=[]
        for k in range(e.K):
            t=s.tours[k]
            if not t: acts.append(2*e.M); continue
            i=t[s.pos[k]%len(t)]
            if e.aoi[i]<e.Ts+e.fwd+1 or not e.alive[i]: s.pos[k]+=1; i=t[s.pos[k]%len(t)]
            acts.append((i,'opt'))
        return acts
def aoi_greedy_hybrid(e,thr=0.3):
    acts=[]; taken=set()
    for k in range(e.K):
        tt=e.travel_time(k,np.arange(e.N)); sc=e.w*(e.aoi+tt)/(tt+e.dt); sc[~e.alive]=-1
        for i in taken|_claimed(e,k):
            if i>=0: sc[i]=-1
        i=int(np.argmax(sc)); taken.add(i)
        acts.append((i,'opt' if (tt[i]<=2*e.dt or e.E[i]<thr*e.E0) else 'ac'))
    return acts
def lyapunov_maxweight(e,V=400.0):
    acts=[]; taken=set()
    for k in range(e.K):
        tt=e.travel_time(k,np.arange(e.N)); d=np.hypot(np.linalg.norm(e.sp-e.ap[k],axis=1),e.depth)
        snr=e.ac_snr_db(d); p_ok=np.exp(-10**0.8/10**(np.minimum(snr,13)/10))
        S=np.stack([e.w*e.aoi*p_ok-V*20/np.maximum(e.E,1), e.w*e.aoi*np.exp(-tt/300)-V*0.1/np.maximum(e.E,1)],1); S[~e.alive]=-1e9
        for i in taken|_claimed(e,k):
            if i>=0: S[i]=-1e9
        i,m=np.unravel_index(np.argmax(S),S.shape); taken.add(int(i)); acts.append((int(i),'ac' if m==0 else 'opt'))
    return acts
