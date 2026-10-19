import numpy as np, jax, jax.numpy as jnp, pickle, time, sys
from jax.flatten_util import ravel_pytree
import mappo as agent; agent.H=16
from mappo import init, forward
from paths import MODELS
import os
from env import Ocean
class Policy:
    def __init__(s,attn=True,prior=True,seed=0):
        s.attn,s.prior=attn,prior; p=init(jax.random.PRNGKey(seed)); th,s.unravel=ravel_pytree(p); s.theta=np.array(th)
        s.fwd=jax.jit(lambda th,C,A,g,Pr: forward(s.unravel(th),C,A,g,Pr,attn,prior)[0])
    def np_params(s,th):
        if getattr(s,'_thid',None) is not th:
            s._p={k:np.asarray(v,dtype=np.float64) for k,v in s.unravel(jnp.asarray(th)).items()}; s._thid=th
        return s._p
    def logits_np(s,th,C,A,g,Pr):
        p=s.np_params(th); r=lambda x:np.maximum(x,0)
        h=r(C@p['c1']+p['cb1']); h=r(h@p['c2']+p['cb2']); u=r(A@p['a1']+p['ab1']); Hd=h.shape[-1]
        if s.attn:
            K=C.shape[0]; nodes=np.concatenate([h,np.broadcast_to(u[None],(K,K,Hd))],1)
            att=np.einsum('kmh,knh->kmn',h@p['Wq'],nodes@p['Wk'])/np.sqrt(Hd); att=np.exp(att-att.max(-1,keepdims=True)); att/=att.sum(-1,keepdims=True)
            z=h+att@(nodes@p['Wv'])
        else: z=h
        lo=z@p['ho']; lg=np.concatenate([lo[...,0],lo[...,1],z.mean(1)@p['hh']],-1)
        if s.prior: lg=lg+p['beta'][0]*Pr
        return lg
    def act(s,th,obs): return s.logits_np(th,*obs).argmax(-1)
def episode(pol,th,seed,**envkw):
    e=Ocean(seed=seed,**envkw); o=e.reset(seed=seed); done=False; R=0
    while not done: o,r,done=e.step(list(pol.act(th,o))); R+=r
    return R,e
def es_train(name,seed,gens=50,pairs=10,neps=2,sigma=0.05,lr=0.03,attn=True,prior=True,**envkw):
    pol=Policy(attn,prior,seed); th=pol.theta.copy(); rng=np.random.default_rng(seed); m=np.zeros_like(th); v=np.zeros_like(th)
    curve=[]; t0=time.time()
    for g in range(gens):
        seeds=[int(rng.integers(1e6)) for _ in range(neps)]
        eps=rng.standard_normal((pairs,len(th))); F=np.zeros(2*pairs)
        for i in range(pairs):
            for j,sg in enumerate([1,-1]):
                F[2*i+j]=np.mean([episode(pol,th+sg*sigma*eps[i],sd,**envkw)[0] for sd in seeds])
        rk=np.argsort(np.argsort(F)); u=rk/(len(F)-1)-0.5
        grad=sum((u[2*i]-u[2*i+1])*eps[i] for i in range(pairs))/(2*pairs*sigma)
        m=0.9*m+0.1*grad; v=0.999*v+0.001*grad**2; th=th+lr*(m/(1-0.9**(g+1)))/(np.sqrt(v/(1-0.999**(g+1)))+1e-8)-lr*0.005*th
        curve.append(F.mean())
        if g%10==0: print(name,seed,g,'F %.2f'%F.mean(),'%.0fs'%(time.time()-t0),flush=True)
    pickle.dump({'theta':th,'curve':curve,'cfg':dict(attn=attn,prior=prior,**envkw)},open(os.path.join(MODELS,f'{name}_s{seed}.pkl'),'wb'))
    print(name,seed,'done %.0fs'%(time.time()-t0),flush=True)
CFG={'aquafresh':dict(),'af_l0':dict(lamE=0.0),'af_l2':dict(lamE=2.0),'no_attn':dict(attn=False),'no_hybrid':dict(hybrid=False),
     'no_cur':dict(cur_aware=False),'no_prior':dict(prior=False)}
if __name__=='__main__':
    name=sys.argv[1]; seeds=[int(x) for x in sys.argv[2].split(',')]; gens=int(sys.argv[3]) if len(sys.argv)>3 else 50
    for sd in seeds: es_train(name,sd,gens=gens,**CFG[name])
