import numpy as np, jax, jax.numpy as jnp, pickle, time, sys
from env import Ocean
FC,FA,FG,H=12,6,4,64
def init(key):
    ks=jax.random.split(key,12); n=lambda k,a,b,s=None:jax.random.normal(k,(a,b))*(s or np.sqrt(2/a))
    return {'c1':n(ks[0],FC,H),'cb1':jnp.zeros(H),'c2':n(ks[1],H,H),'cb2':jnp.zeros(H),'a1':n(ks[2],FA,H),'ab1':jnp.zeros(H),
            'Wq':n(ks[3],H,H,0.1),'Wk':n(ks[4],H,H,0.1),'Wv':n(ks[5],H,H,0.1),'ho':n(ks[6],H,2,0.01),'hh':n(ks[7],H,1,0.01),
            'beta':jnp.array([3.0]),'v1':n(ks[8],H+FG,H),'vb1':jnp.zeros(H),'v2':n(ks[9],H,1,0.01)}
def forward(p,C,A,g,Pr,attn,prior=True):
    h=jax.nn.relu(C@p['c1']+p['cb1']); h=jax.nn.relu(h@p['c2']+p['cb2']); u=jax.nn.relu(A@p['a1']+p['ab1'])
    if attn:
        K=C.shape[0]; nodes=jnp.concatenate([h,jnp.broadcast_to(u[None],(K,K,H))],1)
        att=jnp.einsum('kmh,knh->kmn',h@p['Wq'],nodes@p['Wk'])/np.sqrt(H); z=h+jax.nn.softmax(att,-1)@(nodes@p['Wv'])
    else: z=h
    lo=z@p['ho']; logits=jnp.concatenate([lo[...,0],lo[...,1],z.mean(1)@p['hh']],-1)
    if prior: logits=logits+p['beta'][0]*Pr
    pool=jnp.concatenate([z.mean((0,1)),g]); v=(jax.nn.relu(pool@p['v1']+p['vb1'])@p['v2'])[0]
    return logits,v
class AquaFresh:
    def __init__(s,attn=True,seed=0,lr=1e-3,prior=True):
        s.attn=attn; s.p=init(jax.random.PRNGKey(seed)); s.lr=lr
        s.m=jax.tree_util.tree_map(jnp.zeros_like,s.p); s.v=jax.tree_util.tree_map(jnp.zeros_like,s.p); s.t=0
        s.prior=prior; s.fwd=jax.jit(lambda p,C,A,g,Pr:forward(p,C,A,g,Pr,attn,prior))
        def loss(p,C,A,g,Pr,act,oldlp,adv,ret):
            lg,v=jax.vmap(lambda c,a,gg,pr:forward(p,c,a,gg,pr,attn,prior))(C,A,g,Pr); lp=jax.nn.log_softmax(lg)
            lpa=jnp.take_along_axis(lp,act[...,None],-1)[...,0]; ratio=jnp.exp(lpa-oldlp); a=adv[:,None]
            pl=-jnp.mean(jnp.minimum(ratio*a,jnp.clip(ratio,0.8,1.2)*a)); ent=-jnp.mean(jnp.sum(jnp.exp(lp)*lp,-1))
            return pl+0.5*jnp.mean((v-ret)**2)-0.003*ent
        s.grad=jax.jit(jax.grad(loss))
    def act(s,obs,rng,greedy=False):
        lg,v=s.fwd(s.p,*obs); lg=np.asarray(lg,np.float64); lg=lg-lg.max(-1,keepdims=True); lp=lg-np.log(np.exp(lg).sum(-1,keepdims=True))
        a=lg.argmax(-1) if greedy else np.array([rng.choice(len(r),p=np.exp(r)/np.exp(r).sum()) for r in lp])
        return a,lp[np.arange(len(a)),a],float(v)
    def adam(s,g):
        s.t+=1; gn=jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(g))); sc=jnp.minimum(1.0,0.5/(gn+1e-8))
        g=jax.tree_util.tree_map(lambda x:x*sc,g)
        s.m=jax.tree_util.tree_map(lambda m,x:0.9*m+0.1*x,s.m,g); s.v=jax.tree_util.tree_map(lambda v,x:0.999*v+0.001*x*x,s.v,g)
        s.p=jax.tree_util.tree_map(lambda p,m,v:p-s.lr*(m/(1-0.9**s.t))/(jnp.sqrt(v/(1-0.999**s.t))+1e-8),s.p,s.m,s.v)
def train(name,seed,iters=160,eps_per_iter=4,attn=True,reward='aoi',prior=True,**envkw):
    ag=AquaFresh(attn=attn,seed=seed,prior=prior); rng=np.random.default_rng(seed); curve=[]; t0=time.time()
    for it in range(iters):
        B={k:[] for k in ['C','A','g','Pr','act','lp','r','v','d']}
        for ep in range(eps_per_iter):
            e=Ocean(seed=seed*100000+it*10+ep,**envkw); o=e.reset(); done=False; R=0
            while not done:
                a,lp,v=ag.act(o,rng); d0=e.stats['deliv']; o2,r,done=e.step(list(a))
                if reward=='thr': r=(e.stats['deliv']-d0)*0.5-0.5*(1-e.alive.mean())
                for k,x in zip(B,[o[0],o[1],o[2],o[3],a,lp,r,v,done]): B[k].append(x)
                o=o2; R+=r
            curve.append((R,np.mean(e.hist),(e.stats['Eac']+e.stats['Eopt'])/e.N))
        r=np.array(B['r']); v=np.array(B['v']+[0]); d=np.array(B['d'],float); adv=np.zeros_like(r); last=0
        for t in reversed(range(len(r))):
            nv=0 if d[t] else v[t+1]; delta=r[t]+0.98*nv-v[t]; last=delta+0.98*0.95*(0 if d[t] else last); adv[t]=last
        ret=adv+v[:-1]; adv=(adv-adv.mean())/(adv.std()+1e-8)
        C,A,g,Pr,act,lp=[jnp.array(np.stack(B[k])) for k in ['C','A','g','Pr','act','lp']]; n=len(r)
        for _ in range(4):
            idx=rng.permutation(n)
            for b in range(0,n,180):
                j=idx[b:b+180]; ag.adam(ag.grad(ag.p,C[j],A[j],g[j],Pr[j],act[j],lp[j],jnp.array(adv[j]),jnp.array(ret[j])))
        if it%40==0: print(name,seed,it,'aoi %.1f E %.1f'%tuple(np.mean([c[1:] for c in curve[-eps_per_iter:]],0)),'%.0fs'%(time.time()-t0),flush=True)
    pickle.dump({'p':jax.device_get(ag.p),'curve':curve,'cfg':dict(attn=attn,reward=reward,prior=prior,**envkw)},open(__import__('os').path.join(__import__('paths').MODELS,f'mappo_{name}_s{seed}.pkl'),'wb'))
    print(name,seed,'done %.0fs'%(time.time()-t0),flush=True)
CFG={'aquafresh':dict(),'af_l0':dict(lamE=0.0),'af_l2':dict(lamE=2.0),'mappo':dict(attn=False),'no_hybrid':dict(hybrid=False),
     'no_cur':dict(cur_aware=False),'thr_reward':dict(reward='thr'),'no_prior':dict(prior=False)}
if __name__=='__main__':
    name=sys.argv[1]; seeds=[int(x) for x in sys.argv[2].split(',')]; iters=int(sys.argv[3]) if len(sys.argv)>3 else 160
    for sd in seeds: train(name,sd,iters=iters,**CFG[name])
