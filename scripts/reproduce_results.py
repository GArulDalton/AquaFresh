import os,sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aquafresh
from aquafresh import MODELS,RESULTS,FIGURES
import numpy as np,pickle,glob,json,time
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from evaluate import *
import es
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.grid':True,'grid.alpha':0.3,'figure.dpi':200,'savefig.dpi':600})
S=list(range(100,110)); KEYS=['aoi','paoi','deliv','optfrac','fail','Etot','Epp','dist','R']
def evalR(pol,seeds=S,envkw={},**kw):
    out=[]
    for sd in seeds:
        e=Ocean(seed=sd,**{**envkw,**kw}); e.reset(seed=sd); P=pol(e) if isinstance(pol,type) else pol; done=False; R=0
        while not done: _,r,done=e.step(P(e)); R+=r
        st=e.stats; out.append(dict(aoi=np.mean(e.hist),paoi=np.mean(st['peak']) if st['peak'] else np.nan,deliv=st['deliv'],
            optfrac=st['opt']/max(st['deliv'],1),fail=st['fail'],Etot=(st['Eac']+st['Eopt'])/e.N,Epp=(st['Eac']+st['Eopt'])/max(st['deliv'],1),
            dist=st['dist']/1000,R=R,hist=np.array(e.hist),alive=e.alive.mean()))
    return out
def m(out): return {k:float(np.nanmean([o[k] for o in out])) for k in KEYS+['alive']}
def ms(list_of_outs):
    M=[m(o) for o in list_of_outs]; return {k:[float(np.mean([x[k] for x in M])),float(np.std([x[k] for x in M]))] for k in M[0]}
p0=es.Policy(); th0=p0.theta
BASE={'Greedy-NN (acoustic)':greedy_nn_acoustic,'TSP tour (optical)':TSPOptical,'AoI-greedy hybrid':aoi_greedy_hybrid,
      'Lyapunov max-weight':lyapunov_maxweight,'Prior only (untrained)':(lambda e: list(p0.act(th0,e.obs())))}
def lpols(tag): return [es_policy(p) for p in sorted(glob.glob(os.path.join(MODELS,f'{tag}_s*.pkl')))]
import os
FIGONLY=os.path.exists(os.path.join(RESULTS,'results.json')) and '--recompute' not in sys.argv
R={}; OUT={}
for n,p in BASE.items():
    if FIGONLY and n not in ['TSP tour (optical)','AoI-greedy hybrid','Lyapunov max-weight']: continue
    o=evalR(p); OUT[n]=o; R[n]={k:[v,0.0] for k,v in m(o).items()}
TAGS={'AquaFresh':'aquafresh','w/o graph attention':'no_attn','w/o Lyapunov prior':'no_prior','w/o hybrid (acoustic only)':'no_hybrid',
      'w/o current awareness':'no_cur','AquaFresh (λ=0)':'af_l0','AquaFresh (λ=2)':'af_l2'}
for n,t in TAGS.items():
    if FIGONLY and n!='AquaFresh': continue
    ps=lpols(t)
    if not ps: continue
    outs=[evalR(f,envkw=kw) for f,kw in ps]; OUT[n]=outs[0]; R[n]=ms(outs)
# sensitivity (zero-shot generalisation)
SENS0={'water':[('clear',{}),('coastal',{}),('turbid',{})],'cur':[(0.0,{}),(0.4,{}),(0.8,{})],'N':[(40,{}),(60,{}),(100,{})],'K':[(2,{}),(3,{}),(5,{})]}
sens={}
SENS={} if FIGONLY else SENS0
AF=lpols('aquafresh')
for ax,vals in SENS.items():
    sens[ax]={}
    for val,_ in vals:
        kw={ax:val}; row={}
        for n in ['AoI-greedy hybrid','Lyapunov max-weight']: row[n]=m(evalR(BASE[n],S[:6],**kw))
        row['AquaFresh']=ms([evalR(f,S[:6],envkw=k2,**kw) for f,k2 in AF[:3]])
        row['AquaFresh']={k:v[0] for k,v in row['AquaFresh'].items()}
        sens[ax][str(val)]={n:{k:row[n][k] for k in ['aoi','Etot','R','optfrac']} for n in row}
# timing
f,kw=AF[0]; e=Ocean(); e.reset(seed=1); t=time.perf_counter()
for _ in range(200): f(e)
tdec=(time.perf_counter()-t)/200*1e3
if not FIGONLY: json.dump({'R':R,'sens':sens,'tdec':tdec},open(os.path.join(RESULTS,'results.json'),'w'),indent=1)
J=json.load(open(os.path.join(RESULTS,'results.json'))); R=J['R']; sens=J['sens']
# ---------- figures ----------
C=[]
for p in sorted(glob.glob(os.path.join(MODELS,'aquafresh_s*.pkl'))): C.append(pickle.load(open(p,'rb'))['curve'])
L=min(len(c) for c in C); C=np.array([c[:L] for c in C]); plt.figure(figsize=(3.5,2.4)); mu=C.mean(0); sd=C.std(0); x=np.arange(len(mu))
plt.plot(x,mu,label='AquaFresh (ES)'); plt.fill_between(x,mu-sd,mu+sd,alpha=0.25)
for n,c in [('Lyapunov max-weight','C1'),('AoI-greedy hybrid','C2')]: plt.axhline(R[n]['R'][0],ls='--',color=c,label=n)
plt.xlabel('ES generation'); plt.ylabel('Mean episode return'); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(FIGURES,'fig_learning.png'))
plt.figure(figsize=(3.5,2.4))
for n in ['TSP tour (optical)','AoI-greedy hybrid','Lyapunov max-weight','AquaFresh']:
    h=np.mean([o['hist'] for o in OUT[n]],0); plt.plot(np.arange(len(h))*20/60,h,label=n,lw=1)
plt.xlabel('Time (min)'); plt.ylabel('Weighted mean AoI (s)'); plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(FIGURES,'fig_aoi_time.png'))
plt.figure(figsize=(3.5,2.6))
for n,mk in [('Greedy-NN (acoustic)','v'),('TSP tour (optical)','s'),('AoI-greedy hybrid','^'),('Lyapunov max-weight','D'),('Prior only (untrained)','x')]:
    plt.scatter(R[n]['Etot'][0],R[n]['aoi'][0],marker=mk,label=n)
pts=[(R[n]['Etot'][0],R[n]['aoi'][0]) for n in ['AquaFresh (λ=0)','AquaFresh','AquaFresh (λ=2)'] if n in R]
if pts: pts=np.array(pts); plt.plot(pts[:,0],pts[:,1],'r-o',label='AquaFresh (λ = 0, 0.5, 2)')
plt.xlabel('Sensor energy (J / sensor / h)'); plt.ylabel('Weighted mean AoI (s)'); plt.legend(fontsize=6); plt.tight_layout(); plt.savefig(os.path.join(FIGURES,'fig_pareto.png'))
fig,ax=plt.subplots(1,4,figsize=(7.2,2.1))
for a,(axn,lab) in zip(ax,[('water','Water type'),('cur','Current (m/s)'),('N','Sensors N'),('K','AUVs K')]):
    vals=list(sens[axn].keys()); x=np.arange(len(vals)); w=0.27
    for i,n in enumerate(['AoI-greedy hybrid','Lyapunov max-weight','AquaFresh']):
        a.bar(x+i*w,[sens[axn][v][n]['aoi'] for v in vals],w,label=n)
    a.set_xticks(x+w); a.set_xticklabels(vals,fontsize=7); a.set_xlabel(lab); 
ax[0].set_ylabel('Weighted AoI (s)'); ax[0].legend(fontsize=5); plt.tight_layout(); plt.savefig(os.path.join(FIGURES,'fig_sens.png'))
# trajectory map
f,kw=AF[0]; e=Ocean(); e.reset(seed=103); tr=[e.ap.copy()]; done=False
while not done: _,_,done=e.step(f(e)); tr.append(e.ap.copy())
tr=np.array(tr); plt.figure(figsize=(3.5,3.3)); g=np.linspace(0,e.L,15); X,Y=np.meshgrid(g,g); U=e.current(np.stack([X,Y],-1),0.0)
plt.quiver(X,Y,U[...,0],U[...,1],color='0.7',width=0.003)
plt.scatter(e.sp[:,0],e.sp[:,1],c=np.where(e.w>1,'r','k'),s=10,label='sensors (red: hotspot)')
for k in range(e.K): plt.plot(tr[:,k,0],tr[:,k,1],lw=1,label=f'AUV {k+1}')
plt.xlabel('x (m)'); plt.ylabel('y (m)'); plt.legend(fontsize=6,loc='upper right'); plt.axis('equal'); plt.tight_layout(); plt.savefig(os.path.join(FIGURES,'fig_traj.png'))
print('ANALYSIS DONE')
