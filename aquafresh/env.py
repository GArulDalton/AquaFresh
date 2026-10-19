"""Ocean-monitoring IoUT environment: hybrid acoustic-optical data harvesting by multiple AUVs in a
time-varying double-gyre current field, with AoI, sensor energy, co-channel acoustic interference, Rayleigh
fading and residual-Doppler ICI."""
import numpy as np
C_SOUND=1500.0
JERLOV={'clear':0.151,'coastal':0.398,'turbid':2.195}
def thorp_db_per_km(f):
    f2=f**2; return 0.11*f2/(1+f2)+44*f2/(4100+f2)+2.75e-4*f2+0.003
def wenz_noise_psd_db(f,shipping=0.5,wind=5.0):
    Nt=17-30*np.log10(f); Ns=40+20*(shipping-0.5)+26*np.log10(f)-60*np.log10(f+0.03)
    Nw=50+7.5*np.sqrt(wind)+20*np.log10(f)-40*np.log10(f+0.4); Nth=-15+20*np.log10(f)
    return 10*np.log10(sum(10**(x/10) for x in [Nt,Ns,Nw,Nth]))
class Ocean:
    L=1500.0; depth=100.0; dt=20.0; T=180; v_auv=1.5; Ts=60.0; pkt=64e3; fwd=2.0
    fc=12.0; B=4e3; P_ac=1.0; P_circ=0.5; eta_t=0.2; SL_per_W=170.8; k_spread=2.0; eta_ac=0.5; eps_dopp_res=0.1; df_sub=10.0
    P_opt=1.0; R_opt=1e6; opt_th_db=-45.0; E0=600.0; E_idle=0.002; sinr_th_db=8.0
    def __init__(s,N=60,K=3,water='coastal',cur=0.4,M=10,seed=0,hybrid=True,cur_aware=True,lamE=0.5):
        s.N,s.K,s.water,s.cur,s.M=N,K,water,cur,M; s.rng=np.random.default_rng(seed)
        s.hybrid,s.cur_aware,s.lamE=hybrid,cur_aware,lamE
        s.c=JERLOV[water]; s.noise_db=wenz_noise_psd_db(s.fc,0.7,10.0)+10*np.log10(s.B); s.alpha=thorp_db_per_km(s.fc)
    def ac_snr_db(s,d):
        d=np.maximum(d,1.0); TL=s.k_spread*10*np.log10(d)+s.alpha*d/1000
        return s.SL_per_W+10*np.log10(s.P_ac*s.eta_t)-TL-s.noise_db
    def opt_gain_db(s,d):
        d=np.maximum(d,1.0); return 10*np.log10(np.exp(-s.c*d)*(0.05/d)**2*50)
    def opt_range(s):
        d=np.linspace(1,200,4000); ok=d[s.opt_gain_db(d)>=s.opt_th_db]; return ok.max() if len(ok) else 1.0
    def current(s,p,t):
        A=s.cur/np.pi; eps=0.25; om=2*np.pi/3600.0
        x=p[...,0]/s.L*2; y=p[...,1]/s.L; a=eps*np.sin(om*t); b=1-2*eps*np.sin(om*t); f=a*x**2+b*x; dfdx=2*a*x+b
        return np.stack([-np.pi*A*np.sin(np.pi*f)*np.cos(np.pi*y),np.pi*A*np.cos(np.pi*f)*np.sin(np.pi*y)*dfdx],-1)
    def reset(s,seed=None):
        r=s.rng if seed is None else np.random.default_rng(seed); s.r=r
        s.sp=r.uniform(50,s.L-50,(s.N,2)); hot=np.array([s.L/2,s.L/2])+r.uniform(-200,200,2)
        s.w=np.where(np.linalg.norm(s.sp-hot,axis=1)<400,3.0,1.0)
        s.ap=r.uniform(0,s.L,(s.K,2)); s.t=0.0; s.aoi=r.uniform(0,600,s.N); s.E=np.full(s.N,s.E0)
        s.alive=np.ones(s.N,bool); s.target=-np.ones(s.K,int); s.vrel=np.full(s.K,0.3); s.rOpt=s.opt_range(); s.step_i=0
        s.stats=dict(ac=0,opt=0,fail=0,Eac=0.0,Eopt=0.0,dist=0.0,deliv=0,peak=[]); s.hist=[]
        return s.obs()
    def travel_time(s,k,idx):
        d=s.sp[idx]-s.ap[k]; dist=np.linalg.norm(d,axis=-1)+1e-6
        if not s.cur_aware: return dist/s.v_auv
        c=s.current(s.ap[k],s.t); return dist/np.maximum(s.v_auv+(d@c)/dist,0.3)
    V_lyap=400.0
    def lyap_scores(s,k):
        tt=s.travel_time(k,np.arange(s.N)); d=np.hypot(np.linalg.norm(s.sp-s.ap[k],axis=1),s.depth)
        snr=s.ac_snr_db(d); p_ok=np.exp(-10**(s.sinr_th_db/10)/10**(np.minimum(snr,13)/10))
        S=np.stack([s.w*s.aoi*p_ok-s.V_lyap*20/np.maximum(s.E,1), s.w*s.aoi*np.exp(-tt/300)-s.V_lyap*0.1/np.maximum(s.E,1)],1)
        S[~s.alive]=-1e9; return S
    def candidates(s,k):
        S=s.lyap_scores(k); s._S[k]=S; return np.argsort(-S.max(1))[:s.M]
    def obs(s):
        cl=[]; s._S={}
        for k in range(s.K):
            c=s.candidates(k)
            if s.target[k]>=0 and s.alive[s.target[k]]:   # pin current target to slot 0 (commitment / multi-step optical approach)
                c=np.concatenate([[s.target[k]],[x for x in c if x!=s.target[k]]])[:s.M]
            cl.append(c)
        s.cand=np.stack(cl); Fc=[];Fa=[]
        for k in range(s.K):
            j=s.cand[k]; d=s.sp[j]-s.ap[k]; dist=np.linalg.norm(d,axis=1); snr=s.ac_snr_db(np.hypot(dist,s.depth)); tt=s.travel_time(k,j)
            claimed=np.array([float(any(s.target[q]==jj for q in range(s.K) if q!=k)) for jj in j])
            oth=np.array([min([np.linalg.norm(s.sp[jj]-s.ap[q]) for q in range(s.K) if q!=k]+[s.L]) for jj in j])/s.L
            c=s.current(s.ap[k],s.t) if s.cur_aware else np.zeros(2)
            Fc.append(np.stack([d[:,0]/s.L,d[:,1]/s.L,dist/s.L,s.aoi[j]/600,s.w[j]/3,s.E[j]/s.E0,(snr-10)/20,tt/600,claimed,oth,
                                np.full(s.M,c[0]),(np.arange(s.M)==0)*float(s.target[k]>=0)],1))
            Fa.append(np.concatenate([s.ap[k]/s.L,(s.sp[s.target[k]]-s.ap[k])/s.L if s.target[k]>=0 else [0,0],c]))
        P=[]
        for k in range(s.K):
            S=s._S[k][s.cand[k]]; sc=S.std()+1e-6; P.append(np.concatenate([(S[:,0]-S.max())/sc,(S[:,1]-S.max())/sc,[-6.0]]))
        s.prior=np.array(P,np.float32)
        g=np.array([np.average(s.aoi,weights=s.w)/600,s.alive.mean(),s.E.mean()/s.E0,s.t/(s.T*s.dt)])
        return np.array(Fc,np.float32),np.array(Fa,np.float32),g.astype(np.float32),s.prior
    def step(s,acts):
        E_before=s.stats['Eac']+s.stats['Eopt']; tx=[]
        for k,a in enumerate(acts):
            if isinstance(a,tuple): i,mode=a
            else:
                a=int(a)
                if a>=2*s.M: s.target[k]=-1; s.vrel[k]=0.3; continue
                mode='ac' if a<s.M else 'opt'; i=s.cand[k][a%s.M]
            if not s.hybrid: mode='ac'
            s.target[k]=i; d=s.sp[i]-s.ap[k]; dist=np.linalg.norm(d)
            stop=0.6*s.rOpt if mode=='opt' else (dist if s.ac_snr_db(np.hypot(dist,s.depth))>=20 else 0.0)
            s.vrel[k]=s.v_auv if dist>stop else 0.3
            if dist>stop:
                h=d/dist; c=s.current(s.ap[k],s.t); stp=min(s.v_auv*s.dt,dist-stop)
                s.ap[k]=np.clip(s.ap[k]+h*stp+c*s.dt*(stp/(s.v_auv*s.dt)),0,s.L); s.stats['dist']+=stp
            else: s.ap[k]=np.clip(s.ap[k]+s.current(s.ap[k],s.t)*s.dt*0.2,0,s.L)
            if mode=='opt':
                d2=np.linalg.norm(s.sp[i]-s.ap[k])
                if d2<=s.rOpt: tx.append((k,i,'opt',d2))
            elif s.alive[i]: tx.append((k,i,'ac',np.hypot(np.linalg.norm(s.sp[i]-s.ap[k]),s.depth)))
        served=set(); ac=[t for t in tx if t[2]=='ac']
        for (k,i,mode,d2) in tx:
            if i in served or not s.alive[i]: continue
            if mode=='opt': e=s.P_opt*s.pkt/s.R_opt; ok=True
            else:
                sig=10**(s.ac_snr_db(d2)/10)
                I=sum(10**(s.ac_snr_db(np.hypot(np.linalg.norm(s.sp[i2]-s.ap[k]),s.depth))/10)*s.r.exponential()
                      for (k2,i2,m2,_) in ac if i2!=i and k2%2==k%2)
                sinr=sig*s.r.exponential()/(1+I)
                fd=s.eps_dopp_res*s.vrel[k]*s.fc*1e3/C_SOUND; ici=(np.pi*fd/s.df_sub)**2/3; sinr=sinr/(1+sinr*ici)
                R=s.eta_ac*s.B*np.log2(1+sinr); ttx=s.pkt/max(R,1e-6); e=(s.P_ac+s.P_circ)*min(ttx,s.dt)
                ok=(ttx<=s.dt-d2/C_SOUND) and sinr>=10**(s.sinr_th_db/10)
            if s.E[i]<e: s.alive[i]=False; continue
            s.E[i]-=e; s.stats['E'+mode]+=e
            if ok:
                served.add(i); s.stats[mode]+=1; s.stats['deliv']+=1; s.stats['peak'].append(s.aoi[i])
                s.aoi[i]=(s.t%s.Ts)+s.fwd+(d2/C_SOUND if mode=='ac' else 0)
            else: s.stats['fail']+=1
        s.E-=s.E_idle*s.dt; s.alive&=s.E>0; s.t+=s.dt; s.aoi+=s.dt; s.step_i+=1
        wa=np.average(s.aoi,weights=s.w); eu=s.stats['Eac']+s.stats['Eopt']-E_before
        s.hist.append(wa)
        return s.obs(),-(wa/600)-0.5*(1-s.alive.mean())-s.lamE*eu/100.0,s.step_i>=s.T
