import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'serif'})
fig,ax=plt.subplots(figsize=(7.6,4.6)); ax.axis('off'); ax.set_xlim(0,100); ax.set_ylim(0,62)
def box(x,y,w,h,t,c,fs=5.6):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.35',fc=c,ec='k',lw=0.8)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs)
def arr(x1,y1,x2,y2,t='',dy=1.0):
    ax.annotate('',(x2,y2),(x1,y1),arrowprops=dict(arrowstyle='->',lw=0.9))
    if t: ax.text((x1+x2)/2,(y1+y2)/2+dy,t,ha='center',fontsize=5.6,style='italic')
def group(x,y,w,h,t,c):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.5',fc=c,ec='0.45',lw=0.8,ls='--',alpha=0.5)); ax.text(x+0.8,y+h-1.6,t,fontsize=6.6,fontweight='bold',va='top')
group(0.5,33,40,28,'(a) Ocean-monitoring field','#e8f1fb'); group(44,33,55.5,28,'(b) Hybrid underwater channel','#eaf7ee')
group(0.5,1,99,29,'(c) AquaFresh decision engine (per AUV, every Δt = 20 s)','#fdf3e3')
box(2,48,17,8.5,'Seabed sensors\n(N = 60), sample\nevery 60 s, E₀ = 600 J','#ffffff')
box(22,48,17,8.5,'Priority hotspot\n(pollution / upwelling)\nwᵢ = 3, else 1','#ffffff')
box(2,35.5,17,9.5,'Double-gyre currents\nψ(x, y, t), peak ≤ 0.8 m/s\n(drifts AUV motion)','#ffffff')
box(22,35.5,17,9.5,'AUV fleet (K = 3)\n1.5 m/s, 2 FDMA ch.\n→ surface buoy / sink','#ffffff')
box(46,46.5,25,10.5,'Acoustic link (12 kHz, 4 kHz)\nThorp absorption + spreading\nWenz noise · Rayleigh fading\nco-channel interference\nresidual-Doppler ICI\nrange ≈ km, ≈10 kb/s, ≈10 J/pkt','#ffffff',fs=5.7)
box(73,46.5,25,10.5,'Optical link (UOWC)\nBeer–Lambert, Jerlov water\nc = 0.151 / 0.398 / 2.195 m⁻¹\nrange 17 / 9.5 / 2.8 m\n1 Mb/s, 0.064 J/pkt','#ffffff',fs=5.7)
box(52,35.2,40,8.2,'Delivery outcome\nAoI reset: Aᵢ ← age of newest sample + delay\nenergy: Eᵢ ← Eᵢ − eₜₓ','#ffffff',fs=6.0)
arr(58.5,46.5,66,43.8); arr(85.5,46.5,78,43.8); arr(39.4,52,45.6,52,'tx',0.8); arr(39.4,40,51.6,39.5,'AUV position',0.8)
box(2,15,17,10,'State observation\nAoI Aᵢ, priority wᵢ,\nenergy Eᵢ, SNR,\ntravel time τₖᵢ,\nother AUVs','#ffffff')
box(22,15,17,10,'Lyapunov prior\ndrift-plus-penalty\nscores (acoustic, optical)\n→ top-10 candidates\nprior $\\ell^0$','#ffffff')
box(42,15,17,10,'Graph-attention\nresidual (shared θ)\ncandidates ↔ AUVs\n$\\ell$ = β$\\ell^0$ + residual','#ffffff')
box(62,15,15,10,'Action (argmax $\\ell$)\n• acoustic transfer\n• optical approach\n• hold','#ffffff')
box(80,15,18,10,'Objective (per slot)\n−Āw/600 − λE/100\n− 0.5(1 − ρalive)','#ffffff')
arr(19.4,20,21.6,20); arr(39.4,20,41.6,20); arr(59.4,20,61.6,20); arr(77.4,20,79.6,20)
box(24,3,58,7.5,'Offline training: antithetic evolution strategies on the team episode return\n(10 mirrored pairs, common random numbers, Adam) → updates θ and β','#f3e8fb',fs=5.8)
arr(89,15,80.5,8.5,'fitness',0.3); arr(55,10.8,51,14.6,'θ',0.2)
arr(10.5,33,10.5,25.4,'observe',0.3); arr(69.5,25.4,69.5,34.8,'execute',0.3); arr(72,34.8,89,25.4,'AoI, energy',0.3)
import os
plt.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'figures','fig_block.png'),dpi=600,bbox_inches='tight')
