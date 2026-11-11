import numpy as np,matplotlib,csv,collections,sys,os
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))); import en
O=sys.argv[1]; R=sys.argv[2]
INK='#222';MUT='#777';B='#1f6fb2';Or='#d9730d';G='#2e8b57';Rd='#b3322c';Pu='#6a4c93';Gy='#b8b8b8'
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':MUT,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,'font.family':'DejaVu Sans','legend.frameon':False})
def stamp(fig,t='MOCKUP (reviewer)'): fig.text(0.995,0.005,t,ha='right',va='bottom',fontsize=6,color=MUT)
# ---------- 1 LIF transfer
def lifr(I,beta,vth,T):
    V=np.zeros_like(I);n=np.zeros_like(I)
    for t in range(T):
        V=beta*V+I;s=V>vth;n+=s;V=np.where(s,0,V)
    return n/T
fig,ax=plt.subplots(1,2,figsize=(7,2.5),sharey=True)
for a,(beta,vth,lab) in zip(ax,[(.83,.6,'FSDD genome  β=0.83, $V_{th}$=0.6'),(.965,.96,'EEG genome  β=0.965, $V_{th}$=0.96')]):
    I=np.linspace(-.1,1.5*vth,6001); rb=vth*(1-beta)
    a.axvspan(I[0],rb,color='#eeeeee',lw=0)
    a.plot(I,np.clip(I,0,1),color=Gy,lw=1.6,label='clamp to [0,1]: the paper\'s stand-in')
    a.plot(I,1/(1+np.exp(-(I-vth*.5)/(vth/8))),color=G,lw=1.2,ls='--',label='sigmoid (rescaled)')
    a.plot(I,lifr(I,beta,vth,4),color=Or,lw=1.1,drawstyle='steps-post',label='LIF time mean, T=4')
    a.plot(I,lifr(I,beta,vth,2000),color=B,lw=1.1,drawstyle='steps-post',label='LIF time mean, T→∞ (2000)')
    a.axhline(.5,color=MUT,lw=.5,ls=':');a.axhline(1,color=MUT,lw=.5,ls=':')
    a.text(vth*1.06,.7,'no level between\n1/2 and 1 at ANY T',fontsize=6.5,color=INK)
    a.text(rb*.3 if rb>.05 else -.09,.92,'dead',fontsize=7,color=MUT)
    a.set_title(lab,fontsize=8,loc='left');a.set_xlabel('constant input current $I$');a.set_ylim(-.03,1.08)
ax[0].set_ylabel('output (spikes per step)');ax[1].legend(loc='lower right',fontsize=6.0,bbox_to_anchor=(1.02,.0))
fig.tight_layout();stamp(fig,'MOCKUP (reviewer): pure mathematics of Eq. 1 with hard reset, direct input, first hidden layer only');fig.savefig(O+'/m1_lif_transfer.png',dpi=220);plt.close(fig)
# ---------- 2 folds + digit
sp=['george','jackson','lucas','nicolas','theo','yweweler']
fig,ax=plt.subplots(1,2,figsize=(7.2,2.7),gridspec_kw={'width_ratios':[1,1.15]})
a=ax[0]
col={'test':Rd,'val':Or,'train':'#d7e6f2'}
for f in range(6):
    for b in range(6):
        k='test' if b==f else 'val' if b==(f+1)%6 else 'train'
        a.add_patch(Rectangle((b,5-f),.94,.9,color=col[k],lw=0))
        a.text(b+.47,5-f+.45,k[:2].upper() if k!='train' else '',ha='center',va='center',fontsize=6,color='white')
a.add_patch(Rectangle((-.08,4.95),6.1,1.0,fill=False,ec=INK,lw=1.2,ls='--'))
a.annotate('',xy=(1.47,4.92),xytext=(1.47,5.0),arrowprops=dict(arrowstyle='->',color=INK,lw=1.2))
a.text(6.1,5.4,'fold 0: design\nstudy looked here',fontsize=6.5,va='center')
a.text(6.1,4.4,'jackson: seen as val,\nnow headline TEST',fontsize=6.5,va='center',color=Rd)
a.text(6.1,.4,'george (dev TEST)\nnow VAL',fontsize=6.5,va='center',color=Or)
a.set_xlim(-.1,8.6);a.set_ylim(-.1,6.05);a.set_xticks(np.arange(6)+.47);a.set_xticklabels(sp,rotation=45,ha='right',fontsize=6.5)
a.set_yticks(np.arange(6)+.45);a.set_yticklabels([f'fold {f}' for f in range(5,-1,-1)],fontsize=6.5);a.tick_params(length=0)
for s in a.spines.values(): s.set_visible(False)
a.set_title('Every headline fold reuses dev-fold speakers',fontsize=8,loc='left')
a=ax[1]; M=np.zeros((6,10))
base=R+'/fsdd_c1_A/meeting01_t1_fsdd_c1_A_fsdd_fold0_target_%s_windows_meta.csv'
for code,s in ((1,'train'),(2,'val'),(3,'test')):
    for r in csv.DictReader(open(base%s)):
        rid=int(r['recording_id']); M[(rid//50)%6,rid//300]+=1 if False else 0
        M[(rid//50)%6,rid//300]=code
from matplotlib.colors import ListedColormap
a.imshow(M,cmap=ListedColormap(['#f3f3f3',B,Or,Rd]),vmin=0,vmax=3,aspect='auto')
a.set_xticks(range(10));a.set_yticks(range(6));a.set_yticklabels(sp,fontsize=6.5);a.set_xlabel('spoken digit');a.tick_params(length=0)
for s in a.spines.values(): s.set_visible(False)
a.set_xticks(np.arange(-.5,10),minor=True);a.set_yticks(np.arange(-.5,6),minor=True);a.grid(which='minor',color='white',lw=1.2);a.tick_params(which='minor',length=0)
a.set_title('FSDD fold 0: what the window caps keep',fontsize=8,loc='left')
for x,y,t,c in ((2,2.5,'train: 0-4 only','white'),(7,2.5,'never seen by any model','#555'),(1,1,'val: 0-2','white'),(4.5,0,'test: all ten','white')): a.text(x,y,t,ha='center',va='center',fontsize=6.5,color=c)
fig.tight_layout();stamp(fig,'MOCKUP (reviewer): left = fold rule of the paper; right = real window metadata');fig.savefig(O+'/m2_folds_and_digits.png',dpi=220);plt.close(fig)
# ---------- 3 first-layer break-even
fig,ax=plt.subplots(1,2,figsize=(7.2,2.7),gridspec_kw={'width_ratios':[1.05,1]})
a=ax[0]
rows=[('ANN-AE',en.ann(256,[59,16])),
      ('SNN direct T=4',en.snn(256,[59,16],4,'direct',.39)),
      ('SNN direct T=28',en.snn(256,[59,16],28,'direct',.448)),
      ('SNN Poisson T=4',en.snn(256,[59,16],4,'poisson',.39,.5)),
      ('SNN Poisson T=19',en.snn(256,[59,16],19,'poisson',.39,.5))]
parts=['first layer','LIF + potentials','to latent','time mean','decoder'];cols=[B,Rd,G,Pu,Gy]
y=np.arange(len(rows))[::-1]
for yy,(n,d) in zip(y,rows):
    x=0
    for p,c in zip(parts,cols):
        v=d.get(p,0);a.barh(yy,v,left=x,color=c,height=.62,edgecolor='white',lw=.6,label=p if n=='SNN Poisson T=19' else None);x+=v
    a.text(x+60,yy,f'{x:,.0f} nJ  ({x/854.485:.1f}×)',va='center',fontsize=6.5)
a.set_yticks(y);a.set_yticklabels([r[0] for r in rows],fontsize=7);a.set_xlim(0,9800);a.set_xlabel('E per window (nJ), widths [59,16]')
a.legend(fontsize=6.3,loc='upper right',bbox_to_anchor=(1.02,1.0));a.set_title('Half the ANN bill is a decoder both arms share',fontsize=8,loc='left')
a=ax[1]
th=np.linspace(0,3,301)
for T,c in ((4,Or),(16,Pu)):
    E=[]
    for t in th:
        e=en.L();en.spk(e,t*256,T,256,59,False);E.append(e.nj())
    a.plot(th,np.array(E),color=c,lw=1.4,label=f'spike-fed first layer, T={T}')
A=en.ann(256,[59,16])['first layer'];a.axhline(A,color=B,lw=1.4,label='analog first layer (ANN, or direct once)')
for x,l,ha in ((1.0,'latency,\nany T\n(one spike\nper sample)','left'),(2.0,'Poisson p≈0.5,\nT=4 → 2.0\n(T=19 → 9.5)','left')):
    a.axvline(x,color=MUT,lw=.7,ls=':');a.text(x+.03,1420,l,fontsize=6.3,va='top',ha=ha)
e0=[];be=None
for t,E_ in zip(th,E): pass
a.set_xlabel('input spikes per sample per window, θ/M');a.set_ylabel('first-layer E (nJ)');a.set_ylim(0,1450);a.set_xlim(0,3)
a.legend(fontsize=6.2,loc='lower right');a.set_title('A spike only wins below ~0.6 spikes per sample',fontsize=8,loc='left')
fig.tight_layout();stamp(fig,'MOCKUP (reviewer): counts of Table II re-implemented; reproduces 854.4853 nJ and the 11 017 nJ first-layer difference; spike rates are parameters, not results');fig.savefig(O+'/m3_energy_structure.png',dpi=220);plt.close(fig)
# breakeven numbers
for T in (2,4,16,32):
    lo,hi=0,3
    for _ in range(40):
        m=(lo+hi)/2;e=en.L();en.spk(e,m*256,T,256,59,False)
        if e.nj()>A: hi=m
        else: lo=m
    print('break-even theta/M T',T,round(lo,3))
# ---------- 4 time mean permutation
rng=np.random.default_rng(3)
T=16;N=8
I=np.array([.12,.15,.2,.25,.32,.4,.55,.7]);beta=.83;vth=.6
V=np.zeros(N);S=np.zeros((T,N))
for t in range(T):
    V=beta*V+I;s=V>vth;S[t]=s;V=np.where(s,0,V)
Sp=np.array([rng.permutation(S[:,i]) for i in range(N)]).T
Sq=np.zeros_like(S)
for i in range(N):
    Sq[:int(S[:,i].sum()),i]=1
fig,ax=plt.subplots(1,4,figsize=(7.2,2.1),gridspec_kw={'width_ratios':[1,1,1,.8]})
for a,X,t in zip(ax,(S,Sp,Sq),('LIF, as it fired','same counts,\ntimes shuffled','same counts,\nall at the start')):
    yy,xx=np.nonzero(X.T);a.scatter(xx,yy,marker='|',s=38,color=INK,lw=1.1);a.set_xlim(-.5,T-.5);a.set_ylim(-.6,N-.4);a.set_title(t,fontsize=7.5,loc='left');a.set_xlabel('step t');a.set_yticks(range(N));a.tick_params(labelsize=6)
ax[0].set_ylabel('hidden neuron')
W=rng.normal(size=(N,1))/2
a=ax[3];z=[(X.mean(0)*W[:,0]).sum() for X in (S,Sp,Sq)]
a.barh([2,1,0],z,color=[B,B,B],height=.55);a.set_yticks([2,1,0]);a.set_yticklabels(['as fired','shuffled','at start'],fontsize=6.5);a.set_title('latent $\\bar z$ = W·mean(s)',fontsize=7.5,loc='left');a.tick_params(axis='x',labelsize=6)
fig.tight_layout();stamp(fig,'MOCKUP (reviewer): simulation of Eq. 1, one latent unit, random weights');fig.savefig(O+'/m4_time_mean.png',dpi=220);plt.close(fig)
