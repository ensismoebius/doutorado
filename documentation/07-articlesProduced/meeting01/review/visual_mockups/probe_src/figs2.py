import numpy as np,matplotlib,csv,collections,sys,os,json
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
O=sys.argv[1]; R=sys.argv[2]; SC=sys.argv[3]
INK='#222';MUT='#777';B='#1f6fb2';Or='#d9730d';G='#2e8b57';Rd='#b3322c';Pu='#6a4c93';Gy='#b8b8b8'
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':MUT,'font.family':'DejaVu Sans','legend.frameon':False})
def stamp(fig,t): fig.text(0.995,0.005,t,ha='right',va='bottom',fontsize=6,color=MUT)
# T1 strip
rows=list(csv.DictReader(open(R+'/t1_summary.csv')))
fig,ax=plt.subplots(2,1,figsize=(3.5,3.1),sharex=True)
refs={'fsdd':dict(PCA=.649,ANN=.661,SNN=.713),'eegmmidb':dict(PCA=.183,ANN=.291,SNN=.407)}
for a,ds,lab in zip(ax,('fsdd','eegmmidb'),('FSDD fold 0','EEGMMIDB fold 0')):
    r=[x for x in rows if x['dataset']==ds and x['training']=='final']; mp=float(r[0]['mean_predictor_test_mse'])
    for i,arm in enumerate('ABC'):
        v=[x for x in r if x['arm']==arm]
        for j,x in enumerate(v):
            m=float(x['test_mse']);ar=int(x['armed_epoch'])>0
            a.scatter(m,2-i+(j-1.5)*.13,s=20,marker='o',facecolor=Or if ar else 'white',edgecolor=Or,lw=.9,zorder=3)
    a.axvline(mp,color=INK,lw=1);a.axvline(.9*mp,color=INK,lw=.7,ls='--')
    a.text(mp+.005,2.55,'mean',fontsize=6,ha='left');a.text(.9*mp-.005,2.55,'-10%',fontsize=6,ha='right')
    for k,c in (('PCA',G),('ANN',B),('SNN',Rd)):
        a.axvline(refs[ds][k],color=c,lw=1);a.text(refs[ds][k]+{'PCA':-.012,'ANN':.0,'SNN':.012}[k] if ds=='fsdd' else refs[ds][k],2.55 if not(ds=='fsdd' and k=='ANN') else 2.78,k,fontsize=6,color=INK,ha={'PCA':'right','ANN':'center','SNN':'left'}[k] if ds=='fsdd' else 'center')
    a.set_yticks([2,1,0]);a.set_yticklabels(['control','warm-up','pre-norm'],fontsize=7);a.set_ylim(-.5,2.5);a.set_title(lab,fontsize=7.5,loc='left',pad=9)
    a.grid(axis='x',color='#eee',lw=.5)
ax[1].set_xlabel('test MSE of the final fits (filled: armed)');ax[1].set_xlim(.15,1.05)
fig.tight_layout();stamp(fig,'MOCKUP: real t1_summary.csv; PCA/ANN/SNN from QT (job 5315335)');fig.savefig(O+'/m5_t1_plateau.png',dpi=220);plt.close(fig)
# design dots
fig,ax=plt.subplots(1,2,figsize=(7.2,2.7),sharey=True)
L=['ANN, leaky ReLU','ANN, sigmoid','ANN, clamp [0,1]','SNN per-step, T=16','SNN time mean, T=16','SNN latent mean, T=4','SNN latent mean, T=16']
K={'f':[.667,.685,.743,.780,.734,.724,.729],'e':[.489,.308,.547,.592,.454,.484,.457]}
X={'f':[.654,None,None,None,None,.729,None],'e':[.293,None,None,.589,None,.449,.396]}
g=json.load(open(SC+'/probe/grid.json'))
def m(ds,k,T=0):
    v=[x['test'] for x in g if x['ds']==ds and x['kind']==k and x['T']==T];return float(np.mean(v)) if v else None
P={'f':[m('fsdd','lrelu'),m('fsdd','sigmoid'),m('fsdd','clamp'),None,None,m('fsdd','lif',4),None],'e':[m('eegmmidb','lrelu'),m('eegmmidb','sigmoid'),m('eegmmidb','clamp'),None,None,m('eegmmidb','lif',4),m('eegmmidb','lif',16)]}
for a,d,t,pca in zip(ax,'fe',('FSDD, dev fold','EEGMMIDB, dev fold'),(.649,.183)):
    y=np.arange(7)[::-1]
    a.axhline(3.5,color='#ddd',lw=.8)
    for yy,k,x,p in zip(y,K[d],X[d],P[d]):
        a.scatter(k,yy+.17,s=26,facecolor='white',edgecolor=Rd,lw=1,zorder=3)
        if x is not None: a.scatter(x,yy,s=26,color=B,zorder=3)
        else: a.text(a.get_xlim()[0],yy,'',fontsize=5)
        if p is not None: a.scatter(p,yy-.17,s=24,marker='D',color=G,zorder=3)
    a.axvline(pca,color=MUT,lw=1,ls='--');a.text(pca,6.6,'PCA',fontsize=6.5,ha='center',color=INK)
    a.set_title(t,fontsize=8,loc='left',pad=10);a.set_xlabel('test MSE');a.grid(axis='x',color='#eee',lw=.5);a.tick_params(axis='y',length=0)
ax[0].set_yticks(y);ax[0].set_yticklabels(L,fontsize=7)
ax[0].scatter([],[],s=26,facecolor='white',edgecolor=Rd,label='Table III, Kaiming (both layers for ANN rows)')
ax[0].scatter([],[],s=26,color=B,label='Table III, Xavier')
ax[0].scatter([],[],s=24,marker='D',color=G,label='reviewer probe: Xavier, encoder only, 2 seeds')
fig.legend(*ax[0].get_legend_handles_labels(),loc='upper center',ncol=3,fontsize=6.5,bbox_to_anchor=(.55,.995))
fig.tight_layout(rect=(0,0,1,.95));stamp(fig,'MOCKUP: Table III as typed in paper.tex plus a numpy replica on the saved windows; not harness output');fig.savefig(O+'/m6_design_dots.png',dpi=220);plt.close(fig)
# header
B0='/home/ensismoebius/Repos/doutorado/software/nn/results/meeting01_t1/fsdd_c1_A/meeting01_t1_fsdd_c1_A_fsdd_fold0_target_%s_windows'
a_=np.load(B0%'test'+'.npy');me=list(csv.DictReader(open(B0%'test'+'_meta.csv')));idx=np.array([int(x['source_window_index']) for x in me])
fig,ax=plt.subplots(1,2,figsize=(7.2,2.3),gridspec_kw={'width_ratios':[1.5,1]})
a=ax[0]
for w in a_[idx>0][:25]: a.plot(w[:60],color=Gy,lw=.4)
for w in a_[idx==0][:25]: a.plot(w[:60],color=Rd,lw=.5)
a.axvline(21.5,color=INK,lw=.7,ls='--');a.text(23,6.6,'22 "samples" = the 44-byte RIFF header',fontsize=7)
a.plot([],[],color=Rd,label='window 0 of 25 test recordings');a.plot([],[],color=Gy,label='25 later windows');a.legend(fontsize=6.5,loc='upper right',bbox_to_anchor=(1,.85))
a.set_xlabel('sample in window');a.set_ylabel('z-scored value');a.set_title('First 60 samples as the networks receive them',fontsize=8,loc='left')
a=ax[1]
lab=['mean predictor','Transformer control','PCA k=16'];w0=[.861,.113,.095];ot=[1.007,1.000,.696]
y=np.arange(3)[::-1]
a.barh(y+.18,ot,height=.32,color=B,label='real speech windows (92%)');a.barh(y-.18,w0,height=.32,color=Rd,label='window 0 (8%)')
for yy,o,z in zip(y,ot,w0): a.text(o+.015,yy+.18,f'{o:.3f}',va='center',fontsize=6.5);a.text(z+.015,yy-.18,f'{z:.3f}',va='center',fontsize=6.5)
a.set_yticks(y);a.set_yticklabels(lab,fontsize=7);a.set_xlim(0,1.45);a.set_xlabel('test MSE, FSDD fold 0');a.legend(fontsize=6.3,loc='lower right',bbox_to_anchor=(1.03,.28));a.set_title('All the "skill" sits in window 0',fontsize=8,loc='left')
fig.tight_layout();stamp(fig,'EVIDENCE (reviewer): saved harness windows and per-window CSV of T1, seed 42 c1 control; for the review, not the paper');fig.savefig(O+'/m7_wav_header.png',dpi=220);plt.close(fig)
# schematic helper
def box(a,x,y,w,h,t,fc='#f2f2f2',ec=MUT,fs=6.8,tc=INK):
    a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.015',fc=fc,ec=ec,lw=.8));a.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,color=tc)
def arr(a,p,q,c=MUT): a.annotate('',xy=q,xytext=p,arrowprops=dict(arrowstyle='-|>',lw=.8,color=c,shrinkA=0,shrinkB=0))
# corrected Fig 1
fig,a=plt.subplots(figsize=(7.2,1.9));a.set_xlim(0,1);a.set_ylim(0,1);a.axis('off')
Red='#f6d9d6';Rc=Rd
ty=.62;h=.2;xs=[.0,.06,.2,.34,.42,.55,.645,.73,.875];
items=[('$x$','#f2f2f2',MUT,.04),('spike code: direct,\nPoisson, latency',Red,Rc,.125),('transform: dense,\nconv1d, recurrent',Red,Rc,.125),('Lin','#f2f2f2',MUT,.065),('LIF, T steps;\n$R,C,V_{th}$ learned',Red,Rc,.115),('Lin→d','#f2f2f2',MUT,.08),('mean\nover T',Red,Rc,.07),('decoder\nLin-LReLU-Lin','#f2f2f2',MUT,.13),('$\\hat x$','#f2f2f2',MUT,.05)]
for (t,f,e,w),x in zip(items,xs): box(a,x,ty,w,h,t,f,e,fs=6)
for i in range(len(xs)-1): arr(a,(xs[i]+items[i][3],ty+h/2),(xs[i+1],ty+h/2))
by=.12
for i,(x,t) in enumerate(((.0,'$x$'),(.34,'Lin'),(.42,'LReLU, once'),(.55,'Lin→d'),(.73,'decoder\nLin-LReLU-Lin'),(.875,'$\\hat x$'))):
    box(a,x,by,items[{0:0,1:3,2:4,3:5,4:7,5:8}[i]][3],h,t,fs=6)
for p,q in (((.04,by+h/2),(.34,by+h/2)),((.405,by+h/2),(.42,by+h/2)),((.535,by+h/2),(.55,by+h/2)),((.63,by+h/2),(.73,by+h/2)),((.86,by+h/2),(.875,by+h/2))): arr(a,p,q)
a.text(0,.9,'SNN-AE',fontsize=7.5,fontweight='bold');a.text(0,.4,'ANN-AE',fontsize=7.5,fontweight='bold')
a.text(.45,.9,'red: everything that differs (five blocks, not one neuron)',fontsize=7,color=INK)
stamp(fig,'MOCKUP (reviewer): replaces Fig. 1 at the same width');fig.savefig(O+'/m8_matched_corrected.png',dpi=220);plt.close(fig)
# pipeline
fig,a=plt.subplots(figsize=(7.2,2.6));a.set_xlim(0,1);a.set_ylim(0,1);a.axis('off')
for x,y,w,h,t,f in ((.0,.72,.13,.2,'train groups\n(cap 1000)','#d7e6f2'),(.0,.42,.13,.2,'val block\n(cap 150)','#f9e0c6'),(.0,.1,.13,.2,'test block\n(cap 1500)','#f3d2cf')): box(a,x,y,w,h,t,f)
box(a,.18,.62,.2,.28,'NSGA-II: 8×6 = 48 genomes\nminimize val MSE and\nspike-blind MAC count',fs=6.3)
box(a,.43,.62,.13,.28,'front re-scored\n3 seeds;\nlowest val MSE\nwins (E unused)')
box(a,.62,.72,.16,.2,'retrain SNN-AE on\ntrain+val−monitor')
box(a,.62,.44,.16,.2,'copy widths: train\nANN-AE, same fit')
box(a,.62,.14,.16,.2,'PCA, mean on\ncapped train+val\n(no E)')
box(a,.84,.44,.16,.3,'test once:\nper-window error,\nE per window','#f3d2cf')
for p,q in (((.13,.82),(.18,.8)),((.13,.52),(.18,.7)),((.38,.76),(.43,.76)),((.56,.8),(.62,.82)),((.56,.7),(.62,.56)),((.78,.82),(.84,.68)),((.78,.54),(.84,.58)),((.78,.24),(.84,.46)),((.13,.2),(.84,.46))): arr(a,p,q)
a.text(.16,.11,'dev folds (FSDD/EEGMMIDB fold 0): initializer, loss, pool point, T range,\nstopping rule chosen outside this box; pre-validation and T1 use them too',fontsize=6.5,color=INK,va='top')
stamp(fig,'MOCKUP (reviewer): one outer fold and one seed');fig.savefig(O+'/m9_selection_pipeline.png',dpi=220);plt.close(fig)
