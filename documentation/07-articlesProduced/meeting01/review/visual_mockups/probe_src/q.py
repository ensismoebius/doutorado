import sys,numpy as np,glob,struct,os
SC=sys.argv[1]; sys.path.insert(0,SC+'/probe'); import en as E
tot=lambda b:sum(b.values())
print('--- Q5 bound: direct, ratio to ANN')
for w in ([59,16],[128,16],[20,16],[65,64],[128,64],[128,64,32,16]) :
    if len(w)==2 or True:
        a=tot(E.ann(256,w)) if len(w)==2 else None
    if len(w)==2:
        print(w,'ANN',round(a,1),'to-latent',round(E.ann(256,w)['to latent'],1),{(T,r):round(tot(E.snn(256,w,T,'direct',r))/a,3) for T,r in((2,.1),(2,.01),(4,.1),(4,.4),(16,.4))})
print('--- event vs MAC by layer size')
for ni,no in ((256,59),(59,16),(65,64),(128,64)):
    ow=(ni+1)*no; p=E.sram(ow*4); q=E.sram(no*4); print((ni,no),'MAC+wt',round(4.6+p,1),'event',round(1+p+2*q,1),'ratio',round((1+p+2*q)/(4.6+p),2))
print('--- p Poisson and recurrent theta/M')
n=np.load(SC+'/probe/fsdd.npz')
B='/home/ensismoebius/Repos/doutorado/software/nn/results/meeting01_t1/eegmmidb_c1_A/meeting01_t1_eegmmidb_c1_A_eegmmidb_fold0_target_%s_windows.npy'
ds={'fsdd_clean':(n['testX'],n['testM']),'eeg':(np.load(B%'test'),None)}
for k,(X,M) in ds.items():
    lo=X.min(1,keepdims=True);hi=X.max(1,keepdims=True);p=(X-lo)/(hi-lo)
    print(k,'poisson p',round(float((p*M).sum()/M.sum()) if M is not None else float(p.mean()),3))
    def rec(x,T,vth,b):
        v=np.zeros_like(x);s=np.zeros_like(x);c=np.zeros_like(x)
        for t in range(T):
            v=b*v+x-s*vth; s=(v>=vth)*1.; c+=s
        return c.sum(1).mean()/256
    for T,vth,b in ((4,1.28,.91),(4,.5,.99),(2,1.5,.8),(8,1.28,.91),(16,1.0,.9),(28,1.0,.9),(4,1.5,.8)):
        print('  direct+recurrent T',T,'vth',vth,'beta',b,'theta/M',round(float(rec(X[:600],T,vth,b)),3))
    # poisson+recurrent
    r=np.random.default_rng(0);
    def recp(x,T,vth,b):
        v=np.zeros_like(x);s=np.zeros_like(x);c=np.zeros_like(x)
        for t in range(T):
            e=(r.random(x.shape)<x)*1.; v=b*v+e-s*vth; s=(v>=vth)*1.; c+=s
        return c.sum(1).mean()/256
    print('  poisson+recurrent T4 vth1.28 b.91',round(float(recp(p[:600],4,1.28,.91)),3),'T8',round(float(recp(p[:600],8,1.28,.91)),3))
print('--- E direct+recurrent with theta/M')
def rec_total(w,T,th,rho):
    e=E.L(); M=256;S=th*M
    e.acc(M*T,M); E.lif(e,T,M,S,True); t=e.nj()
    e2=E.L();E.spk(e2,S,T,M,w[0],False);f=e2.nj()
    b=E.snn(M,w,T,'direct',rho); b['first layer']=f+t
    b2=dict(b)
    e3=E.L();E.lif(e3,T,w[0],rho*w[0]*T,False);b2['LIF + potentials']=e3.nj(); return tot(b2),t,f
for th,T in ((.05,4),(.1,4),(.2,4),(.3,4),(.6,4),(.1,16)):
    x,t,f=rec_total([59,16],T,th,.2); print('theta/M',th,'T',T,'total',round(x,1),'ratio',round(x/tot(E.ann(256,[59,16])),3),'transform',round(t,1),'spike first layer',round(f,1))
print('--- PCA window draw spread, clean')
DB='/home/ensismoebius/Documentos/academico/UNESP/doutorado/databases/fsdDataset/recordings'
fs=sorted(glob.glob(DB+'/*.wav'))
sig=[]
for f in fs:
    b=open(f,'rb').read();m=struct.unpack('<I',b[40:44])[0]//2;sig.append(np.frombuffer(b[44:44+2*m],'<i2').astype(np.float64))
def wz(s,i):
    seg=s[i*256:(i+1)*256];x=np.zeros(256);x[:len(seg)]=(seg-seg.mean())/max(seg.std(),1e-8);return x
trp=[i for i,f in enumerate(fs) if any(('_%s_'%k) in f for k in('lucas','nicolas','theo','yweweler'))]
vap=[i for i,f in enumerate(fs) if '_jackson_' in f]
te=n['testX'].astype(np.float64);tm=n['testM']
res=[];R=[]
for d in range(24):
    r=np.random.default_rng(1000+d)
    pick=[ (i,int(r.integers(int(np.ceil(len(sig[i])/256))))) for i in r.permutation(trp)[:1000]]+[(i,int(r.integers(int(np.ceil(len(sig[i])/256))))) for i in r.permutation(vap)[:150]]
    X=np.array([wz(sig[i],k) for i,k in pick]);mu=X.mean(0);U,s,Vt=np.linalg.svd(X-mu,full_matrices=False);V=Vt[:16]
    y=(te-mu)@V.T@V+mu;res.append(((y-te)**2*tm).sum(1).sum()/tm.sum())
print('24 draws PCA16 mean',round(np.mean(res),4),'sd',round(np.std(res,ddof=1),4),'min',round(min(res),3),'max',round(max(res),3))
