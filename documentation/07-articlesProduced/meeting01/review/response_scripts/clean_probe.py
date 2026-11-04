import os,sys,json,csv,glob,struct,time
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np, multiprocessing as mp
SC=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,SC); import ae
DB='/home/ensismoebius/Documentos/academico/UNESP/doutorado/databases/fsdDataset/recordings'
files=sorted(glob.glob(DB+'/*.wav'))
R='/home/ensismoebius/Repos/doutorado/software/nn/results/meeting01_t1/%s_c1_A/meeting01_t1_%s_c1_A_%s_fold0_target_%s_windows'
def clean(f):
    b=open(f,'rb').read(); n=struct.unpack('<I',b[40:44])[0]//2; return np.frombuffer(b[44:44+2*n],'<i2').astype(np.float64)
def z(seg):
    x=np.zeros(256,np.float32); m=np.zeros(256,np.float32); x[:len(seg)]=(seg-seg.mean())/max(seg.std(),1e-8); m[:len(seg)]=1; return x,m
def from_pairs(p):
    X=[];M=[]
    for rid,i in p:
        x,m=z(clean(files[rid])[i*256:(i+1)*256]); X.append(x);M.append(m)
    return np.array(X),np.array(M)
meta=lambda s:list(csv.DictReader(open(R%('fsdd','fsdd','fsdd',s)+'_meta.csv')))
sav={s:[(int(r['recording_id']),int(r['source_window_index'])) for r in meta(s)] for s in ('train','val','test')}
def sp(rid): return os.path.basename(files[rid]).split('_')
def balanced(seed):
    r=np.random.default_rng(seed); tr=[];va=[]
    pool={}
    for i in range(len(files)):
        d,s,t=sp(i); pool.setdefault((s,d),[]).append(i)
    def pick(spk,per):
        out=[]
        for s in spk:
            for d in map(str,range(10)):
                for rid in r.permutation(pool[(s,d)])[:per]:
                    nw=int(np.ceil(len(clean(files[rid]))/256)); out.append((int(rid),int(r.integers(nw))))
        return out
    return pick(['lucas','nicolas','theo','yweweler'],25),pick(['jackson'],15)
D={}
te=from_pairs(sav['test']); dig=np.array([int(sp(r)[0]) for r,_ in sav['test']])
D['trunc']=dict(tr=from_pairs(sav['train']),va=from_pairs(sav['val']),te=te)
for k in (1,2):
    a,b=balanced(100+k); D[f'bal{k}']=dict(tr=from_pairs(a),va=from_pairs(b),te=te)
def e(y,x,m): return ((y-x)**2*m).sum(1)/m.sum(1)
for k,d in D.items():
    Xt=np.vstack([d['tr'][0],d['va'][0]]); mu=Xt.mean(0); U,s,Vt=np.linalg.svd(Xt-mu,full_matrices=False)
    out={}
    for q in (16,64):
        V=Vt[:q]; err=e((te[0]-mu)@V.T@V+mu,te[0],te[1]); out[q]=(round(float(err.mean()),4),round(float(err[dig<5].mean()),4),round(float(err[dig>=5].mean()),4))
    print('PCA',k,'k:(all,digits0-4,digits5-9)',out,flush=True)
B=R
def le(s):
    b=B%('eegmmidb','eegmmidb','eegmmidb',s); a=np.load(b+'.npy').astype(np.float32); m=list(csv.DictReader(open(b+'_meta.csv')))
    v=np.array([int(x['valid_length']) for x in m]); return a,(np.arange(256)[None]<v[:,None]).astype(np.float32)
D['eeg']=dict(tr=le('train'),va=le('val'),te=le('test'))
GEN=lambda ds:(65,64,.96,.965) if ds=='eeg' else (64,16,.6,.83)
def job(a):
    ds,kind,T,seed=a; h,zd,vth,beta=GEN(ds); d=D[ds]; t=time.time()
    net,best,cur=ae.train(kind,d['tr'],d['va'],seed,h,zd,T=T,vth=vth,beta=beta)
    err=net.mse(*d['te']); o=dict(ds=ds,kind=kind,T=T,seed=seed,test=float(err.mean()),val=best,sec=round(time.time()-t),bestep=int(np.argmin(cur))+1)
    r=net.enc(d['te'][0])
    o['near_0']=float((r<=.01).mean()); o['near_1']=float((r>=.99).mean())
    if ds!='eeg': o['test_0to4']=float(err[dig<5].mean()); o['test_5to9']=float(err[dig>=5].mean())
    if ds=='eeg': np.save(f"{SC}/eegerr_{kind}_{seed}.npy",err)
    print(json.dumps(o),flush=True); return o
if __name__=='__main__':
    J=[('trunc','lrelu',0,41),('trunc','lrelu',0,42),('trunc','lif',4,41),('trunc','lif',4,42),('trunc','sigmoid',0,41),('trunc','clamp',0,41),
       ('bal1','lrelu',0,41),('bal1','lif',4,41),('bal2','lrelu',0,41),('bal2','lif',4,41),('eeg','lrelu',0,41),('eeg','sigmoid',0,41),('eeg','lif',4,41)]
    with mp.Pool(11) as p: r=p.map(job,J,chunksize=1)
    json.dump(r,open(SC+'/clean_probe.json','w'),indent=1); print('DONE')
