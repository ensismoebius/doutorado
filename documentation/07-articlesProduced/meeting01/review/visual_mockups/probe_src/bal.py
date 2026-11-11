import os,sys,json
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,glob,wave,multiprocessing as mp
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import ae
SC=os.path.dirname(os.path.abspath(__file__))
DB='/home/ensismoebius/Documentos/academico/UNESP/doutorado/databases/fsdDataset/recordings'
files=sorted(glob.glob(DB+'/*.wav'))
rng=np.random.default_rng(7)
def win(f,idx=None):
    w=wave.open(f);s=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)
    n=(len(s)+255)//256; i=int(rng.integers(n)) if idx is None else idx
    seg=s[i*256:(i+1)*256];x=np.zeros(256,np.float32);m=np.zeros(256,np.float32);x[:len(seg)]=(seg-seg.mean())/max(seg.std(),1e-4);m[:len(seg)]=1
    return x,m
tr=[f for f in files if any('_%s_'%s in f for s in('lucas','nicolas','theo','yweweler'))]
va=[f for f in files if '_jackson_' in f]
tr=list(rng.permutation(tr))[:1000]; va=list(rng.permutation(va))[:150]
T=[win(f) for f in tr];V=[win(f) for f in va]
tr_=(np.array([a for a,_ in T]),np.array([b for _,b in T]));va_=(np.array([a for a,_ in V]),np.array([b for _,b in V]))
n=np.load(SC+'/fsdd.npz');te=(n['testX'],n['testM']);td=n['testd']
def pca(X,M,k=16):
    mu=(X*M).sum(0)/M.sum(0);Fx=X*M+(1-M)*mu;U,S,Vt=np.linalg.svd(Fx-mu,full_matrices=False);P=Vt[:k]
    Tx=te[0]*te[1]+(1-te[1])*mu;r=(Tx-mu)@P.T@P+mu;return ((r-te[0])**2*te[1]).sum(1)/te[1].sum(1)
e=pca(np.vstack([tr_[0],va_[0]]),np.vstack([tr_[1],va_[1]]))
e0=pca(np.vstack([n['trainX'],n['valX']]),np.vstack([n['trainM'],n['valM']]))
print('PCA header-free: digits0-4 train',e0.mean(),'[0-4]',e0[td<5].mean(),'[5-9]',e0[td>=5].mean(),'| balanced',e.mean(),'[0-4]',e[td<5].mean(),'[5-9]',e[td>=5].mean(),flush=True)
def job(a):
    kind,T_,bal,seed=a
    t,v=(tr_,va_) if bal else ((n['trainX'],n['trainM']),(n['valX'],n['valM']))
    net,b,c=ae.train(kind,t,v,seed,64,16,T=T_)
    e=net.mse(*te);out=dict(kind=kind,T=T_,bal=bal,seed=seed,test=float(e.mean()),d04=float(e[td<5].mean()),d59=float(e[td>=5].mean()));print(json.dumps(out),flush=True);return out
if __name__=='__main__':
    J=[('lrelu',0,True,41),('lif',4,True,41),('lrelu',0,False,41),('lif',4,False,41),('lrelu',0,True,42),('lif',4,True,42)]
    with mp.Pool(6) as p: r=p.map(job,J,chunksize=1)
    print('DONE')
