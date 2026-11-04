import numpy as np,csv,glob,wave,os,collections,sys
R=sys.argv[1]; SC=sys.argv[2]
base='/home/ensismoebius/Repos/doutorado/software/nn/results/meeting01_t1/fsdd_c1_A/meeting01_t1_fsdd_c1_A_fsdd_fold0_target_%s_windows'
def load(s):
    a=np.load(base%s+'.npy').astype(np.float32); m=list(csv.DictReader(open(base%s+'_meta.csv')))
    idx=np.array([int(x['source_window_index']) for x in m]); v=np.array([int(x['valid_length']) for x in m])
    mk=(np.arange(256)[None]<v[:,None]).astype(np.float32); rid=np.array([int(x['recording_id']) for x in m]); return a,mk,idx,rid
def pca(fitX,fitM,teX,teM,k):
    mu=(fitX*fitM).sum(0)/fitM.sum(0); Fx=fitX*fitM+(1-fitM)*mu
    U,S,Vt=np.linalg.svd(Fx-mu,full_matrices=False); P=Vt[:k]
    Tx=teX*teM+(1-teM)*mu; r=(Tx-mu)@P.T@P+mu
    e=((r-teX)**2*teM).sum(1)/teM.sum(1); em=((mu-teX)**2*teM).sum(1)/teM.sum(1); return e,em
tr=load('train');va=load('val');te=load('test')
e,em=pca(np.vstack([tr[0],va[0]]),np.vstack([tr[1],va[1]]),te[0],te[1],16)
z=te[2]==0
print('SAVED windows: PCA',e.mean(),'mean',em.mean(),' | window-0 share',z.mean(),': PCA win0',e[z].mean(),'others',e[~z].mean(),'; mean-pred win0',em[z].mean(),'others',em[~z].mean())
d=np.load(SC+'/probe/fsdd.npz')
e2,em2=pca(np.vstack([d['trainX'],d['valX']]),np.vstack([d['trainM'],d['valM']]),d['testX'],d['testM'],16)
print('HEADER-FREE same (rid,index): PCA',e2.mean(),'mean',em2.mean())
# T1 per window
for arm in 'ABC':
  for cfg in ('c1','c2'):
    n=f'fsdd_{cfg}_{arm}'; rows=list(csv.DictReader(open(f'{R}/{n}/meeting01_t1_{n}_fsdd_fold0_per_window_errors.csv')))
    rows=[r for r in rows if r['split']=='test']
    by=collections.defaultdict(list)
    for r in rows: by[r['seed']].append(r)
    for sd,rr in sorted(by.items()):
        q=np.array([float(r['mse']) for r in rr]); w0=np.array([int(r['source_window_index'])==0 for r in rr])
        # mean-pred paired
        key={(int(a),int(b)):em[i] for i,(a,b) in enumerate(zip(te[3],te[2]))}
        mp=np.array([key[(int(r['recording_id']),int(r['source_window_index']))] for r in rr])
        good=q<0.9*mp
        print(n,sd,'test',round(q.mean(),4),'win0 mse',round(q[w0].mean(),3),'others',round(q[~w0].mean(),4),'mean-pred others',round(mp[~w0].mean(),4),'share<0.9mp',round(good.mean(),3),'of which win0',round((good&w0).sum()/max(good.sum(),1),3))
