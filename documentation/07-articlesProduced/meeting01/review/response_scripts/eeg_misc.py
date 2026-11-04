import numpy as np,csv
R='/home/ensismoebius/Repos/doutorado/software/nn/results/meeting01_t1/eegmmidb_c1_A/meeting01_t1_eegmmidb_c1_A_eegmmidb_fold0_target_%s_windows'
d={s:np.load(R%s+'.npy').astype(np.float64) for s in('train','val','test')}
X=np.vstack([d['train'],d['val']]);mu=X.mean(0);U,S,Vt=np.linalg.svd(X-mu,full_matrices=False);V=Vt[:64]
y=(d['test']-mu)@V.T@V+mu;print('EEG PCA64',((y-d['test'])**2).mean(),'mean',((mu-d['test'])**2).mean())
for s in('train','val','test'):
    m=list(csv.DictReader(open(R%s+'_meta.csv')));ids=[int(r['speaker_id']) for r in m]
    print(s,'windows from S088,S092,S100 (ids 87,91,99)',sum(i in(87,91,99) for i in ids),'of',len(ids))
    x=d[s];lo=x.min(1,keepdims=True);hi=x.max(1,keepdims=True);print('  minmax mean p',((x-lo)/(hi-lo)).mean())
