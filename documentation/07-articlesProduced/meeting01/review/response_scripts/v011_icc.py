import csv,glob,os,collections
import numpy as np
root='/home/ensismoebius/Repos/doutorado/software/nn/results/meeting01_t1'
snap=os.environ.get('SNAP')
def load(ds,c,a):
    f=glob.glob(f'{root}/{ds}_{c}_{a}/*per_window_errors.csv')[0]
    return [r for r in csv.DictReader(open(f)) if r['split']=='test']
sumr=list(csv.DictReader(open(root.replace('/home/ensismoebius/Repos/doutorado/software/nn',snap)+'/t1_summary.csv')))
print('FSDD: model-level, by window index 0 vs other')
for c in ('c1','c2'):
  for a in 'ABC':
    R=load('fsdd',c,a); d=collections.defaultdict(list)
    for r in R: d[r['run_id'],r['seed']].append(r)
    for k,v in sorted(d.items()):
        w=np.array([int(x['source_window_index'])==0 for x in v]); e=np.array([float(x['mse']) for x in v])
        print(c,a,k,'n',len(v),'all %.4f win0 %.4f other %.4f share0 %.3f'%(e.mean(),e[w].mean(),e[~w].mean(),w.mean()))
# mean predictor by idx
print('ICC: EEG, recording-level paired difference between two T1 trainings, subjects as clusters')
meta={}
for r in csv.DictReader(open(glob.glob(f'{root}/eegmmidb_c1_A/*target_test_windows_meta.csv')[0])): meta[int(r['recording_id'])]=r['speaker_id']
def recm(R):
    d=collections.defaultdict(lambda: collections.defaultdict(list))
    for r in R: d[r['run_id']][int(r['recording_id'])].append(float(r['mse']))
    return {k:{a:np.mean(b) for a,b in v.items()} for k,v in d.items()}
A=recm(load('eegmmidb','c1','A')); B=recm(load('eegmmidb','c1','B'))
ka=sorted(A)[0]; kb=sorted(B)[0]
def icc(x,y):
    rs=sorted(set(x)&set(y)); d=np.array([x[r]-y[r] for r in rs]); g=np.array([meta[r] for r in rs])
    u=sorted(set(g)); n=np.array([(g==s).sum() for s in u]); m=np.array([d[g==s].mean() for s in u])
    msb=(n*(m-d.mean())**2).sum()/(len(u)-1); msw=sum(((d[g==s]-d[g==s].mean())**2).sum() for s in u)/(len(d)-len(u)); n0=n.mean()
    return (msb-msw)/(msb+(n0-1)*msw),d.mean(),m.std(ddof=1),len(u),len(d)
print('A vs B',[round(float(v),4) for v in icc(A[ka],B[kb])])
print('A vs B same run ids',{k:round(float(icc(A[k],B[k])[0]),3) for k in A if k in B})
