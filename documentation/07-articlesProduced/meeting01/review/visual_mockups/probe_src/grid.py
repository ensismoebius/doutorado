import os,sys,json,time
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,csv,multiprocessing as mp
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import ae
SC=os.path.dirname(os.path.abspath(__file__))
B='/home/ensismoebius/Repos/doutorado/software/nn/results/meeting01_t1/%s_c1_A/meeting01_t1_%s_c1_A_%s_fold0_target_%s_windows'
def load(ds,s):
    b=B%(ds,ds,ds,s); a=np.load(b+'.npy').astype(np.float32); m=list(csv.DictReader(open(b+'_meta.csv')))
    v=np.array([int(x['valid_length']) for x in m]); return a,(np.arange(256)[None]<v[:,None]).astype(np.float32),np.array([int(x['source_window_index']) for x in m])
D={}
for ds in ('fsdd','eegmmidb'):
    D[ds]={s:load(ds,s) for s in ('train','val','test')}
n=np.load(SC+'/fsdd.npz'); D['fsdd_noheader']={s:(n[s+'X'],n[s+'M'],None) for s in ('train','val','test')}
GEN={'fsdd':(64,16,.6,.83),'fsdd_noheader':(64,16,.6,.83),'eegmmidb':(65,64,.96,.965)}
def job(a):
    ds,kind,T,seed=a; h,z,vth,beta=GEN[ds]; d=D[ds]; t=time.time()
    net,best,cur=ae.train(kind,d['train'][:2],d['val'][:2],seed,h,z,T=T,vth=vth,beta=beta)
    e=net.mse(*d['test'][:2]); out=dict(ds=ds,kind=kind,T=T,seed=seed,test=float(e.mean()),val=best,sec=time.time()-t,bestep=int(np.argmin(cur))+1,vth=float(net.vth[0]),beta=float(net.beta[0]))
    if kind=='lif':
        out['rate']=float(net.enc(d['test'][0]).mean()); r=net.enc(d['test'][0]); out['frac0']=float((r==0).mean()); out['frac1']=float((r==1).mean())
    elif kind in('clamp','clampste'):
        r=net.W[0]; I=d['test'][0]@net.W[0]+net.b[0]; out['frac0']=float((I<=0).mean()); out['frac1']=float((I>=1).mean())
    if d['test'][2] is not None:
        z_=d['test'][2]==0
        if z_.any(): out['test_win0']=float(e[z_].mean()); out['test_other']=float(e[~z_].mean())
    print(json.dumps(out),flush=True); return out
if __name__=='__main__':
    jobs=[]
    for s in (41,42):
        for k,T in (('lrelu',0),('clamp',0),('sigmoid',0),('clampste',0),('lif',4)): jobs.append(('fsdd',k,T,s))
        for k,T in (('lrelu',0),('clamp',0),('sigmoid',0),('clampste',0),('lif',4),('lif',16)): jobs.append(('eegmmidb',k,T,s))
    jobs+= [('fsdd_noheader','lrelu',0,41),('fsdd_noheader','lif',4,41)]
    with mp.Pool(int(sys.argv[1])) as p: r=p.map(job,jobs,chunksize=1)
    json.dump(r,open(SC+'/grid.json','w'),indent=1)
    print('DONE')
