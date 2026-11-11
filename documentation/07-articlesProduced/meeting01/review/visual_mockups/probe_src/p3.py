import os,sys,json
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np,multiprocessing as mp
SC=sys.argv[1]; sys.path.insert(0,SC+'/probe'); import ae
n=dict(np.load(SC+'/probe/fsdd.npz')); tr=(n['trainX'],n['trainM']);va=(n['valX'],n['valM']);te=(n['testX'],n['testM'])
def job(a):
    kind,T,seed=a
    net,b,c=ae.train(kind,tr,va,seed,64,16,T=T)
    e=net.mse(*te); r=net.enc(te[0])
    o=dict(kind=kind,T=T,seed=seed,test=float(e.mean()),bestep=int(np.argmin(c))+1,lo01=float((r<=.01).mean()),hi01=float((r>=.99).mean()),lo10=float((r<=.1).mean()),hi10=float((r>=.9).mean()),hi_gt1=float((r>1).mean()))
    print(json.dumps(o),flush=True);return o
J=[(k,0,s) for k in('lrelu','clamp','sigmoid') for s in(41,42,43,44)]+[('lif',4,s) for s in(41,42,43,44)]+[('lif',16,41),('lif',16,42)]
if __name__=='__main__':
    with mp.Pool(11) as p: r=p.map(job,J,chunksize=1)
    json.dump(r,open(SC+'/r3/p3.json','w'),indent=1);print('DONE')
