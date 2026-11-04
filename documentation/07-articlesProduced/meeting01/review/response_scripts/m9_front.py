import numpy as np
rng=np.random.default_rng(1)
N=48;B=40000
def front(v,c):
    o=np.lexsort((v,c)); best=np.inf; f=[]
    for i in o:
        if v[i]<best: best=v[i]; f.append(i)
    return np.array(f)
for s in (1.0,.5):
    r={'single':[], '1+2':[], '3fresh':[]}; rep={'1+2':[], '3fresh':[]}; L=[]
    for _ in range(B):
        mu=rng.normal(0,1,N); c=rng.normal(0,1,N); v=mu+rng.normal(0,s,N)
        f=front(v,c); L.append(len(f))
        r['single'].append(mu[f[v[f].argmin()]])
        x=mu[f][:,None]+rng.normal(0,s,(len(f),3))
        a=(v[f]+x[:,1:].sum(1))/3; b=x.mean(1)
        i=a.argmin(); j=b.argmin(); r['1+2'].append(mu[f[i]]); r['3fresh'].append(mu[f[j]]); rep['1+2'].append(a[i]-mu[f[i]]); rep['3fresh'].append(b[j]-mu[f[j]])
    print('noise',s,'front size',round(np.mean(L),2),{k:round(float(np.mean(v)),3) for k,v in r.items()},'score-truth',{k:round(float(np.mean(v)),3) for k,v in rep.items()})
