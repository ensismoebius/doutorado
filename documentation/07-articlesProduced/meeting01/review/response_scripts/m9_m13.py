import numpy as np
from scipy import stats
rng=np.random.default_rng(0)
N=48;B=200000
for sd_noise in (1.0,0.5,0.25):
    mu=rng.normal(0,1,(B,N)); e0=rng.normal(0,sd_noise,(B,N)); val=mu+e0
    idx=np.arange(B)
    i=val.argmin(1)
    # A: best of search single seed
    a=mu[idx,i].mean()
    # B: code on all candidates: 1 search + 2 fresh
    f=mu[...,None]+rng.normal(0,sd_noise,(B,N,2)); r=(val+f.sum(2))/3; j=r.argmin(1); b=mu[idx,j].mean(); br=r[idx,j].mean()
    # C: 3 fresh on all
    f3=(mu[...,None]+rng.normal(0,sd_noise,(B,N,3))).mean(2); k=f3.argmin(1); c=mu[idx,k].mean(); cr=f3[idx,k].mean()
    # reviewer's bug: fresh without mu
    bug=(rng.normal(0,sd_noise,(B,N,3))).mean(2).argmin(1)
    print(f'noise/sd(mu)={sd_noise}: E[mu|sel] single {a:.3f}; 1+2 {b:.3f}; 3fresh {c:.3f}; reported score minus truth: single {val[idx,i].mean()-a:.3f}, 1+2 {br-b:.3f}, 3fresh {cr-c:.3f}; buggy fresh {mu[idx,bug].mean():.3f}')
# reviewer exact formula
mu=rng.normal(0,1,(B,N)); val=mu+rng.normal(0,1,(B,N)); res=(val+rng.normal(0,1,(B,N,2)).sum(2))/3
print('reviewer formula 1+2 (fresh noise has no mu)',round(mu[np.arange(B),res.argmin(1)].mean(),3))
# M13: false positive of recording-level Wilcoxon on 2500 recs /5 speakers at delta=0
def sim(icc,S,R,reps=2000):
    fp=0;fpg=0
    for _ in range(reps):
        u=rng.normal(0,np.sqrt(icc),S); d=(u[:,None]+rng.normal(0,np.sqrt(1-icc),(S,R)))
        fp+=stats.wilcoxon(d.ravel()).pvalue<.05
        fpg+=abs(stats.ttest_1samp(d.mean(1),0).pvalue)<.05
    return fp/reps,fpg/reps
for icc in (0,0.01,0.05,0.2):
    print('ICC',icc,'recording Wilcoxon FP, speaker t FP',sim(icc,5,500), 'design effect',1+499*icc)
