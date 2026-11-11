import math, itertools, random, collections
# 1. surrogate normalisation
from math import pi
try:
    from scipy import integrate, stats
    a=2.0; f=lambda v:a/(2*(1+(pi/2*a*v)**2))
    print('surrogate integral',integrate.quad(f,-float('inf'),float('inf'))[0],'peak',f(0))
except ImportError: print('no scipy')
# 2. LIF constant current
def lif(I,beta,vth,T):
    v=0;n=0;tr=[]
    for t in range(T):
        v=beta*v+I
        s=v>vth
        tr.append(int(s))
        if s: v=0;n+=1
    return n,tr
for beta,vth in [(0.9,1.0),(0.83,0.6)]:
    rb=vth*(1-beta)
    print('beta',beta,'vth',vth,'rheobase',rb)
    print(' I=rheobase, T=2000:',lif(rb,beta,vth,2000)[0],' I=rheobase*1.0001:',lif(rb*1.0001,beta,vth,2000)[0])
    print(' I=vth (exact), first 8:',lif(vth,beta,vth,8)[1],' I=vth+eps:',lif(vth+1e-6,beta,vth,8)[1],' I=vth-eps',lif(vth-1e-6,beta,vth,8)[1])
    for T in (4,16,32):
        levels=sorted({lif(rb*0.5+k*(vth*1.2-rb*.5)/20000,beta,vth,T)[0]/T for k in range(20001)})
        print(f' T={T}: distinct time-mean output levels under constant I:',len(levels),[round(x,3) for x in levels][:12])
# clamp vs LIF staircase mid-points, inverse of period formula
# 3. Wilcoxon bound
if True:
    print('Wilcoxon n=5 min p',stats.wilcoxon([1,2,3,4,5],alternative='two-sided',method='exact').pvalue, 2/2**5, ' after Holm x3',min(1,3*2/2**5))
# 4. repair_widths re-implementation: reachable depths
def repair(w,latent,maxl=4,minw=4,maxw=128):
    w=list(w)
    if not w: w=[maxw]
    hmin=max(minw,latent+1)
    w=[min(max(x,hmin),maxw) for x in w]
    w.sort(reverse=True)
    fixed=[];c=maxw+1
    for x in w:
        v=min(x,c-1)
        if v<hmin: break
        fixed.append(v);c=v
    cap=max(0,maxl-1)
    fixed=fixed[:cap]
    fixed.append(latent); return fixed
for lat in (16,64):
    cnt=collections.Counter()
    for _ in range(200000):
        d=random.randint(1,4)
        w=[random.randint(4,128) for _ in range(d)]
        cnt[len(repair(w,lat))]+=1
    print('latent',lat,'P(len(encoder_widths)):',{k:round(v/200000,3) for k,v in sorted(cnt.items())})
    # mutation that deletes down to the single latent
    print(' repair([d])=',repair([lat],lat),' repair([120])=',repair([120],lat))
