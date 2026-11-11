import numpy as np
from scipy import stats

ADD, MUL, IADD = 0.9, 3.7, 0.1
PTS = [(8*1024, 10.0), (32*1024, 20.0), (1024*1024, 100.0)]

def sram(b):
    if b <= PTS[0][0]: return PTS[0][1]
    for (b0,p0),(b1,p1) in zip(PTS,PTS[1:]):
        if b <= b1:
            return p0 + (p1-p0)*(b-b0)/(b1-b0)
    return PTS[-1][1]

class Ledger:
    def __init__(self): self.mac=self.add=self.mul=self.ia=0.0; self.acc=0.0; self.mem=0.0
    def access(self,n,words): self.acc += n; self.mem += n*sram(words*4)
    def ops(self): return self.mac*(ADD+MUL)+self.add*ADD+self.mul*MUL+self.ia*IADD
    def total(self, mem_scale=1.0): return (self.ops()+mem_scale*self.mem)/1000

def dense(e,R,ni,no,bias=True):
    e.mac += R*ni*no; e.ia += R*(ni+no)
    e.access(R*ni, ni)
    w = (ni+1)*no if bias else ni*no
    e.access(R*ni*no, w)
    e.access(R*no, no)
    if bias:
        e.add += R*no; e.access(R*no, w)

def nl(e,v): e.add += v; e.mul += v

def ann(M,w):
    e=Ledger(); seq=[M]+w
    for i in range(len(w)-1):
        dense(e,1,seq[i],seq[i+1]); nl(e,seq[i+1])
    dense(e,1,seq[-2],seq[-1])
    r=w[::-1]
    for i in range(len(r)-1):
        dense(e,1,r[i],r[i+1]); nl(e,r[i+1])
    dense(e,1,r[-1],M)
    return e

e=ann(256,[59,16])
print("ENERGY ANN [59,16]")
print("  E (64-bit price) =", round(e.total(),4), "nJ")
print("  E (32-bit = half memory) =", round(e.total(0.5),2), "nJ")
print("  memory share =", round(e.mem/(e.ops()+e.mem),3))
print("  arithmetic =", round(e.ops()/1000,2), "nJ")
print("  ratio 64/32 =", round(e.total()/e.total(0.5),3))

print("\nSRAM PRICE CHECK (paper says 32-bit word)")
for b in (256, 1024, 8*1024, 32*1024, 1024*1024):
    print(f"  {b:>8} B: {sram(b):8.3f} pJ/access ; per 32-bit word if 64-bit price halved: {sram(b)/2:8.3f}")

print("\nLIF CONSTANT-CURRENT MAP (V0=0, hard reset to 0)")
def period(I, beta, Vth, maxn=100000):
    V=0.0
    for n in range(1,maxn+1):
        V = beta*V + I
        if V > Vth:
            return n
    return None
for beta,Vth in ((0.9,1.0),(0.99,1.0)):
    print(f"  beta={beta}, Vth={Vth}, Vth*(1-beta)={Vth*(1-beta):.4f}")
    for I in (0.5*Vth*(1-beta), 0.5, 0.9, 0.99, Vth*(1-beta)*1.0001, Vth*0.999, Vth, Vth*1.001, 1.5*Vth):
        k=period(I,beta,Vth)
        print(f"    I={I:.6f}: period={k}, rate={1.0/k if k else 0:.6f}")

print("\nEQ (2) RESET TERM: v_post = v_pre + s*(Vreset - v_pre), s=Theta(v_pre-Vth)")
Vth, Vreset, surr = 1.0, 0.0, 0.5
for v in (1.2, 0.9):
    s = 1.0 if v > Vth else 0.0
    paper = surr                              # Eq (2): dS/dv only
    code  = surr + (1.0-s)                    # grad_v_pre = grad_out*surr + grad_from_next*beta*(1-s), beta=1, grad_next=1
    exact = surr + (1.0-s) - (v-Vreset)*surr  # includes ds/dv term
    print(f"  v={v}: s={s}, paper={paper}, code={code}, exact={exact}, missing={exact-code:+.3f}")

print("\nWILCOXON EXACT MIN TWO-SIDED p")
for n in (5, 10, 14, 20, 50, 60, 109):
    print(f"  n={n}: 2/2^n={2/2**n:.6g}")
print("  scipy wilcoxon all-positive n=5:", stats.wilcoxon(np.ones(5), alternative='two-sided').pvalue)
print("  scipy wilcoxon all-positive n=6:", stats.wilcoxon(np.ones(6), alternative='two-sided').pvalue)

print("\nT1 COUNTS / EXACT INTERVALS")
for k,n,label in ((1,4,'FSDD warm-up final'),(0,4,'FSDD control final'),(1,4,'FSDD pre-norm final'),
                  (1,8,'FSDD warm-up all 8'),(0,8,'FSDD control all 8'),(1,8,'FSDD pre-norm all 8'),
                  (0,4,'EEG warm-up final'),(1,4,'EEG control final'),(1,4,'EEG pre-norm final')):
    lo,hi = stats.beta.ppf([0.025,0.975], k+0.5, n-k+0.5)
    print(f"  {label:28s} {k}/{n}: Clopper-Pearson [{lo:.3f},{hi:.3f}]")
print("  Fisher 1/4 vs 0/4:", stats.fisher_exact([[1,3],[0,4]]).pvalue)
print("  Fisher 1/8 vs 0/8:", stats.fisher_exact([[1,7],[0,8]]).pvalue)

print("\nWINNER'S CURSE: 48 candidates, true=0, val=test+noise, select min val")
rng=np.random.default_rng(0)
B=200000
true=np.zeros(B); val=true+rng.normal(0,1,B); test=true+rng.normal(0,1,B)
i=np.argmin(val)
print("  E[val|min]=", round(val[i],3), " E[test|min]=", round(test[i],3), " (naive test mean 0)")
# re-score with 1 search seed + 2 fresh: mean of 3, still includes the selected seed
vals=np.stack([val, true+rng.normal(0,1,B), true+rng.normal(0,1,B)],1)
res=vals.mean(1); j=np.argmin(res)
print("  E[re-scored|argmin]=", round(res[j],3), " E[test|argmin]=", round(test[j],3),
      " (fresh-only would be unbiased for true test)")
# fresh-only 3 seeds
fresh=np.stack([true+rng.normal(0,1,B) for _ in range(3)],1).mean(1); k=np.argmin(fresh)
print("  E[fresh3|argmin]=", round(fresh[k],3), " E[test|argmin]=", round(test[k],3))

print("\nWINNER'S CURSE (corrected): mu_i~N(0,1) true quality, val=mu+noise, test=mu+noise2")
rng=np.random.default_rng(1)
N=48; T=200000
mu=rng.normal(0,1,(T,N))
val=mu+rng.normal(0,1,(T,N))
test=mu+rng.normal(0,1,(T,N))
i=np.argmin(val,axis=1)
print("  select min val: E[mu|sel]=", round(mu[np.arange(T),i].mean(),3),
      " E[val|sel]=", round(val[np.arange(T),i].mean(),3),
      " E[test|sel]=", round(test[np.arange(T),i].mean(),3))
# re-score: mean of search-seed val + 2 fresh (same mu, new noise), then argmin
fresh=rng.normal(0,1,(T,N,2))
res=(val+fresh.sum(2))/3
j=np.argmin(res,axis=1)
print("  re-score incl. search seed: E[mu|sel]=", round(mu[np.arange(T),j].mean(),3),
      " E[test|sel]=", round(test[np.arange(T),j].mean(),3))
# fresh-only 3 seeds
fresh3=rng.normal(0,1,(T,N,3)).mean(2)
k=np.argmin(fresh3,axis=1)
print("  fresh-only 3 seeds:        E[mu|sel]=", round(mu[np.arange(T),k].mean(),3),
      " E[test|sel]=", round(test[np.arange(T),k].mean(),3))

print("\nT1 MEAN TEST MSE (final fits, 4 per arm per dataset)")
import csv
rows=list(csv.DictReader(open('software/nn/results/meeting01_t1/t1_summary.csv')))
for ds in ('fsdd','eegmmidb'):
    for arm in ('A','B','C'):
        v=[float(r['test_mse']) for r in rows if r['dataset']==ds and r['arm']==arm and r['training']=='final']
        print(f"  {ds} {arm}: n={len(v)} mean={np.mean(v):.4f} sd={np.std(v,ddof=1):.4f} values={v}")
    a=[float(r['test_mse']) for r in rows if r['dataset']==ds and r['arm']=='A' and r['training']=='final']
    b=[float(r['test_mse']) for r in rows if r['dataset']==ds and r['arm']=='B' and r['training']=='final']
    print(f"  {ds} t-test A vs B: t={stats.ttest_ind(a,b,equal_var=False).statistic:.3f} p={stats.ttest_ind(a,b,equal_var=False).pvalue:.3f}")
