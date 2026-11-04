import sys,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import en_copy as E
MS=[1.0]
_a=E.L.acc
def acc(s,n,w): s.mem+=MS[0]*n*E.sram(w*4)
E.L.acc=acc
M=256;w=[59,16]
tot=lambda b:sum(b.values())
A=lambda:tot(E.ann(M,w)); 
print('ANN',round(A(),2),E.ann(M,w))
# per-event prices at the first layer
ow=(M+1)*w[0]; p=E.sram(ow*4); pn=E.sram(w[0]*4); pi=E.sram(M*4)
print('prices pJ: weight',round(p,2),'pot',round(pn,2),'input',round(pi,2))
print('MAC cost per synapse: ',round(E.ADD+E.MUL+p,2),' synaptic event (add+addr+weight+pot r/w):',round(E.ADD+E.IA+p+2*pn,2))
# break-even: first layer + LIF, poisson vs direct-once/ANN first layer
a1=E.ann(M,w)['first layer']
for T in (2,4,8,16,28):
    d=E.snn(M,w,T,'direct',.4)
    # solve rin*T for first-layer equality with the ANN first layer
    lo,hi=0,1.
    f=lambda r:E.snn(M,w,T,'poisson',.4,rin=r)['first layer']-a1
    if f(0)>0: be=None
    else:
        for _ in range(60):
            mid=(lo+hi)/2
            if f(mid)>0: hi=mid
            else: lo=mid
        be=lo*T
    print('T',T,'direct once first layer',round(d['first layer'],1),'vs ANN',round(a1,1),'poisson first layer at r=.52',round(E.snn(M,w,T,'poisson',.4,rin=.52)['first layer'],1),'latency r=1/T',round(E.snn(M,w,T,'poisson',.4,rin=1/T)['first layer'],1),'break-even spikes/sample/window',None if be is None else round(be,3))
# direct saving at rho
for T,rho in ((2,.1),(4,.4),(4,.05),(28,.4)):
    s=E.snn(M,w,T,'direct',rho); print('direct T',T,'rho',rho,{k:round(v,1) for k,v in s.items()},'total',round(tot(s),1),'ratio',round(tot(s)/A(),3))
print('M5: SNN/ANN ratio under memory scale')
for ms in (1,.75,.5):
    MS[0]=ms; a=A()
    row=[]
    for lab,T,enc,rho,rin in (('direct T4',4,'direct',.4,None),('direct T28',28,'direct',.4,None),('poisson T4',4,'poisson',.4,.52),('poisson T28',28,'poisson',.4,.52),('latency T28',28,'poisson',.4,1/28)):
        row.append((lab,round(tot(E.snn(M,w,T,enc,rho,rin=rin))/a,3)))
    print(ms,'ANN',round(a,1),row)
MS[0]=1
print('M6: once vs as executed')
for T in (4,28):
    o=tot(E.snn(M,w,T,'direct',.4,once=True)); x=tot(E.snn(M,w,T,'direct',.4,once=False)); print(T,round(o,1),round(x,1),'ratio to ANN',round(o/A(),2),round(x/A(),2))
