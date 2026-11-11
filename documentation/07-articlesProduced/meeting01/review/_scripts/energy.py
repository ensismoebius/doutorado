# independent re-implementation of Table tab:cost / Meeting01Energy.cpp (analog path)
ADD,MUL,IA=0.9,3.7,0.1
PTS=[(8*1024,10.),(32*1024,20.),(1024*1024,100.)]
def sram(b):
    if b<=PTS[0][0]: return PTS[0][1]
    s=1
    while s+1<len(PTS) and b>PTS[s][0]: s+=1
    a,c=PTS[s-1],PTS[s]
    return a[1]+(c[1]-a[1])*(b-a[0])/(c[0]-a[0])
class L:
    def __init__(s): s.mac=s.add=s.mul=s.ia=0;s.mem=0;s.acc=0
    def access(s,n,w):
        s.acc+=n; s.mem+=n*sram(w*4)
    def ops(s): return s.mac*(ADD+MUL)+s.add*ADD+s.mul*MUL+s.ia*IA
    def tot(s): return (s.ops()+s.mem)/1000
    def at(s,b): return (s.ops()+s.acc*sram(b))/1000
def dense(e,R,ni,no,wr=True,bias=True):
    e.mac+=R*ni*no; e.ia+=R*(ni+no); e.access(R*ni,ni)
    w=(ni+1)*no if bias else ni*no
    e.access(R*ni*no,w)
    if wr: e.access(R*no,no)
    if bias:
        e.add+=R*no; e.access(R*no,w)
def nl(e,v): e.add+=v; e.mul+=v
def ann(M,w):
    e=L(); seq=[M]+w
    # encoder
    for i in range(len(w)-1):
        dense(e,1,seq[i],seq[i+1],True); nl(e,seq[i+1])
    dense(e,1,seq[-2],seq[-1],True)
    # decoder mirror
    r=w[::-1]
    for i in range(len(r)-1):
        dense(e,1,r[i],r[i+1],True); nl(e,r[i+1])
    dense(e,1,r[-1],M,True)
    return e
e=ann(256,[59,16]); print('ANN [59,16] E=',round(e.tot(),4),'nJ  (notes: 854.4853); ops+mem share',round(e.mem/1000/e.tot(),3),' 8K',round(e.at(8*1024),1),'32K',round(e.at(32*1024),1),'1M',round(e.at(1024*1024),1),'MAC arith',e.mac*4.6/1000)
# PCA priced by same model: encoder Linear M->k no bias, decoder k->M no bias; plus mean sub/add M adds; output buffers
for M,k,lab in ((256,16,'PCA k=16'),(256,64,'PCA k=64')):
    p=L(); p.add+=M; p.access(M,M)   # subtract mean
    dense(p,1,M,k,True,bias=False); dense(p,1,k,M,True,bias=False); p.add+=M; p.access(M,M)
    print(lab,'E=',round(p.tot(),1),'nJ ; arithmetic-only',round((p.mac*4.6+p.add*.9)/1000,1))
e2=ann(256,[65,64]); print('ANN [65,64]',round(e2.tot(),1))
# sensitivity of memory 64->32 bit: halve all memory price
print('ANN [59,16] with 32-bit access at half the 64-bit price',round((e.ops()+e.mem/2)/1000,1))
