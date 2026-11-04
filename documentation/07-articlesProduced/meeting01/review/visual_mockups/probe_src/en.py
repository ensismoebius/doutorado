ADD,MUL,IA,IM=0.9,3.7,0.1,3.1
PTS=[(8*1024,10.),(32*1024,20.),(1024*1024,100.)]
def sram(b):
    if b<=PTS[0][0]: return PTS[0][1]
    s=1
    while s+1<len(PTS) and b>PTS[s][0]: s+=1
    a,c=PTS[s-1],PTS[s]; return a[1]+(c[1]-a[1])*(b-a[0])/(c[0]-a[0])
class L:
    def __init__(s): s.ops=0.;s.mem=0.
    def acc(s,n,w): s.mem+=n*sram(w*4)
    def nj(s): return (s.ops+s.mem)/1000
def matmul(e,R,ni,no,ow,wr):
    e.ops+=R*ni*no*(ADD+MUL)+IA*R*(ni+no); e.acc(R*ni,ni); e.acc(R*ni*no,ow)
    if wr: e.acc(R*no,no)
def dense(e,R,ni,no,wr):
    w=(ni+1)*no; matmul(e,R,ni,no,w,wr); e.ops+=R*no*ADD; e.acc(R*no,w)
def spk(e,S,R,ni,no,wr):
    ev=S*no; w=(ni+1)*no; e.ops+=(ev+R*no)*ADD+ev*IA; e.acc(S,ni); e.acc(ev+R*no,w); e.acc(ev+R*no,no); e.acc(ev+R*no,no)
    if wr: e.acc(R*no,no)
def lif(e,R,n,S,dense_in):
    e.ops+=R*n*(ADD+MUL)
    if dense_in: e.acc(R*n,n); e.acc(R*n,n)
    e.ops+=(R*n+S)*ADD; e.acc(S,n)
def nl(e,v): e.ops+=v*(ADD+MUL)
def decoder(e,w,M):
    r=w[::-1]
    for i in range(len(r)-1): dense(e,1,r[i],r[i+1],True); nl(e,r[i+1])
    dense(e,1,r[-1],M,True)
def ann(M,w):
    b={}
    e=L(); seq=[M]+w
    dense(e,1,seq[0],seq[1],True); 
    if len(w)>1: nl(e,seq[1])
    b['first layer']=e.nj(); 
    e2=L()
    for i in range(1,len(w)-1): dense(e2,1,seq[i],seq[i+1],True); nl(e2,seq[i+1])
    dense(e2,1,seq[-2],seq[-1],True); b['to latent']=e2.nj()
    e3=L(); decoder(e3,w,M); b['decoder']=e3.nj(); return b
def snn(M,w,T,enc,rho,rin=None,once=True):
    """one hidden LIF layer [h,z] or more: rho spike fraction per neuron-step"""
    b={}
    seq=[M]+w
    e=L()
    if enc=='direct':
        if once: dense(e,1,M,w[0],True); e.acc(T*w[0],w[0])
        else: dense(e,T,M,w[0],False)
    else:
        Sin=rin*M*T
        e.acc(M*T,M); e.ops+=M*T*(IM+IA+ADD); e.acc(Sin,M)
        spk(e,Sin,T,M,w[0],False)
    b['first layer']=e.nj()
    e=L(); S=rho*w[0]*T; lif(e,T,w[0],S,enc=='direct'); 
    for i in range(1,len(w)-1):
        spk(e,S,T,seq[i],seq[i+1],False); S2=rho*seq[i+1]*T; lif(e,T,seq[i+1],S2,False); S=S2
    b['LIF + potentials']=e.nj()
    e=L(); spk(e,S,T,seq[-2],seq[-1],True); b['to latent']=e.nj()
    e=L(); e.ops+=T*w[-1]*ADD+w[-1]*(ADD+MUL); e.acc(T*w[-1],w[-1]); e.acc(w[-1],w[-1]); b['time mean']=e.nj()
    e=L(); decoder(e,w,M); b['decoder']=e.nj(); return b
if __name__=='__main__':
    a=ann(256,[59,16]); print('ANN',sum(a.values()),a)
    for rho in (.1,.2,.3,.4):
        x=snn(256,[59,16],28,'direct',rho); y=snn(256,[59,16],28,'direct',rho,once=False)
        print(rho,'once',round(sum(x.values()),1),'exec',round(sum(y.values()),1),'diff',round(sum(y.values())-sum(x.values()),1))
    print('target once 1304, exec 12321, diff 11017')
