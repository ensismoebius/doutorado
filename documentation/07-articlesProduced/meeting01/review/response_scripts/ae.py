import numpy as np, math, time
class Adam:
    def __init__(s,ps,lr):
        s.ps=ps; s.lr=lr; s.m=[np.zeros_like(p) for p in ps]; s.v=[np.zeros_like(p) for p in ps]; s.t=0
    def step(s,gs):
        s.t+=1; c=1-0.9**s.t; d=1-0.999**s.t
        for p,g,m,v in zip(s.ps,gs,s.m,s.v):
            m*=0.9; m+=0.1*g; v*=0.999; v+=0.001*g*g
            p-=s.lr*(m/c)/(np.sqrt(v/d)+1e-8)
def lrelu(x): return np.where(x>0,x,0.01*x)
def dlrelu(x): return np.where(x>0,1.0,0.01)
def sig(x): return 1/(1+np.exp(-np.clip(x,-30,30)))
A={'lrelu':(lrelu,dlrelu),
   'sigmoid':(sig,lambda x:(lambda s:s*(1-s))(sig(x))),
   'clamp':(lambda x:np.clip(x,0,1),lambda x:((x>0)&(x<1))*1.0),
   'clampste':(lambda x:np.clip(x,0,1),lambda x:np.ones_like(x))}
def lif(I,T,beta,vth,alpha=2.0):
    """I (B,h) const. Returns rate, and closure for backward."""
    V=np.zeros_like(I); Vs=[];Ss=[];Vp=[]
    for t in range(T):
        Vp.append(V); V=beta*V+I; s=(V>vth)*1.0; Vs.append(V); Ss.append(s); V=V*(1-s)
    return np.mean(Ss,axis=0),Vs,Ss,Vp
def surr(V,vth,a=2.0): return a/(2*(1+(math.pi/2*a*(V-vth))**2))
def init(i,o,rng,kind):
    lim=math.sqrt(6/i) if kind=='kaiming' else math.sqrt(6/(i+o))
    return rng.uniform(-lim,lim,(i,o)).astype(np.float32)   # x@W
class Net:
    def __init__(s,kind,h,z,M=256,seed=0,initk='xavier',T=4,vth=.6,beta=.83):
        r=np.random.default_rng(seed); s.kind=kind;s.T=T
        s.W=[init(M,h,r,initk),init(h,z,r,initk),init(z,h,r,initk),init(h,M,r,initk)]
        s.b=[np.zeros(h,np.float32),np.zeros(z,np.float32),np.zeros(h,np.float32),np.zeros(M,np.float32)]
        s.vth=np.array([vth],np.float32); s.beta=np.array([beta],np.float32)
        s.o=Adam(s.W+s.b,1e-3); s.obio=Adam([s.vth,s.beta],1e-4)
    def enc(s,x):
        I=x@s.W[0]+s.b[0]
        if s.kind=='lif': return lif(I,s.T,s.beta[0],s.vth[0])[0]
        return A[s.kind][0](I)
    def fwd(s,x):
        a=s.enc(x); z=a@s.W[1]+s.b[1]; return lrelu(z@s.W[2]+s.b[2])@s.W[3]+s.b[3]
    def step(s,x,m):
        W,b=s.W,s.b
        I=x@W[0]+b[0]
        if s.kind=='lif':
            a,Vs,Ss,Vp=lif(I,s.T,s.beta[0],s.vth[0])
        else: a=A[s.kind][0](I)
        z=a@W[1]+b[1]; h3=z@W[2]+b[2]; a3=lrelu(h3); y=a3@W[3]+b[3]
        e=(y-x)*m; n=m.sum(); dy=2*e/n
        g3=dy@W[3].T; ga3=g3*dlrelu(h3); gz=ga3@W[2].T; ga=gz@W[1].T
        if s.kind=='lif':
            gv=np.zeros_like(I); dI=np.zeros_like(I); dth=0.0; dbeta=0.0
            for t in range(s.T-1,-1,-1):
                sg=surr(Vs[t],s.vth[0]); go=ga/s.T
                gv=go*sg+gv
                dI+=gv; dth-=float((go*sg).sum()); dbeta+=float((gv*Vp[t]).sum())
                if t>0: gv=gv*s.beta[0]*(1-Ss[t-1])
            s.obio.step([np.array([dth],np.float32),np.array([dbeta],np.float32)])
            s.vth[0]=max(s.vth[0],1e-6); s.beta[0]=min(max(s.beta[0],1e-3),0.9999)
            gI=dI
        else: gI=ga*A[s.kind][1](I)
        G=[np.outer(x,gI),np.outer(a,gz),np.outer(z,ga3),np.outer(a3,dy)]
        B=[gI,gz,g3 if False else ga3,dy]
        s.o.step(G+B)
    def mse(s,X,Mk):
        y=s.fwd(X); e=((y-X)**2*Mk).sum(1)/Mk.sum(1); return e
    def state(s): return [p.copy() for p in s.W+s.b]+[s.vth.copy(),s.beta.copy()]
    def load(s,st):
        for p,q in zip(s.W+s.b+[s.vth,s.beta],st): p[...]=q
def train(kind,tr,va,seed,h,z,epochs=30,T=4,vth=.6,beta=.83,initk='xavier',keep_curve=False):
    n=Net(kind,h,z,seed=seed,T=T,vth=vth,beta=beta,initk=initk); r=np.random.default_rng(seed)
    best=9;st=None;cur=[]
    X,M=tr
    for ep in range(epochs):
        for i in r.permutation(len(X)): n.step(X[i],M[i])
        v=float(n.mse(*va).mean()); cur.append(v)
        if v<best: best=v;st=n.state()
    n.load(st); return n,best,cur
