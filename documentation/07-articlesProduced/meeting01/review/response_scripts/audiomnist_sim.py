# Port of stratified_window_cap + assign_speaker_fold on the documented AudioMNIST layout NN/d_NN_t.wav
import collections
files=sorted(f'{s:02d}/{d}_{s:02d}_{t}.wav' for s in range(1,61) for d in range(10) for t in range(50))
sid={s:i for i,s in enumerate(range(1,61))}
def group(rank,n=60,k=6): return rank*k//n
def cap_rr(windows,cap):
    by=collections.OrderedDict()
    for w in windows: by.setdefault(w[0],[]).append(w)
    keys=sorted(by); keep=[]; ptr=0; pos={k:0 for k in keys}; prog=True
    while len(keep)<cap and prog:
        prog=False
        for k in keys:
            if len(keep)>=cap: break
            if pos[k]<len(by[k]): keep.append(by[k][pos[k]]); pos[k]+=1; prog=True
    return keep
for fold in (1,3):
    tr=[];va=[];te=[]
    for rid,f in enumerate(files):
        s=int(f[:2]); g=group(sid[s])
        for w in (0,1):
            x=(rid,w,s,int(f[3]))
            (te if g==fold else va if g==(fold+1)%6 else tr).append(x)
    for n,w,c in (('train',tr,1000),('val',va,150),('test',te,1500)):
        k=cap_rr(w,c); print('fold',fold,n,'pool recs',len({x[0] for x in w}),'pool speakers',len({x[2] for x in w}),'kept recs',len({x[0] for x in k}),'speakers',sorted({x[2] for x in k}),'digits',sorted({x[3] for x in k}),'window0 share',round(sum(x[1]==0 for x in k)/len(k),2))
