import csv,os,glob,struct,collections,sys
import numpy as np
ROOT=sys.argv[1]
DB='/home/ensismoebius/Documentos/academico/UNESP/doutorado/databases/fsdDataset/recordings'
files=sorted(glob.glob(DB+'/*.wav'))
B='/home/ensismoebius/Repos/doutorado/software/nn/results/meeting01_t1/fsdd_c1_A/meeting01_t1_fsdd_c1_A_fsdd_fold0_target_%s_windows'
print('n files',len(files))
# header survey
off=collections.Counter(); bits=collections.Counter(); sr=collections.Counter(); ch=collections.Counter(); ld=[]
for f in files:
    b=open(f,'rb').read()
    i=b.find(b'data'); off[i+8]+=1
    fmt=b.find(b'fmt '); c,s,_,_,bt=struct.unpack('<HIIHH',b[fmt+10:fmt+24]); bits[bt]+=1; sr[s]+=1; ch[c]+=1
    ld.append(struct.unpack('<I',b[i+4:i+8])[0]//2)
print('data offset',off.most_common(5),'bits',bits,'sr',sr,'ch',ch)
ld=np.array(ld); print('samples: median',np.median(ld),'min',ld.min(),'max',ld.max(),'files<=256',int((ld<=256).sum()),'share 22/len median',22/np.median(ld))
def asbuilt(f,start):
    b=open(f,'rb').read(); i=b.find(b'data'); n=struct.unpack('<I',b[i+4:i+8])[0]//2
    return np.frombuffer(b[start(i):start(i)+2*n] if False else b,dtype='<i2',count=None,offset=0) if False else None
def samples(f,hdr):
    b=open(f,'rb').read(); i=b.find(b'data'); n=struct.unpack('<I',b[i+4:i+8])[0]//2
    o=0 if hdr else i+8
    return np.frombuffer(b[o:o+2*n],dtype='<i2').astype(np.float64)
def z(seg):
    x=np.zeros(256); sd=seg.std(); x[:len(seg)]=(seg-seg.mean())/max(sd,1e-8); return x
res={}
for split in ('train','val','test'):
    W=np.load(B%split+'.npy'); M=list(csv.DictReader(open(B%split+'_meta.csv')))
    a=[];c=[];z0=[]
    for k,r in enumerate(M):
        f=files[int(r['recording_id'])]; i=int(r['source_window_index'])
        for h,l in ((True,a),(False,c)):
            s=samples(f,h)[i*256:(i+1)*256]; l.append(z(s))
    a=np.array(a);c=np.array(c);res[split]=(W,a,c,M)
    print(split,W.shape,'max|saved-withheader|',float(np.abs(W-a).max()),'max|saved-clean|',float(np.abs(W-c).max()),'rmse clean',float(np.sqrt(((W-c)**2).mean())))
# PCA/mean with and without header on test
def mk(M,n=256):
    v=np.array([int(r['valid_length']) for r in M]); return (np.arange(n)[None]<v[:,None]).astype(float)
def perwin(y,x,m): return (((y-x)**2)*m).sum(1)/m.sum(1)
for name,idx in (('with header',1),('clean',2)):
    tr=np.vstack([res['train'][idx],res['val'][idx]]); te=res['test'][idx]; m=mk(res['test'][3]); w0=np.array([int(r['source_window_index'])==0 for r in res['test'][3]])
    mu=tr.mean(0); e=perwin(np.tile(mu,(len(te),1)),te,m)
    print(name,'mean ref',round(float(e.mean()),4),'win0',round(float(e[w0].mean()),4),'other',round(float(e[~w0].mean()),4))
    U,s,Vt=np.linalg.svd(tr-mu,full_matrices=False)
    for k in (16,64):
        V=Vt[:k]; y=(te-mu)@V.T@V+mu; e=perwin(y,te,m)
        print('  PCA',k,round(float(e.mean()),4),'win0',round(float(e[w0].mean()),4),'other',round(float(e[~w0].mean()),4))
    print('  share window 0 in test',round(float(w0.mean()),4),'train',round(float(np.mean([int(r['source_window_index'])==0 for r in res['train'][3]])),4))
# how much of window-0 variance is header: var of first 22 samples relative to rest
t=res['test'][1]; w=np.array([int(r['source_window_index'])==0 for r in res['test'][3]])
print('window0 energy in first 22 samples share',float((t[w,:22]**2).sum()/(t[w]**2).sum()),'max abs mean',float(np.abs(t[w]).max(1).mean()))
print('window0 inter-window corr (mean pairwise, 50):',float(np.corrcoef(t[w][:50]).mean()))
np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(__file__)),'fsdd_clean.npz'),**{s+'X':res[s][2].astype(np.float32) for s in res},**{s+'M':mk(res[s][3]).astype(np.float32) for s in res})
