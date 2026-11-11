import csv,os,glob,wave,numpy as np,collections
ROOT=os.environ['ROOT']
DB='/home/ensismoebius/Documentos/academico/UNESP/doutorado/databases/fsdDataset/recordings'
files=sorted(glob.glob(DB+'/*.wav'))   # same order as std::sort on paths
def rd(f):
    w=wave.open(f); n=w.getnframes(); s=np.frombuffer(w.readframes(n),dtype=np.int16).astype(np.float32); return s,w.getframerate()
META=ROOT+'/software/nn/results/meeting01_t1/fsdd_c1_A/meeting01_t1_fsdd_c1_A_fsdd_fold0_target_%s_windows_meta.csv'
def windows(split):
    X=[];Mk=[];dig=[];rec=[];bad=0
    for r in csv.DictReader(open(META%split)):
        rid=int(r['recording_id']); i=int(r['source_window_index']); v=int(r['valid_length'])
        f=files[rid]; s,sr=rd(f)
        seg=s[i*256:(i+1)*256]
        if len(seg)!=v: bad+=1
        x=np.zeros(256,np.float32); x[:len(seg)]=seg
        m=np.zeros(256,np.float32); m[:len(seg)]=1
        mu=seg.mean(); sd=seg.std(); x[:len(seg)]=(seg-mu)/max(sd,1e-8)
        X.append(x);Mk.append(m);dig.append(int(os.path.basename(f)[0]));rec.append(rid)
    return np.array(X),np.array(Mk),np.array(dig),np.array(rec),bad
if __name__=='__main__':
    print(len(files),files[0][-16:],files[300][-16:])
    for s in ('train','val','test'):
        X,M,d,r,bad=windows(s)
        print(s,X.shape,'valid_length mismatches',bad,'digits',sorted(collections.Counter(d.tolist()).items()))
        # poisson probability: min-max scaled
        lo=X.min(1,keepdims=True);hi=X.max(1,keepdims=True);p=(X-lo)/(hi-lo+1e-12)
        print(' mean Poisson p over valid samples',float((p*M).sum()/M.sum()),'mean sample std of window (should be ~1)',float(((X**2*M).sum(1)/M.sum(1)).mean()))
