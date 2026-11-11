import csv,glob,os,collections,sys
import numpy as np
DB='/home/ensismoebius/Documentos/academico/UNESP/doutorado/databases/fsdDataset/recordings'
files=sorted(glob.glob(DB+'/*.wav'))
RB='/home/ensismoebius/Repos/doutorado/software/nn/results/meeting01_t1/%s_c1_A/meeting01_t1_%s_c1_A_%s_fold0_target_%s_windows_meta.csv'
for s in ('train','val','test'):
    M=list(csv.DictReader(open(RB%('fsdd','fsdd','fsdd',s))))
    print('FSDD',s,'windows',len(M),'columns',list(M[0])[:12] if s=='train' else '')
    rid=[int(r['recording_id']) for r in M]; rs=sorted(set(rid))
    names=[os.path.basename(files[i])[:-4].split('_') for i in rs]
    print('  recordings',len(rs),'id range',rs[0],rs[-1],'by speaker',sorted(collections.Counter(n[1] for n in names).items()))
    print('  digits(recs)',sorted(collections.Counter(int(n[0]) for n in names).items()))
    print('  digit windows',sorted(collections.Counter(int(os.path.basename(files[i])[0]) for i in rid).items()))
    c=collections.Counter(rid); print('  windows per rec: min',min(c.values()),'max',max(c.values()),'mean',round(np.mean(list(c.values())),2))
    print('  source_window_index hist',sorted(collections.Counter(int(r['source_window_index']) for r in M).items())[:8])
    if s=='test':
        for d in ('train','val'): pass
for s in ('train','val','test'):
    M=list(csv.DictReader(open(RB%('eegmmidb','eegmmidb','eegmmidb',s))))
    rid=[int(r['recording_id']) for r in M]; c=collections.Counter(rid)
    k=[x for x in M[0] if x in('speaker','speaker_id','subject','subject_id')]
    sub=collections.defaultdict(set)
    for r in M: sub[r[k[0]]].add(int(r['recording_id']))
    print('EEG',s,'windows',len(M),'recs',len(c),'subjects',len(sub),'id range',min(rid),max(rid),'win/rec min/mean/max',min(c.values()),round(np.mean(list(c.values())),2),max(c.values()))
    print('  recs per subject (first 25)',sorted(((k2,len(v)) for k2,v in sub.items()),key=lambda t:t[0])[:25])
    print('  source idx max',max(int(r['source_window_index']) for r in M), 'channel col?',[x for x in M[0] if 'chan' in x.lower()])
