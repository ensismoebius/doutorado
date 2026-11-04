import numpy as np,csv,sys,os
SC=os.path.dirname(os.path.abspath(__file__))
m=list(csv.DictReader(open('/home/ensismoebius/Repos/doutorado/software/nn/results/meeting01_t1/eegmmidb_c1_A/meeting01_t1_eegmmidb_c1_A_eegmmidb_fold0_target_test_windows_meta.csv')))
a=np.load(SC+'/eegerr_lif_41.npy'); b=np.load(SC+'/eegerr_lrelu_41.npy'); d=a-b
rec=np.array([int(x['recording_id']) for x in m]); sub=np.array([x['speaker_id'] for x in m])
R=sorted(set(rec)); dr=np.array([d[rec==r].mean() for r in R]); g=np.array([sub[rec==r][0] for r in R])
u=sorted(set(g)); n=np.array([(g==s).sum() for s in u]); me=np.array([dr[g==s].mean() for s in u])
msb=(n*(me-dr.mean())**2).sum()/(len(u)-1); msw=sum(((dr[g==s]-dr[g==s].mean())**2).sum() for s in u)/(len(dr)-len(u)); n0=n.mean()
icc=(msb-msw)/(msb+(n0-1)*msw)
print('mean d',dr.mean(),'subjects',len(u),'recordings',len(dr),'ICC',icc,'design effect',1+(n0-1)*icc,'sd subj means',me.std(ddof=1),'subjects with d>0',int((me>0).sum()))
print('naive se',dr.std(ddof=1)/np.sqrt(len(dr)),'cluster se',me.std(ddof=1)/np.sqrt(len(u)))
