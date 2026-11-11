import csv,glob,collections,os,sys
import numpy as np
root='/home/ensismoebius/Repos/doutorado/software/nn/results/meeting01_t1'
print(sorted(os.listdir(root))[:30])
for d in sorted(glob.glob(root+'/fsdd_c1_*')+glob.glob(root+'/fsdd_c*')):
    pass
ds=sorted(set(glob.glob(root+'/*')))
print(open(glob.glob(root+'/fsdd_c1_A/*per_window_errors.csv')[0]).readline())
