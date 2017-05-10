import subprocess as sb
import re
import pdb
import threading
import time
import sys

fin = open('slurm_test.sh','r')
origlines = fin.readlines()
fin.close()
print origlines

fin = open('slurm_test.sh','w+')
fin.writelines(origlines[0:13])
lines = fin.readlines()
fin.close()
print lines

list = ['string1','string2','string3']
paramchanges = 'some bullshit'

for i,j in enumerate(list):
    fout=open('slurm_test.sh','a')
    newline = 'python simulate_for_posterior.py ' + list[i] + ' ' + paramchanges + '& \n'
    fout.write(newline)
    fout.close()

fout=open('slurm_test.sh','a')
newline = 'wait \n'
fout.write(newline)
fout.close()

p = sb.Popen(['sbatch ./slurm_test_subproc.sh'],shell=True,stdout=sb.PIPE)
out1 = p.communicate()
jobid = out1[0][-9:-1]


for i in range(10000000):
    out = sb.Popen(["squeue", "--job", str(jobid)],
                   stdout=sb.PIPE)
    lines = out.stdout.readlines()
    if len(lines)>1:
        print "The job is still running"
        print lines
    else:
        print "The job is finished!!"
        break
    time.sleep(60)

print "well, this was easier"

