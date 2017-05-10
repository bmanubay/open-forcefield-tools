from collections import OrderedDict


theta_init = OrderedDict([('[#6X4:1]-[#1:2]',OrderedDict([('k',500),('length',0.8)])),('[#6X4:1]-[#6X4:2]',OrderedDict([('k',700),('length',1.6)]))])

SMIRKS_and_params = [[j,jj] for j in theta_init for jj in theta_init[j]]

theta_current = [theta_init[j][jj] for j in theta_init for jj in theta_init[j]]

mol_list = ['AlkEthOH_c0','AlkEthOH_r0','AlkEthOH_r405']

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
    #newline = 'python simulate_for_posterior.py '+mol_list[i]+' '+'"SMIRKS_and_params"'+' '+str(theta_current)+' '+'& \n' 
    newline = """python simulate_for_posterior.py %s "%s" "%s" & \n""" %(mol_list[i],SMIRKS_and_params,theta_current)
    fout.write(newline)
    fout.close()

fout=open('slurm_test.sh','a')
newline = 'wait \n'
fout.write(newline)
fout.close()

