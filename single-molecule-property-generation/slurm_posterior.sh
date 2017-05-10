#!/bin/bash

#SBATCH --job-name 'Single Molecule Evidence simulation for small case'
#SBATCH --qos janus
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 12
#SBATCH --time 20:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=brma3379@colorado.edu

#some modules I use
ml slurm

python simulate_for_posterior.py AlkEthOH_c0 "[['[#6X4:1]-[#1:2]', 'k'], ['[#6X4:1]-[#1:2]', 'length'], ['[#6X4:1]-[#6X4:2]', 'k'], ['[#6X4:1]-[#6X4:2]', 'length']]" "[500, 0.8, 700, 1.6]" & 
python simulate_for_posterior.py AlkEthOH_r0 "[['[#6X4:1]-[#1:2]', 'k'], ['[#6X4:1]-[#1:2]', 'length'], ['[#6X4:1]-[#6X4:2]', 'k'], ['[#6X4:1]-[#6X4:2]', 'length']]" "[500, 0.8, 700, 1.6]" & 
python simulate_for_posterior.py AlkEthOH_r405 "[['[#6X4:1]-[#1:2]', 'k'], ['[#6X4:1]-[#1:2]', 'length'], ['[#6X4:1]-[#6X4:2]', 'k'], ['[#6X4:1]-[#6X4:2]', 'length']]" "[500, 0.8, 700, 1.6]" & 
wait 
