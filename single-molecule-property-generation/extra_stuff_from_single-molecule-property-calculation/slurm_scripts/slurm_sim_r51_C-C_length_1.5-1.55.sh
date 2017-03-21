#!/bin/bash

#SBATCH --job-name 'Single Molecule Parameter Exploration AlkEthOH_r51'
#SBATCH --qos janus
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --time 19:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=brma3379@colorado.edu

#some modules I use
ml slurm

#commands
python run_molecule.py AlkEthOH_r51 1.5000 1.5500 0.025 [#6X4:1]-[#6X4:2]
