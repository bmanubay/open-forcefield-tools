#!/bin/bash

#SBATCH --job-name 'Calculate C-H bond length stats across large k_bond and x_0 parameter space and save'
#SBATCH --qos janus
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 12
#SBATCH --time 20:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=brma3379@colorado.edu

#some modules I use
ml slurm

#commands
python sample_posterior.py
wait
