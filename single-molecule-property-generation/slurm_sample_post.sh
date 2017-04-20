#!/bin/bash

#SBATCH --job-name 'sample posterior probability distribution of parameters k_bond and x_0 and save 2d hexmap'
#SBATCH --qos janus
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 12
#SBATCH --time 05:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=brma3379@colorado.edu

#some modules I use
ml slurm

#commands
python graph_bl.py
wait
