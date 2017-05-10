#!/bin/bash

#SBATCH --job-name 'Single Molecule Evidence simulation for small case'
#SBATCH --qos janus
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 12
#SBATCH --time 00:10:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=brma3379@colorado.edu

#some modules I use
ml slurm

python helloworld.py &
python countto100000.py &
wait 
