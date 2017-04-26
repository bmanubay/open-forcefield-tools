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

#commands
python run_molecule_v2.py AlkEthOH_c0 1.09 1.11 0.02 680 700 20 [#6X4:1]-[#1:2] length k &
python run_molecule_v2.py AlkEthOH_c38 1.09 1.11 0.02 680 700 20 [#6X4:1]-[#1:2] length k &
python run_molecule_v2.py AlkEthOH_c488 1.09 1.11 0.02 680 700 20 [#6X4:1]-[#1:2] length k &
python run_molecule_v2.py AlkEthOH_c901 1.09 1.11 0.02 680 700 20 [#6X4:1]-[#1:2] length k &
python run_molecule_v2.py AlkEthOH_c1143 1.09 1.11 0.02 680 700 20 [#6X4:1]-[#1:2] length k &
python run_molecule_v2.py AlkEthOH_c1162 1.09 1.11 0.02 680 700 20 [#6X4:1]-[#1:2] length k &
python run_molecule_v2.py AlkEthOH_c1163 1.09 1.11 0.02 680 700 20 [#6X4:1]-[#1:2] length k &
python run_molecule_v2.py AlkEthOH_c1284 1.09 1.11 0.02 680 700 20 [#6X4:1]-[#1:2] length k &
python run_molecule_v2.py AlkEthOH_c1285 1.09 1.11 0.02 680 700 20 [#6X4:1]-[#1:2] length k &
python run_molecule_v2.py AlkEthOH_c1302 1.09 1.11 0.02 680 700 20 [#6X4:1]-[#1:2] length k &
python run_molecule_v2.py AlkEthOH_r131 1.09 1.11 0.02 680 700 20 [#6X4:1]-[#1:2] length k &
wait
