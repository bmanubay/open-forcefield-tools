#!/bin/bash

#SBATCH --job-name 'Single Molecule Parameter Exploration general'
#SBATCH --qos janus
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 12
#SBATCH --time 20:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=brma3379@colorado.edu

#some modules I use
ml slurm

#commands
python run_molecule_v3.py AlkEthOH_c1143 1.27 1.41 0.02 980 [#6X4:1]-[#1:2] length k &
python run_molecule_v3.py AlkEthOH_c1143 0.97 1.09 0.02 980 [#6X4:1]-[#1:2] length k &
python run_molecule_v3.py AlkEthOH_c1143 1.27 1.41 0.02 380 [#6X4:1]-[#1:2] length k &
python run_molecule_v3.py AlkEthOH_c1143 0.97 1.09 0.02 380 [#6X4:1]-[#1:2] length k &
python run_molecule_v3.py AlkEthOH_c1143 880 1000 20 1.39 [#6X4:1]-[#1:2] k length &
python run_molecule_v3.py AlkEthOH_c1143 880 1000 20 0.79 [#6X4:1]-[#1:2] k length &
python run_molecule_v3.py AlkEthOH_c1143 580 680 20 1.39 [#6X4:1]-[#1:2] k length &
python run_molecule_v3.py AlkEthOH_c1143 580 680 20 0.79 [#6X4:1]-[#1:2] k length &
python run_molecule_v3.py AlkEthOH_c1143 880 1000 20 1.24 [#6X4:1]-[#1:2] k length &
python run_molecule_v3.py AlkEthOH_c1143 880 1000 20 0.94 [#6X4:1]-[#1:2] k length &
python run_molecule_v3.py AlkEthOH_c1143 580 680 20 1.24 [#6X4:1]-[#1:2] k length &
python run_molecule_v3.py AlkEthOH_c1143 580 680 20 0.94 [#6X4:1]-[#1:2] k length &
wait
