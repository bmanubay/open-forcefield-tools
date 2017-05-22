#!/bin/env python

import time
import simtk.openmm as mm
from simtk.openmm import app
from simtk.openmm import Platform
from simtk.unit import *
import numpy as np
from mdtraj.reporters import NetCDFReporter
from smarty import *
import sys
import numpy as np
from ast import literal_eval

molname = [sys.argv[1]]
SMIRKS_and_params = literal_eval(sys.argv[2])
theta_current = literal_eval(sys.argv[3])
mol_filename = ['Mol2_files/'+m+'.mol2' for m in molname]
time_step = 0.8 #Femtoseconds
temperature = 300 #kelvin
friction = 1 # per picosecond
num_steps = 2500000
trj_freq = 1000 #steps
data_freq = 1000 #steps

# Load OEMol
for moldex,j in enumerate(mol_filename):
    mol = oechem.OEGraphMol()
    ifs = oechem.oemolistream(j)
    flavor = oechem.OEIFlavor_Generic_Default | oechem.OEIFlavor_MOL2_Default | oechem.OEIFlavor_MOL2_Forcefield
    ifs.SetFlavor( oechem.OEFormat_MOL2, flavor)
    oechem.OEReadMolecule(ifs, mol )
    oechem.OETriposAtomNames(mol)

    # Get positions
    coordinates = mol.GetCoords()
    natoms = len(coordinates)
    positions = np.zeros([natoms,3], np.float64)
    for index in range(natoms):
        (x,y,z) = coordinates[index]
        positions[index,0] = x
        positions[index,1] = y
        positions[index,2] = z
    positions = Quantity(positions, angstroms)

    # Load forcefield
    forcefield = ForceField(get_data_filename('forcefield/smirff99Frosst.ffxml'))

    # Define system
    topology = generateTopologyFromOEMol(mol)
    params = forcefield.getParameter(smirks='[#1:1]-[#8]')
    params['rmin_half']='0.01'
    params['epsilon']='0.01'
    forcefield.setParameter(params, smirks='[#1:1]-[#8]')
    for ind,i in enumerate(SMIRKS_and_params):
        params = forcefield.getParameter(smirks=i[0])
        print params
        params[i[1]]=str(theta_current[ind])
        print params
        forcefield.setParameter(params,smirks=i[0])
    system = forcefield.createSystem(topology, [mol])
    filename_string = []
    for ind,i in enumerate(SMIRKS_and_params):
        temp = i[0]+'_'+i[1]+'_'+str(theta_current[ind])
        filename_string.append(temp)
    filename_string = '_'.join(filename_string)

    #Do simulation
    integrator = mm.LangevinIntegrator(temperature*kelvin, friction/picoseconds, time_step*femtoseconds)
    platform = mm.Platform.getPlatformByName('Reference')
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(temperature*kelvin)
    traj_name = 'traj_posterior/'+molname[moldex]+'_'+filename_string+'.nc'
    netcdf_reporter = NetCDFReporter(traj_name, trj_freq)
    simulation.reporters.append(netcdf_reporter)
    simulation.reporters.append(app.StateDataReporter('StateData_posterior/data_'+molname[moldex]+'_'+filename_string+'.csv', data_freq, step=True, 
                                potentialEnergy=True, temperature=True, density=True))

    print("Starting simulation")
    start = time.clock()
    simulation.step(num_steps)
    end = time.clock()

    print("Elapsed time %.2f seconds" % (end-start))
    netcdf_reporter.close()
    print("Done!")

