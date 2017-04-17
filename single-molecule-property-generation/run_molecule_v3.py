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

molname = [sys.argv[1]]
mol_filename = ['Mol2_files/'+m+'.mol2' for m in molname]
time_step = 0.8 #Femtoseconds
temperature = 300 #kelvin
friction = 1 # per picosecond
num_steps = 7500000
trj_freq = 1000 #steps
data_freq = 1000 #steps

# Load OEMol
for ind,j in enumerate(mol_filename):
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
    positions = Quantity(positions, unit.angstroms)
    
    
    # Load forcefield
    forcefield = ForceField(get_data_filename('forcefield/smirff99Frosst.ffxml'))

    # Define system
    topology = generateTopologyFromOEMol(mol)
    params = forcefield.getParameter(smirks='[#1:1]-[#8]')
    params['rmin_half']='0.01'
    params['epsilon']='0.01'
    forcefield.setParameter(params, smirks='[#1:1]-[#8]')
    system = forcefield.createSystem(topology, [mol])

    paramlist1 = np.arange(float(sys.argv[2]),float(sys.argv[3]),float(sys.argv[4]))
    j = sys.argv[5]
    
    smirkseries = sys.argv[6]#'[#6X4:1]-[#1:2]'
    
    paramtype1 = sys.argv[7]#'length'
    paramtype2 = sys.argv[8]


    param = forcefield.getParameter(smirks=smirkseries)
    for i in paramlist1:
        param[paramtype1] = str(i)
        param[paramtype2] = str(j)
        forcefield.setParameter(param, smirks=smirkseries)
        system = forcefield.createSystem(topology, [mol])


        #Do simulation
        integrator = mm.LangevinIntegrator(temperature*kelvin, friction/picoseconds, time_step*femtoseconds)
        platform = mm.Platform.getPlatformByName('Reference')
        simulation = app.Simulation(topology, system, integrator)
        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(temperature*kelvin)
        netcdf_reporter = NetCDFReporter('traj4ns_c1143/'+molname[ind]+'_'+smirkseries+'_'+paramtype1+str(i)+'_'+paramtype2+str(j)+'.nc', trj_freq)
        simulation.reporters.append(netcdf_reporter)
        simulation.reporters.append(app.StateDataReporter('StateData4ns_c1143/data_'+molname[ind]+'_'+smirkseries+'_'+paramtype1+str(i)+'_'+paramtype2+str(j)+'.csv', data_freq, step=True, potentialEnergy=True, temperature=True, density=True))

        print("Starting simulation")
        start = time.clock()
        simulation.step(num_steps)
        end = time.clock()

        print("Elapsed time %.2f seconds" % (end-start))
        netcdf_reporter.close()
        print("Done!")

#Do simulation
#integrator = mm.LangevinIntegrator(temperature*kelvin, friction/picoseconds, time_step*femtoseconds)
#platform = mm.Platform.getPlatformByName('Reference')
#simulation = app.Simulation(topology, system, integrator)
#simulation.context.setPositions(positions)
#simulation.context.setVelocitiesToTemperature(temperature*kelvin)
#netcdf_reporter = NetCDFReporter('traj/'+molname+'.nc', trj_freq)
#simulation.reporters.append(netcdf_reporter)
#simulation.reporters.append(app.StateDataReporter('StateData/data_'+molname+'.csv', data_freq, step=True, potentialEnergy=True, temperature=True, density=True))

#print("Starting simulation")
#start = time.clock()
#simulation.step(num_steps)
#end = time.clock()

#print("Elapsed time %.2f seconds" % (end-start))
#netcdf_reporter.close()
#print("Done!")
