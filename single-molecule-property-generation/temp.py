# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:09:09 2017

@author: Bryce Manubay
"""

"""
This script represents prototyping for constructing posterior distributions
across parameter space given some initial set of data.

Author: Bryce Manubay
"""

import matplotlib as mpl

mpl.use('Agg')


from smarty.forcefield import *
import openeye
from openeye import oechem
from smarty import *
from smarty.utils import get_data_filename
from simtk import openmm as mm
from simtk.unit import *
from simtk.openmm import app
from simtk.openmm import Platform
import time
from mdtraj.reporters import NetCDFReporter
import numpy as np
import netCDF4 as netcdf
import collections as cl
import pandas as pd
import pymbar
from pymbar import timeseries
import glob
import sys
from smarty.forcefield import generateTopologyFromOEMol
import pdb
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import scipy as sp
import seaborn as sns

from scipy.stats import norm
from scipy.stats import multivariate_normal
import sys

sns.set_style('white')
sns.set_context('talk')

np.random.seed(123)

data = np.random.randn(20)

#----------------------------------------------------------------------
# CONSTANTS
#----------------------------------------------------------------------

kB = 0.001987204118  #Boltzmann constant (Gas constant) in kcal/(mol*K)

#ax = plt.subplot()
#sns.distplot(data, kde=False, ax=ax)
#_ = ax.set(title='Histogram of observed data', xlabel='x', ylabel='# observations');

#----------------------------------------------------------------------
# Functions for getting initial data
# Utilities for reading trajectories and running simulations
#----------------------------------------------------------------------
def read_traj(ncfiles,indkeep=100000000):
    """
    Take multiple .nc files and read in coordinates in order to re-valuate energies based on parameter changes

    Parameters
    -----------
    ncfiles - a list of trajectories in netcdf format

    Returns
    ----------
    data - all of the data contained in the netcdf file
    xyzn - the coordinates from the netcdf in angstroms
    """

    data = netcdf.Dataset(ncfiles)
    xyz = data.variables['coordinates']
    xyzn = unit.Quantity(xyz[-indkeep:], unit.angstroms)

    return data, xyzn
#----------------------------------------------------------------------
def read_or_run(optparam):
    """
    Function initializes reading in a trajectory or running a simulation for the inference process. If the file name we seek
    exists (based on the `optparam` parameter we feed it) this function will simply read in the trajectory coordinates and 
    record the parameters of the state. If the file name we're looking for does not exist then we will simulate the state 
    based on the argument passed to `optparam`.

    `optparam`: this is an argument used to construct the trajectory file name we seek. In the event that the trajectory file doesn't exist
                the function will instead launch a simulation of the OpenMM platform in order to get the simulation data we need to initialize the
                posterior sampling process. The form of the `optparam` argument should be a dictionary of length 3 lists indicating all of the changes in 
                parameter to make where the key is the molecule AlkEthOH ID. Specifically, we need to indicate the SMIRKS string and the parameter type 
                to be changed as well as the new value.
                I.e. :
                       optparam = {'AlkEthOH_c100':[['[#6X4:1]-[#1:2]','k',650],['[#6X4:1]-[#1:2]','length',1.1],
                                                   ['[#6X4:1]-[#6X4:2]-[#8X2H1:3]-[#1:4]','k1',0.2],...]}
    """
    filename_string = []
    AlkEthOH_IDs = []

    for i in optparam:
        AlkEthOH_IDs.append(i)
    for AlkEthOH_ID in AlkEthOH_IDs:
        for j in optparam[AlkEthOH_ID]:
            temp = j[0]+'_'+j[1]+'_'+j[2]
            filename_string.append(temp)
    filename_string = '_'.join(filename_string)
    param_list = optparam[AlkEthOH_IDs[0]]
    
    try:
        #AlkEthOH_id = traj_name.slit('_')[:2]
        #AlkEthOH_ID = AlkEthOH_id[0] + AlkEthOH_id[1]
        #param_list = traj_name.rsplit('.',1)[:-1][0].split('_')[2:] 
        traj_name = AlkEthOH_IDs[0]+'_'+filename_string+'.nc'
        print traj_name
        data, xyzn = read_traj(traj_name) 
    except:
        print('File does not exist!') 
        print('Simulating specified state instead')
        for AlkEthOH_ID in optparam:
            molname = [AlkEthOH_ID]
            mol_filename = ['Mol2_files/'+m+'.mol2' for m in molname]
            time_step = 0.8 #Femtoseconds
            temperature = 300 #kelvin
            friction = 1 # per picosecond
            num_steps = 5000000
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
                positions = Quantity(positions, angstroms)
                
                # Load forcefield
                forcefield = ForceField(get_data_filename('forcefield/smirff99Frosst.ffxml'))

                # Define system
                topology = generateTopologyFromOEMol(mol)
                params = forcefield.getParameter(smirks='[#1:1]-[#8]')
                params['rmin_half']='0.01'
                params['epsilon']='0.01'
                forcefield.setParameter(params, smirks='[#1:1]-[#8]')
                for i in optparam[AlkEthOH_ID]:
                    params = forcefield.getParameter(smirks=i[0])
                    params[i[1]]=i[2]
                    forcefield.setParameter(params,smirks=i[0]) 
                system = forcefield.createSystem(topology, [mol])       
                
                filename_string = []
                for i in optparam[AlkEthOH_ID]:
                    temp = i[0]+'_'+i[1]+'_'+i[2]
                    filename_string.append(temp)
                filename_string = '_'.join(filename_string)
                 
                #Do simulation
                integrator = mm.LangevinIntegrator(temperature*kelvin, friction/picoseconds, time_step*femtoseconds)
                platform = mm.Platform.getPlatformByName('Reference')
                simulation = app.Simulation(topology, system, integrator)
                simulation.context.setPositions(positions)
                simulation.context.setVelocitiesToTemperature(temperature*kelvin)
                traj_name = 'traj4ns/'+molname[ind]+'_'+filename_string+'.nc'
                netcdf_reporter = NetCDFReporter(traj_name, trj_freq)
                simulation.reporters.append(netcdf_reporter)
                #simulation.reporters.append(app.StateDataReporter('StateData4ns/data_'+molname[ind]+'_'+smirkseries+'_'+paramtype+str(i)+'.csv', data_freq, step=True, potentialEnergy=True, temperature=True, density=True))

                print("Starting simulation")
                start = time.clock()
                simulation.step(num_steps)
                end = time.clock()

                print("Elapsed time %.2f seconds" % (end-start))
                netcdf_reporter.close()
                print("Done!")
        data, xyzn = read_traj(traj_name)                                     
    return AlkEthOH_IDs, param_list, xyzn
#----------------------------------------------------------------------
read_or_run(optparam = {'AlkEthOH_c1143':[['[#6X4:1]-[#1:2]','k','650'],
                                          ['[#6X4:1]-[#1:2]','length','1.1'],
                                          ['[#6X4:1]-[#6X4:2]-[#8X2H1:3]-[#1:4]','k1','0.2']]})

##########################################################################################################################
##########################################################################################################################

#----------------------------------------------------------------------
# Utility functions for reweighting simulated data to new states
#----------------------------------------------------------------------
def constructDataFrame(mol_files):
    """ 
    Construct a pandas dataframe to be populated with computed single molecule properties. Each unique bond, angle and torsion has it's own column for a value
    and uncertainty.
    
    Parameters
    -----------
    mol_files -  a list of mol2 files from which we determine connectivity using OpenEye Tools and construct the dataframe using Pandas.
    
    Returns
    -----------
    df - data frame in form molecules x property id that indicates if a specific property exists for a molecule (1 in cell if yes, 0 if no)
    """    
    
    molnames = []
    for i in mol_files:
        molname = i.replace(' ', '')[:-5]
        molname = molname.rsplit('/' ,1)[1]
        molnames.append(molname)

    OEMols=[]
    for i in mol_files:
        mol = oechem.OEGraphMol()
        ifs = oechem.oemolistream(i)
        flavor = oechem.OEIFlavor_Generic_Default | oechem.OEIFlavor_MOL2_Default | oechem.OEIFlavor_MOL2_Forcefield
        ifs.SetFlavor(oechem.OEFormat_MOL2, flavor)
        oechem.OEReadMolecule(ifs, mol)
        oechem.OETriposAtomNames(mol)
        OEMols.append(mol)

    ff = ForceField(get_data_filename('/data/forcefield/smirff99Frosst.ffxml'))
    
    

    labels = []
    lst0 = []
    lst1 = []
    lst2 = []
    lst00 = [[] for i in molnames]
    lst11 = [[] for i in molnames]
    lst22 = [[] for i in molnames] 
    lst_0 = [[] for i in molnames]
    lst_1 = [[] for i in molnames]
    lst_2 = [[] for i in molnames] 
  

 
    for ind, val in enumerate(OEMols):
        label = ff.labelMolecules([val], verbose = False) 
        for entry in range(len(label)):
            for bond in label[entry]['HarmonicBondGenerator']:
                lst0.extend([str(bond[0])])
	        lst00[ind].extend([str(bond[0])])
                lst_0[ind].append([str(bond[0]),str(bond[2])])
	    for angle in label[entry]['HarmonicAngleGenerator']:
	        lst1.extend([str(angle[0])])
	        lst11[ind].extend([str(angle[0])])
                lst_1[ind].append((str(angle[0]),str(angle[2])))
	    for torsion in label[entry]['PeriodicTorsionGenerator']:  
                lst2.extend([str(torsion[0])])
	        lst22[ind].extend([str(torsion[0])])
                lst_2[ind].append([str(torsion[0]),str(torsion[2])])

    # Return unique strings from lst0
    cols0 = set()
    for x in lst0:
	cols0.add(x)
    cols0 = list(cols0)


    # Generate data lists to populate dataframe
    data0 = [[] for i in range(len(lst00))]
    for val in cols0:
	for ind,item in enumerate(lst00):
	    if val in item:
		data0[ind].append(1)
	    else: 
		data0[ind].append(0)

    # Return unique strings from lst1
    cols1 = set()
    for x in lst1:
	cols1.add(x)
    cols1 = list(cols1)

    # Generate data lists to populate frame (1 means val in lst11 was in cols1, 0 means it wasn't)
    data1 = [[] for i in range(len(lst11))]
    for val in cols1:
	for ind,item in enumerate(lst11):
	    if val in item:
		data1[ind].append(1)
	    else: 
	        data1[ind].append(0)

    # Return unique strings from lst2
    cols2 = set()
    for x in lst2:
	cols2.add(x)
    cols2 = list(cols2)   
    
    # Generate data lists to populate frame (1 means val in lst22 was in cols2, 0 means it wasn't)
    data2 = [[] for i in range(len(lst22))]
    for val in cols2:
	for ind,item in enumerate(lst22):
	    if val in item:
		data2[ind].append(1)
	    else: 
		data2[ind].append(0)

    # Clean up clarity of column headers and molecule names
    cols0t = ["BondEquilibriumLength " + i for i in cols0]
    cols0temp = ["BondEquilibriumLength_std " + i for i in cols0]
    cols0 = cols0t + cols0temp

    cols1t = ["AngleEquilibriumAngle " + i for i in cols1]
    cols1temp = ["AngleEquilibriumAngle_std " + i for i in cols1]
    cols1 = cols1t + cols1temp

    cols2t = ["TorsionFourier1 " + i for i in cols2]
    cols2temp = ["TorsionFourier1_std " + i for i in cols2]
    cols2 = cols2t + cols2temp

    data0 = [i+i for i in data0]
    data1 = [i+i for i in data1]
    data2 = [i+i for i in data2]

    # Construct dataframes
    df0 = pd.DataFrame(data = data0, index = molnames, columns = cols0)
    df0['molecule'] = df0.index
    df1 = pd.DataFrame(data = data1, index = molnames, columns = cols1)
    df1['molecule'] = df1.index
    df2 = pd.DataFrame(data = data2, index = molnames, columns = cols2)
    df2['molecule'] = df2.index

    dftemp = pd.merge(df0, df1, how = 'outer', on = 'molecule')
    df = pd.merge(dftemp, df2, how = 'outer', on = 'molecule')

    return df, lst_0, lst_1, lst_2

#------------------------------------------------------------------

def ComputeBondsAnglesTorsions(xyz, bonds, angles, torsions):
    """ 
    compute a 3 2D arrays of bond lengths for each frame: bond lengths in rows, angle lengths in columns
    
    Parameters 
    -----------
    xyz - xyz files, an array of length-2 arrays
    bonds, angles, torsions - numbered atom indices tuples associated with all unqiue bonds, angles and torsions
 
    Returns
    ----------
    bond_dist, angle_dist, torsion_dist - computed bonds, angles and torsions across the provided time series
    """

    niterations = xyz.shape[0] # no. of frames
    natoms = xyz.shape[1]
    
    nbonds = np.shape(bonds)[0]
    nangles = np.shape(angles)[0]
    ntorsions = np.shape(torsions)[0] 
    bond_dist = np.zeros([niterations,nbonds])
    angle_dist = np.zeros([niterations,nangles])
    torsion_dist = np.zeros([niterations,ntorsions])

    for n in range(niterations):
        xyzn = xyz[n] # coordinates this iteration
        bond_vectors = np.zeros([nbonds,3])
	for i, bond in enumerate(bonds): 
	    bond_vectors[i,:] = xyzn[bond[0]] - xyzn[bond[1]]  # calculate the length of the vector 
            bond_dist[n,i] = np.linalg.norm(bond_vectors[i]) # calculate the bond distance

        # we COULD reuse the bond vectors and avoid subtractions, but would involve a lot of bookkeeping
        # for now, just recalculate

        bond_vector1 = np.zeros(3)
        bond_vector2 = np.zeros(3)
        bond_vector3 = np.zeros(3)

        for i, angle in enumerate(angles):
            bond_vector1 = xyzn[angle[0]] - xyzn[angle[1]]  # calculate the length of the vector
            bond_vector2 = xyzn[angle[1]] - xyzn[angle[2]]  # calculate the length of the vector
            dot = np.dot(bond_vector1,bond_vector2)
            len1 = np.linalg.norm(bond_vector1)
            len2 = np.linalg.norm(bond_vector2)
            angle_dist[n,i] = np.arccos(dot/(len1*len2))  # angle in radians

        for i, torsion in enumerate(torsions):
            # algebra from http://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates, Daniel's answer
            bond_vector1 = xyzn[torsion[0]] - xyzn[torsion[1]]  # calculate the length of the vector
            bond_vector2 = xyzn[torsion[1]] - xyzn[torsion[2]]  # calculate the length of the vector
            bond_vector3 = xyzn[torsion[2]] - xyzn[torsion[3]]  # calculate the length of the vector
            bond_vector1 /= np.linalg.norm(bond_vector1)
            bond_vector2 /= np.linalg.norm(bond_vector2)
            bond_vector3 /= np.linalg.norm(bond_vector3)
            n1 = np.cross(bond_vector1,bond_vector2)
            n2 = np.cross(bond_vector2,bond_vector3)
            m = np.cross(n1,bond_vector2)
            x = np.dot(n1,n2)
            y = np.dot(m,n2)
            torsion_dist[n,i] = np.arctan2(y,x)  # angle in radians

    return bond_dist, angle_dist, torsion_dist

#------------------------------------------------------------------

def calculateBondsAnglesTorsionsStatistics(properties, bond_dist, angle_dist, torsion_dist, bonds, angles, torsions, torsionbool):

    """
    Parameters
    -----------
    properties: A list of property strings we want value for
    bond_dist: a Niterations x nbonds list of bond lengths
    angle_dist: a Niterations x nbonds list of angle angles (in radians)
    torsion_dist: a Niterations x nbonds list of dihedral angles (in radians)
    bonds: a list of bonds (ntorsions x 2)
    angles: a list of angles (ntorsions x 3)
    torsions: a list of torsion atoms (ntorsions x 4)
    torsionbool: boolean which suppresses torsion statistical analysis if False
    # we assume the bond_dist / bonds , angle_dist / angles, torsion_dist / torsion were constucted in the same order.
    PropertyDict - dictionary of average value of bond, angle or torsion across time series with associated uncertainty in mean and uncertainty in uncertainty
    """
    PropertyDict = dict()
    nbonds = np.shape(bonds)[0]
    nangles = np.shape(angles)[0]
    ntorsions = np.shape(torsions)[0]
    
    nsamp = np.shape(bond_dist)[0]-1 #WARNING: assumes data points uncorrelated!
    for p in properties:        
        AtomList = p.split(' ', 1)[1:]  # figure out which bond this is: 
	AtomList = [i.lstrip('[').rstrip(']') for i in AtomList]  # we assume bond_dist /bond is in the same order.
	for i in AtomList:
            AtomList = i.strip().split(',')
        AtomList = map(int, AtomList) 

        if 'BondEquilibriumLength' in p:
            for i in range(nbonds):
                if np.array_equal(AtomList, bonds[i]): 
                    value = np.mean(bond_dist[:,i])
                    uncertainty = np.std(bond_dist[:,i])/np.sqrt(nsamp)
                    PropertyDict[p] = [value,uncertainty]

        if 'BondEquilibriumLength_std' in p:
            for i in range(nbonds):
        	if np.array_equal(AtomList, bonds[i]): 
                    value = np.std(bond_dist[:,i])
                    uncertainty = np.std(bond_dist[:,i])**2/np.sqrt(nsamp/2)
                    PropertyDict[p] = [value,uncertainty]

	if 'AngleEquilibriumAngle' in p:
       	    for i in range(nangles):
                if np.array_equal(AtomList, angles[i]): 
                    value = np.mean(angle_dist[:,i])
                    uncertainty = np.std(angle_dist[:,i])/np.sqrt(nsamp)
                    PropertyDict[p] = [value,uncertainty]

        if torsionbool==True:
	    if 'TorsionFourier1' in p:
                for i in range(ntorsions):
                    if np.array_equal(AtomList, torsions[i]): 
                    	value = np.mean(torsion_dist[:,i])
                    	uncertainty = np.std(torsion_dist[:,i])/np.sqrt(nsamp)
                    	PropertyDict[p] = [value,uncertainty]

	    if 'TorsionFourier1_std' in p:
	    	    for i in range(ntorsions):
	                if np.array_equal(AtomList, torsions[i]):
	            	    value = np.std(torsion_dist[:,i])
		    	    uncertainty = np.std(torsion_dist[:,i])**2/np.sqrt(nsamp/2)
		    	    PropertyDict[p] = [value,uncertainty]

	# Circular distribution alternate for torsion calculation
        
	    if 'TorsionFourier1' in p:
		for i in range(ntorsions):
		    if np.array_equal(AtomList, torsions[i]):
		        value = np.array([])
			for j in range(nsamp):
			    val = np.real((np.exp(cmath.sqrt(-1)*torsion_dist[:,i]))**j)
			    value = np.append(value, val)
			    value = (1/nsamp)*np.sum(value)
			    uncertainty = np.std(torsion_dist[:,i])/np.sqrt(nsamp)
			    PropertyDict[p] = [value, uncertainty]

	    if 'TorsionFourier1_std' in p:
		for i in range(ntorsions):
                    if np.array_equal(AtomList, torsions[i]):
                        value = np.std(torsion_dist[:,i])
                        uncertainty = np.std(torsion_dist[:,i])**2/np.sqrt(nsamp/2)
                        PropertyDict[p] = [value,uncertainty]
	else:
	    pass
                 
    return PropertyDict

#------------------------------------------------------------------

def get_properties_from_trajectory(mol2, ncfiles, torsionbool=True):

    """
    take multiple .nc files with identifier names and a pandas dataframe with property 
    names for single atom bonded properties (including the atom numbers) and populate 
    those property pandas dataframe.
    
    Parameters
    -----------
    mol2 - mol2 files used to identify and index molecules  
    ncfiles -  a list of trajectories in netcdf format. Names should correspond to the identifiers in the pandas dataframe.
    torsionbool - boolean value passed to computeBondsAnglesTorsionsStatistics() to supress torsion statistics analysis. Default set to True (torsion calculatio                  n not supressed). 
    
    Returns
    ----------
    bond_dist - calculated bond distribution across trajectory
    angle_dist - calculated angle distribution across trajectory
    torsion_dist - calculated torsion distribution across trajectory
    Properties - dictionary of an average value of bond, angle or torsion across time series with associated uncertainty in mean and uncertainty in uncertainty
    """

    PropertiesPerMolecule = dict()

    # here's code that generate list of properties to calculate for each molecule and 
    # populate PropertiesPerMolecule
     
    mol_files = mol2
   
    df = constructDataFrame(mol_files)
    MoleculeNames = df.molecule.tolist()
    properties = df.columns.values.tolist()
 
    for ind, val in enumerate(MoleculeNames):
        defined_properties  = list()
        for p in properties:
            if (p is not 'molecule') and ('_std' not in p):
                if df.iloc[ind][p] != 0:
		    defined_properties.append(p)
                PropertiesPerMolecule[val] = defined_properties

   
    AtomDict = dict()
    AtomDict['MolName'] = list()
    for fname in ncfiles:
        MoleculeName = fname.split('.')[0]
        AtomDict['MolName'].append(MoleculeName)
         	
        # extract the xyz coordinate for each frame
     
	data = netcdf.Dataset(fname)
        xyz = data.variables['coordinates']
	

        # what is the property list for this molecule
        PropertyNames = PropertiesPerMolecule[MoleculeName]

	# extract the bond/angle/torsion lists
        AtomDict['Bond'] = list()
        AtomDict['Angle'] = list()
        AtomDict['Torsion'] = list()

        # which properties will we use to construct the bond list
        ReferenceProperties = ['BondEquilibriumLength','AngleEquilibriumAngle','TorsionFourier1']
	for p in PropertyNames:
            PropertyName = p.split(' ', 1)[0]
            AtomList = p.split(' ', 1)[1:]
	    AtomList = [i.lstrip('[').rstrip(']') for i in AtomList]
	    for i in AtomList:
                AtomList = i.strip().split(',')
            AtomList = map(int, AtomList) 
            if any(rp in p for rp in ReferenceProperties):
                if 'Bond' in p:
                    AtomDict['Bond'].append(AtomList)
                if 'Angle' in p:
                    AtomDict['Angle'].append(AtomList)
                if 'Torsion' in p:
                    AtomDict['Torsion'].append(AtomList)
         

        bond_dist, angle_dist, torsion_dist = computeBondsAnglesTorsions(xyz,
                                                                         AtomDict['Bond'],
                                                                         AtomDict['Angle'],
                                                                         AtomDict['Torsion'])
		

        Properties = calculateBondsAnglesTorsionsStatistics(PropertyNames,
                                                            bond_dist, angle_dist, torsion_dist,
                                                            AtomDict['Bond'], AtomDict['Angle'], AtomDict['Torsion'], torsionbool)

        #Put properties back in dataframe and return

    return [bond_dist, angle_dist, torsion_dist, Properties]

#------------------------------------------------------------------

def read_col(filename,colname,frames):
    """
    Reads in columns from .csv outputs of OpenMM StateDataReporter 
    
    Parameters
    -----------
    filename (string) - the path to the folder of the csv
    colname (string) - the column you wish to extract from the csv
    frames (integer) - the number of frames you wish to extract		
    
    Returns
    ----------
    dat - the pandas column series written as a matrix
    """

    #print "--Reading %s from %s/..." % (colname,filename)

    # Read in file output as pandas df
    df = pd.read_csv(filename, sep= ',')
	
    # Read values direct from column into numpy array
    dat = df.as_matrix(columns = colname)
    dat = dat[-frames:]


    return dat

#------------------------------------------------------------------

def get_energy(system, positions):
    """
    Return the potential energy.
    Parameters
    ----------
    system : simtk.openmm.System
        The system to check
    positions : simtk.unit.Quantity of dimension (natoms,3) with units of length
        The positions to use
    Returns
    ---------
    energy
    """
     
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = openmm.Context(system, integrator)
    context.setPositions(positions)
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy() / unit.kilocalories_per_mole
    return energy

#------------------------------------------------------------------

def new_param_energy(mol2, traj, smirkss, N_k, params, paramtype, samps, indkeep, *coords):
    """
    Return potential energies associated with specified parameter perturbations.
    Parameters
    ----------
    mol2: mol2 file associated with molecule of interest used to construct OEMol object
    traj: trajectory from the simulation ran on the given molecule
    smirkss: list of smirks strings we wish to apply parameter changes to (Only changing 1 type of string at a time now. All bonds, all angles or all torsions)
    N_k: numpy array of number of samples per state
    params: a numpy array of the parameter values we wish to test
    paramtype: the type of ff param being edited
        BONDS - k (bond force constant), length (equilibrium bond length) 
        ANGLES - k (angle force constant), angle (equilibrium bond angle)
        TORSIONS - k{i} (torsion force constant), idivf{i} (torsional barrier multiplier), periodicity{i} (periodicity of the torsional barrier), phase{i} 
                   (phase offset of the torsion)
        NONBONDED - epsilon and rmin_half (where epsilon is the LJ parameter epsilon and rmin_half is half of the LJ parameter rmin)
    samps: samples per energy calculation
    Returns
    -------
    energies: a list of the energies associated with the forcfield parameters used as input
    """
    #-------------------
    # PARAMETERS
    #-------------------
    params = params
    N_k = N_k
    ncfiles = traj
    

    # Determine number of simulations
    K = np.size(N_k)
    #if np.shape(params) != np.shape(N_k): raise "K_k and N_k must have same dimensions"


    # Determine max number of samples to be drawn from any state

    #-------------
    # SYSTEM SETUP
    #-------------
    verbose = False # suppress echos from OEtoolkit functions
    ifs = oechem.oemolistream(mol2)
    mol = oechem.OEMol()
    # This uses parm@frosst atom types, so make sure to use the forcefield-flavor reader
    flavor = oechem.OEIFlavor_Generic_Default | oechem.OEIFlavor_MOL2_Default | oechem.OEIFlavor_MOL2_Forcefield
    ifs.SetFlavor( oechem.OEFormat_MOL2, flavor)
    oechem.OEReadMolecule(ifs, mol )
    # Perceive tripos types
    oechem.OETriposAtomNames(mol)

    # Get positions for use below
    if not coords:
        data, xyz = readtraj(traj,indkeep)
        #indkeep = int(lentraj*perckeep)
        xyzn = xyz[-indkeep:]
    else:
        xyzn = coords
     
    # Load forcefield file
    ffxml = get_data_filename('forcefield/smirff99Frosst.ffxml')
    ff = ForceField(ffxml)

    # Generate a topology
    from smarty.forcefield import generateTopologyFromOEMol
    topology = generateTopologyFromOEMol(mol)

    #-----------------
    # MAIN
    #-----------------

    # Calculate energies 
    
    energies = np.zeros([len(smirkss),len(params),samps],np.float64)
    for inds,s in enumerate(smirkss):
        temp0 = np.zeros([len(params),samps],np.float64)
        param = ff.getParameter(smirks=s)
        for ind,val in enumerate(params):
            for p in paramtype:
                temp1 = np.zeros(samps,np.float64)
                for a,b in zip(val,p):
                    param[b] = str(a)      
                ff.setParameter(param, smirks = s)
                system = ff.createSystem(topology, [mol], verbose=verbose)
                for i,a in enumerate(xyzn): 
                    e = np.float(get_energy(system, a))
                    energies[inds,ind,i] = e
   
    return energies, xyzn, system

#------------------------------------------------------------------

def get_small_mol_dict(mol2, traj):
    """
    Return dictionary specifying the bond, angle and torsion indices to feed to ComputeBondsAnglesTorsions()
    Parameters
    ----------
    mol2: mol2 file associated with molecule of interest used to determine atom labels
    traj: trajectory from the simulation ran on the given molecule
     
    Returns
    -------
    AtomDict: a dictionary of the bond, angle and torsion indices for the given molecule
    """
    PropertiesPerMolecule = dict()    
    mol_files = [] 
    for i in mol2:
        temp = i
        mol_files.append(temp)
    df,lst_0,lst_1,lst_2 = constructDataFrame(mol_files)
    MoleculeNames = df.molecule.tolist()
    properties = df.columns.values.tolist()
    #print MoleculeNames 
    for ind, val in enumerate(MoleculeNames):
        defined_properties  = list()
        for p in properties:
            if (p is not 'molecule') and ('_std' not in p):
                if df.iloc[ind][p] != 0:
		    defined_properties.append(p)
                PropertiesPerMolecule[val] = defined_properties

   
    AtomDict = dict()
    AtomDict['MolName'] = list()
    for fname in traj:
        MoleculeName = fname.split('.')[0][8:]
        AtomDict['MolName'].append(MoleculeName)
         	
        
        # what is the property list for this molecule
        PropertyNames = PropertiesPerMolecule[MoleculeName]

        # extract the bond/angle/torsion lists
        AtomDict['Bond'] = list()
        AtomDict['Angle'] = list()
        AtomDict['Torsion'] = list()

        # which properties will we use to construct the bond list
        ReferenceProperties = ['BondEquilibriumLength','AngleEquilibriumAngle','TorsionFourier1']
        for p in PropertyNames:
            PropertyName = p.split(' ', 1)[0]
            AtomList = p.split(' ', 1)[1:]
            AtomList = [i.lstrip('[').rstrip(']') for i in AtomList]
	    for i in AtomList:
                AtomList = i.strip().split(',')
            AtomList = map(int, AtomList) 
            if any(rp in p for rp in ReferenceProperties):
                if 'Bond' in p:
                    AtomDict['Bond'].append(AtomList)
                if 'Angle' in p:
                    AtomDict['Angle'].append(AtomList)
                if 'Torsion' in p:
                     AtomDict['Torsion'].append(AtomList)

    return AtomDict,lst_0,lst_1,lst_2

#------------------------------------------------------------------

def subsampletimeseries(timeser,xyzn,N_k):
    """
    Return a subsampled timeseries based on statistical inefficiency calculations.
    Parameters
    ----------
    timeser: the timeseries to be subsampled
    xyzn: the coordinates associated with each frame of the timeseries to be subsampled
    N_k: original # of samples in each timeseries
    
    Returns
    ---------
    N_k_sub: new number of samples per timeseries
    ts_sub: the subsampled timeseries
    xyz_sub: the subsampled configuration series
    """
    # Make a copy of the timeseries and make sure is numpy array of floats
    ts = timeser
    xyz = xyzn

    # initialize array of statistical inefficiencies
    g = np.zeros(len(ts),np.float64)    


    for i,t in enumerate(ts):
        if np.count_nonzero(t)==0:
	    g[i] = np.float(1.)
            print "WARNING FLAG"
        else:
            g[i] = timeseries.statisticalInefficiency(t)
     
    N_k_sub = np.array([len(timeseries.subsampleCorrelatedData(t,g=b)) for t, b in zip(ts,g)])
    ind = [timeseries.subsampleCorrelatedData(t,g=b) for t,b in zip(ts,g)]
    
    #xyz_sub = np.array([unit.Quantity(c[i], unit.angstroms) for c,i in zip(xyz,ind)])
    if (N_k_sub == N_k).all():
        ts_sub = ts
        xyz_sub = xyz
        print "No sub-sampling occurred"
    else:
        print "Sub-sampling..." 
        ts_sub = np.array([t[timeseries.subsampleCorrelatedData(t,g=b)] for t,b in zip(ts,g)])
        for c in xyz:
            xyz_sub = [c[timeseries.subsampleCorrelatedData(t,g=b)] for t,b in zip(ts,g)]
    return ts_sub, N_k_sub, xyz_sub, ind

#------------------------------------------------------------------

##########################################################################################################################
##########################################################################################################################

#------------------------------------------------------------------
# Functions for data fitting and surrogate modeling 
#------------------------------------------------------------------

# In the works


##########################################################################################################################
##########################################################################################################################

#------------------------------------------------------------------
# Functions for MCMC sampling
#------------------------------------------------------------------

def calc_posterior_analytical(data, x, mu_0, sigma_0):
    sigma = 1.
    n = len(data)
    mu_post = (mu_0 / sigma_0**2 + data.sum() / sigma**2) / (1. / sigma_0**2 + n / sigma**2)
    sigma_post = (1. / sigma_0**2 + n / sigma**2)**-1
    return norm(mu_post, np.sqrt(sigma_post)).pdf(x)
#------------------------------------------------------------------

ax1 = plt.subplot()
x = np.linspace(-1, 1, 500)
posterior_analytical = calc_posterior_analytical(data, x, 0., 1.)
ax1.plot(x, posterior_analytical)
ax1.set(xlabel='mu', ylabel='belief', title='Analytical posterior');
sns.despine()
#------------------------------------------------------------------

def sampler(data, samples=4, mu_init=.5, proposal_width=.5, plot=False, mu_prior_mu=0, mu_prior_sd=1.):
    """
    Outline:
    1)Take data and calculate observable
    2)Reweight observable to different state and calculate observable based on new state
	- smarty move in many parameters
        - will have to start without torsion moves
        - Safe moves in equilibrium bond length and angle is ~3%. For force constants ~5%.
    3)Will have to make decision here:
        a)Continue to sample in order to gather more data? -or-
        b)Attempt to create surrogate models from the data we have? What does that entail?
            i)We want a surrogate model for every observable we have, $O\left(\theta\right)$
            ii)Thus for bonds and angles; we have 4 observables as a function of however many parameters we're working with at the time
            iii)Choice of surrogate model becomes important. Start with splining though
            iv)What is the best surrogate modeling technique to use when we have very sparse data?
    4)
   
            
          

    Other things to consider:
    1)Choice of surrogate models:
        a)Splining
        b)Rich's ideas
        c)Other ideas from Michael he got at conference last week
    2)Choice of likelihood:
        a)Gaussian likelihood
        b)More general based on mean squared error
    3)Prior
        a)Start with uniforms with physically relevant bounds for given parameter
        b)Informationless priors
  
    Expanding initial knowledge region using MBAR
    1) Simulate single thermodynamic state
    2) Use MBAR to reweight in parameter space
        a) Will go to full extent of parameters within region where we know MBAR estimates are good
        b) Reweighting at multiple steps to full extent and along diagonals between parameters in order to create grid of points of evidence
        c) Now we cheaply achieved a region of evidence vs a single point
    3) Can fit our hypercube to multiple planes
        a) Assuming trends in very local space will be incredibly linear
        b) Probably a pretty safe assumption given minute change in parameter
    """
    # Begin process by loading a prescribed simulation or performing it if it doesn't exist in the specified directory
    
    mu_current = mu_init
    posterior = [mu_current]
    for i in range(samples):
        # suggest new position
        mu_proposal = norm(mu_current, proposal_width).rvs()

        # Compute likelihood by multiplying probabilities of each data point
        likelihood_current = norm(mu_current, 1).pdf(data).prod()
        likelihood_proposal = norm(mu_proposal, 1).pdf(data).prod()
        
        # Compute prior probability of current and proposed mu        
        prior_current = norm(mu_prior_mu, mu_prior_sd).pdf(mu_current)
        prior_proposal = norm(mu_prior_mu, mu_prior_sd).pdf(mu_proposal)
        
        p_current = likelihood_current * prior_current
        p_proposal = likelihood_proposal * prior_proposal
        
        # Accept proposal?
        p_accept = p_proposal / p_current
        
        # Usually would include prior probability, which we neglect here for simplicity
        accept = np.random.rand() < p_accept
        
        if plot:
            plot_proposal(mu_current, mu_proposal, mu_prior_mu, mu_prior_sd, data, accept, posterior, i)
        
        if accept:
            # Update position
            mu_current = mu_proposal
        
        posterior.append(mu_current)
        
    return posterior
#------------------------------------------------------------------
# Function to display
#------------------------------------------------------------------

def plot_proposal(mu_current, mu_proposal, mu_prior_mu, mu_prior_sd, data, accepted, trace, i):
    from copy import copy
    trace = copy(trace)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(16, 4))
    fig.suptitle('Iteration %i' % (i + 1))
    x = np.linspace(-3, 3, 5000)
    color = 'g' if accepted else 'r'
        
    # Plot prior
    prior_current = norm(mu_prior_mu, mu_prior_sd).pdf(mu_current)
    prior_proposal = norm(mu_prior_mu, mu_prior_sd).pdf(mu_proposal)
    prior = norm(mu_prior_mu, mu_prior_sd).pdf(x)
    ax1.plot(x, prior)
    ax1.plot([mu_current] * 2, [0, prior_current], marker='o', color='b')
    ax1.plot([mu_proposal] * 2, [0, prior_proposal], marker='o', color=color)
    ax1.annotate("", xy=(mu_proposal, 0.2), xytext=(mu_current, 0.2),
                 arrowprops=dict(arrowstyle="->", lw=2.))
    ax1.set(ylabel='Probability Density', title='current: prior(mu=%.2f) = %.2f\nproposal: prior(mu=%.2f) = %.2f' % (mu_current, prior_current, mu_proposal, prior_proposal))
    
    # Likelihood
    likelihood_current = norm(mu_current, 1).pdf(data).prod()
    likelihood_proposal = norm(mu_proposal, 1).pdf(data).prod()
    y = norm(loc=mu_proposal, scale=1).pdf(x)
    sns.distplot(data, kde=False, norm_hist=True, ax=ax2)
    ax2.plot(x, y, color=color)
    ax2.axvline(mu_current, color='b', linestyle='--', label='mu_current')
    ax2.axvline(mu_proposal, color=color, linestyle='--', label='mu_proposal')
    #ax2.title('Proposal {}'.format('accepted' if accepted else 'rejected'))
    ax2.annotate("", xy=(mu_proposal, 0.2), xytext=(mu_current, 0.2),
                 arrowprops=dict(arrowstyle="->", lw=2.))
    ax2.set(title='likelihood(mu=%.2f) = %.2f\nlikelihood(mu=%.2f) = %.2f' % (mu_current, 1e14*likelihood_current, mu_proposal, 1e14*likelihood_proposal))
    
    # Posterior
    posterior_analytical = calc_posterior_analytical(data, x, mu_prior_mu, mu_prior_sd)
    ax3.plot(x, posterior_analytical)
    posterior_current = calc_posterior_analytical(data, mu_current, mu_prior_mu, mu_prior_sd)
    posterior_proposal = calc_posterior_analytical(data, mu_proposal, mu_prior_mu, mu_prior_sd)
    ax3.plot([mu_current] * 2, [0, posterior_current], marker='o', color='b')
    ax3.plot([mu_proposal] * 2, [0, posterior_proposal], marker='o', color=color)
    ax3.annotate("", xy=(mu_proposal, 0.2), xytext=(mu_current, 0.2),
                 arrowprops=dict(arrowstyle="->", lw=2.))
    #x3.set(title=r'prior x likelihood $\propto$ posterior')
    ax3.set(title='posterior(mu=%.2f) = %.5f\nposterior(mu=%.2f) = %.5f' % (mu_current, posterior_current, mu_proposal, posterior_proposal))
    
    if accepted:
        trace.append(mu_proposal)
    else:
        trace.append(mu_current)
    ax4.plot(trace)
    ax4.set(xlabel='iteration', ylabel='mu', title='trace')
    plt.tight_layout()
    #plt.legend()
    
np.random.seed(123)

posterior = sampler(data, samples=8000, mu_init=2.)
fig, ax = plt.subplots()
ax.plot(posterior)
_ = ax.set(xlabel='sample', ylabel='mu');

sys.exit()
ax1 = plt.subplot()

sns.distplot(posterior[500:], ax=ax1, label='estimated posterior')
x = np.linspace(-.5, .5, 500)
post = calc_posterior_analytical(data, x, 0, 1)
ax1.plot(x, post, 'g', label='analytic posterior')
_ = ax1.set(xlabel='mu', ylabel='belief');
ax1.legend();


#Each piece of experimental data that we use as evidence (i.e. bond length for 
#one smirks vs any other or variance of a specific bond angle, etc.) has a 
#separate likelihood to evaluate and the product of all of those likelihoods
#the overall likelihood for a particular proposal. So we propose a change in 
#all parameters in one go and then evaluate that full change with all of the 
#evidence we have available.
