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
from simtk import unit 
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
#read_or_run(optparam = {'AlkEthOH_c1143':[['[#6X4:1]-[#1:2]','k','650'],
#                                          ['[#6X4:1]-[#1:2]','length','1.1'],
#                                          ['[#6X4:1]-[#6X4:2]-[#8X2H1:3]-[#1:4]','k1','0.2']]})

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

    niterations = len(xyz)#.shape[0] # no. of frames
    
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

def new_param_energy(coords, params):
    """
    Return potential energies associated with specified parameter perturbations.
    Parameters
    ----------
    coords: coordinates from the simulation(s) ran on the given molecule
    params:  arbitrary length dictionary of changes in parameter across arbitrary number of states. Highest level key is the molecule AlkEthOH_ID,
             second level of keys are the new state, the values of each of these subkeys are a arbitrary length list of length 3 lists where the 
             length 3 lists contain information on a parameter to change in the form: [SMIRKS, parameter type, parameter value]. I.e. :
    
             params = {'AlkEthOH_c1143':{'State 1':[['[6X4:1]-[#1:2]','k','620'],['[6X4:1]-[#6X4:2]','length','1.53'],...],'State 2':[...],...}}
    Returns
    -------
    energies: a list of the energies associated with the forcfield parameters used as input
    """
    #-------------------
    # PARAMETERS
    #-------------------
    params = params
   
    # Determine number of states we wish to estimate potential energies for
    mols = []
    for i in params: 
        mols.append(i)
    mol = 'Mol2_files/'+mols[0]+'.mol2' 
    K = len(params[mols[0]].keys())
    

    #if np.shape(params) != np.shape(N_k): raise "K_k and N_k must have same dimensions"


    # Determine max number of samples to be drawn from any state

    #-------------
    # SYSTEM SETUP
    #-------------
    verbose = False # suppress echos from OEtoolkit functions
    ifs = oechem.oemolistream(mol)
    mol = oechem.OEMol()
    # This uses parm@frosst atom types, so make sure to use the forcefield-flavor reader
    flavor = oechem.OEIFlavor_Generic_Default | oechem.OEIFlavor_MOL2_Default | oechem.OEIFlavor_MOL2_Forcefield
    ifs.SetFlavor( oechem.OEFormat_MOL2, flavor)
    oechem.OEReadMolecule(ifs, mol )
    # Perceive tripos types
    oechem.OETriposAtomNames(mol)
 
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
    
    energies = np.zeros([K,len(coords)],np.float64)
    for i,j in enumerate(params):
        AlkEthOH_id = j    
        for k,l in enumerate(params[AlkEthOH_id]):
            for m,n in enumerate(params[AlkEthOH_id][l]):
                newparams = ff.getParameter(smirks=n[0])
                newparams[n[1]]=n[2]
                ff.setParameter(newparams,smirks=n[0]) 
                system = ff.createSystem(topology, [mol])
                for o,p in enumerate(positions):
                    e = np.float(get_energy(system,p))
                    energies[m,o] = e
       # temp0 = np.zeros([len(params),samps],np.float64)
       # param = ff.getParameter(smirks=s)
       # for ind,val in enumerate(params):
       #     for p in paramtype:
       #         temp1 = np.zeros(samps,np.float64)
       #         for a,b in zip(val,p):
       #             param[b] = str(a)      
       #         ff.setParameter(param, smirks = s)
       #         system = ff.createSystem(topology, [mol], verbose=verbose)
       #         for i,a in enumerate(xyzn): 
       #             e = np.float(get_energy(system, a))
       #             energies[inds,ind,i] = e
   
    return energies,system

#------------------------------------------------------------------
ncfiles = glob.glob('*.nc')
mol2 = 'Mol2_files/AlkEthOH_c1143'
positions = []
for i,j in enumerate(ncfiles):
    data, xyz = read_traj(j)
    for pos in xyz:
       positions.append(pos)
#for i,j in enumerate(positions):
#    print j
#    if i > 1:
#        break 
print len(positions)
params = {'AlkEthOH_c1143':{'State 1':[['[#6X4:1]-[#1:2]','k','620'],['[#6X4:1]-[#6X4:2]','length','1.53']],
                            'State 2':[['[#6X4:1]-[#1:2]','k','680'],['[#6X4:1]-[#6X4:2]','length','1.60']]}}

energies,system = new_param_energy(positions, params)

#------------------------------------------------------------------

def get_small_mol_dict(mol2):
    """
    Return dictionary specifying the bond, angle and torsion indices to feed to ComputeBondsAnglesTorsions()
    Parameters
    ----------
    mol2: mol2 file associated with molecule of interest used to determine atom labels
         
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
    for fname in mol2:
        MoleculeName = fname.split('/')[-1].rsplit('.')[0]
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

AtomDict,lst_0,lst_1,lst_2 = get_small_mol_dict(['Mol2_files/AlkEthOH_c1143.mol2'])

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
     
    if (N_k_sub == N_k).all():
        ts_sub = ts
        xyz_sub = xyz
        print "No sub-sampling occurred"
    else:
        print "Sub-sampling..." 
        ts_sub = np.array([t[timeseries.subsampleCorrelatedData(t,g=b)] for t,b in zip(ts,g)])
        #for c in xyz:
        #    xyz_sub = [c[timeseries.subsampleCorrelatedData(t,g=b)] for t,b in zip(ts,g)]
        for i,j in enumerate(xyz):
            xyz_sub = [j[ii] for ii in ind[i]]
            
    return ts_sub, N_k_sub, xyz_sub, ind

ts_sub,N_k_sub,xyz_sub,ind = subsampletimeseries([energies[0],energies[1]],[positions,positions],[10000,10000])
obs = ComputeBondsAnglesTorsions(positions,AtomDict['Bond'],AtomDict['Angle'],AtomDict['Torsion'])[0]
print [bond[0] for bond in obs]
sys.exit()
#------------------------------------------------------------------
def calc_u_kn(xyzn,params,T=300.,):
    """
    Given a kxN matrix of coordinates, calculate full u_kn matrix for arbitrary number of new states.
    Where k is the original number of states and N is the length of each trajectory.
    -------------
    xyzn - kxN matrix of coordinates representing all configurations visited from the original states
    params - Arbitrary length dictionary of changes in parameter across arbitrary number of states. Highest level key is the molecule AlkEthOH_ID,
             second level of keys are the new state, the values of each of these subkeys are a arbitrary length list of length 3 lists where the 
             length 3 lists contain information on a parameter to change in the form: [SMIRKS, parameter type, parameter value]. I.e. :
    
             params = {'AlkEthOH_c100':{'State 1':[['[6X4:1]-[#1:2]','k','620'],['[6X4:1]-[#6X4:2]','length','1.53'],...],'State 2':[...],...},'AlkEthOH_c1143':                       {...},...}
    
    Returns
    -------------
    
    """
    # Step 1: Calculate the energies of the original configurations so that we can subsample
    
    
    N_k = len(xyzn)
    beta = 1/(kB*T)
    for i in params:
        print i

    return


N_k = 5000
files = ['AlkEthOH_c1143_[#6X4:1]-[#6X4:2]-[#8X2H1:3]-[#1:4]_k1_0.08.nc','AlkEthOH_c1143_[#6X4:1]-[#6X4:2]-[#8X2H1:3]-[#1:4]_k1_0.09.nc']
xyzn_tot = []
count = 0
for i in files:
    data, xyzn = read_traj(i)
    for i in xyzn:
        xyzn_tot.append(i)
        count += 1
        if count == N_k*len(files):
            break
print xyzn_tot        
print len(xyzn_tot)
g = timeseries.statisticalInefficiency(xyzn_tot) 
xyz_sub = [xyzn_tot[timeseries.subsampleCorrelatedData(xyzn_tot,g)]]

sys.exit()
    
# MAIN

#-----------------------------------------------------------------
# PARAMETERS
#-----------------------------------------------------------------
#N_k = np.array([100, 100, 100, 100, 100])
#N_k_orig = 10000.

indkeep = 4000
N_k= np.array([4000])

K = np.size(N_k)

N_max = np.max(N_k)

K_extra_vals = np.arange(0.08,0.25,0.01)

#K_k = np.array([[106], [104], [102], [100], [98]])

K_k = np.array([[[1.090]] for val in K_extra_vals])

#K_extra = np.array([[96], [99], [103], [105], [108]]) # unsampled force constants
K_extra = np.array([[[val]] for val in K_extra_vals])

paramtype = [['k1']]
obstype = 'Torsion'

#mol2 = [['molecules/AlkEthOH_r0.mol2'],['molecules/AlkEthOH_r48.mol2'],['molecules/AlkEthOH_r51.mol2'],['molecules/AlkEthOH_c581.mol2'],['molecules/AlkEthOH_c100.mol2'],['molecules/AlkEthOH_c1161.mol2'],['molecules/AlkEthOH_c1266.mol2'],['molecules/AlkEthOH_c38.mol2'],['molecules/AlkEthOH_r118.mol2'],['molecules/AlkEthOH_r12.mol2']]
#mol2 = [['molecules/AlkEthOH_r0.mol2'],['molecules/AlkEthOH_c581.mol2'],['molecules/AlkEthOH_c100.mol2'],['molecules/AlkEthOH_c1266.mol2'],['molecules/AlkEthOH_r51.mol2'],['molecules/AlkEthOH_r48.mol2']]

mol2 = [['Mol2_files/'+sys.argv[1]+'.mol2']]

#mol2en = ['molecules/AlkEthOH_r0.mol2','molecules/AlkEthOH_r48.mol2','molecules/AlkEthOH_r51.mol2','molecules/AlkEthOH_c581.mol2','molecules/AlkEthOH_c100.mol2','molecules/AlkEthOH_c1161.mol2','molecules/AlkEthOH_c1266.mol2','molecules/AlkEthOH_c38.mol2','molecules/AlkEthOH_r118.mol2','molecules/AlkEthOH_r12.mol2']
mol2en = [val[0] for val in mol2]

#traj = ['traj/AlkEthOH_r0.nc','traj/AlkEthOH_r48.nc','traj/AlkEthOH_r51.nc','traj/AlkEthOH_c581.nc','traj/AlkEthOH_c100.nc','traj/AlkEthOH_c1161.nc','traj/AlkEthOH_c1266.nc','traj/AlkEthOH_c38.nc','traj/AlkEthOH_r118.nc','traj/AlkEthOH_r12.nc']
#traj = ['traj/AlkEthOH_r0.nc','traj/AlkEthOH_c581.nc','traj/AlkEthOH_c100.nc','traj/AlkEthOH_c1266.nc','traj/AlkEthOH_r51.nc','traj/AlkEthOH_r48.nc']
traj = ['traj4ns/'+sys.argv[1]+'.nc']


#trajs = [['traj/AlkEthOH_r0.nc'],['traj/AlkEthOH_r48.nc'],['traj/AlkEthOH_r51.nc'],['traj/AlkEthOH_c581.nc'],['traj/AlkEthOH_c100.nc'],['traj/AlkEthOH_c1161.nc'],['traj/AlkEthOH_c1266.nc'],['traj/AlkEthOH_c38.nc'],['traj/AlkEthOH_r118.nc'],['traj/AlkEthOH_r12.nc']] 
#trajs = [['traj/AlkEthOH_r0.nc'],['traj/AlkEthOH_c581.nc']]
trajs = [[val] for val in traj]

smirkss = ['[#6X4:1]-[#6X4:2]-[#8X2H1:3]-[#1:4]']

trajstest = [[[] for i in K_extra] for _ in traj]
for ind,val in enumerate(trajs):
    for ind1,val1 in enumerate(K_extra):
        trajstest[ind][ind1] = [val[0][:-3]+'_'+smirkss[0]+'_k1'+str(val1[0][0])+'.nc']

# Create lists to store data that will eventually be written in pandas df and saved as csv/json/pkl
molnamedf = []
smirksdf = []
obstypedf = []
paramtypedf = []
newparamval = []
N_subsampleddf = []
percentshiftdf = []
E_expectdf = [] 
dE_expectdf = []
dE_bootdf = []
Enew_expectdf = [] 
dEnew_expectdf = []
dEnew_bootdf = []
A_expectdf = [] 
dA_expectdf = []
dA_bootdf = []
Anew_sampleddf = []
Anew_expectdf = [] 
dAnew_expectdf = []
dAnew_bootdf = []
varAnew_bootdf = []
varAnew_sampdf = []
altvarAnew_bootdf = []
dvarAnew_bootdf = []
altdvarAnew_bootdf = []
varAnew_bootdf2 = []
altvarAnew_bootdf2 = []
dvarAnew_bootdf2 = []
altdvarAnew_bootdf2 = []
A_boot_new_sampdf = []
dA_boot_new_sampdf = []
# Return AtomDict needed to feed to ComputeBondsAnglesTorsions()
for ind,(i,j) in enumerate(zip(mol2,traj)):
    AtomDict,lst_0,lst_1,lst_2 = get_small_mol_dict(i, [j]) 
    mylist = [ii[1] for ii in lst_2[0]] 
    myset = set(mylist)
    poplist = np.zeros([len(myset)],np.float64) 
    for b,k in enumerate(myset):
        print "%s occurs %s times" %(k, mylist.count(k))
        poplist[b] = mylist.count(k)
    pctlist = 100.*poplist/sum(poplist)
    pctdict = dict()
    for c,k in enumerate(myset):
        pctdict[k] = pctlist[c]   
    
    print '#################################################################################'
    Atomdictmatches = []
    for sublist in lst_2[0]:    
        if sublist[1] == smirkss[0]:
            Atomdictmatches.append(sublist[0])  
    if not Atomdictmatches:
        print 'No matches found'
        continue 
    
    Atomdictmatchinds = []
    for yy in Atomdictmatches:
        for z,y in enumerate(AtomDict[obstype]):
            if yy == str(AtomDict[obstype][z]):
                Atomdictmatchinds.append(z)
    
    obs_ind = Atomdictmatchinds[0]
    
    # Calculate energies at various parameters of interest
    for indparam,valparam in enumerate(K_extra):
        energies, xyzn, system = new_param_energy(mol2en[ind],j, smirkss, N_k, K_k[indparam], paramtype, N_max, indkeep)
        energiesnew, xyznnew, systemnew = new_param_energy(mol2en[ind],j, smirkss, N_k, K_extra[indparam], paramtype, N_max, indkeep)
         
        xyznsampled = [[] for i in trajs[ind]]
        A = np.zeros([K,N_max],np.float64)
        for i,x in enumerate(trajs[ind]):
            coord = readtraj(x,indkeep)[1]
            xyznsampled[i] = coord   
            obs = ComputeBondsAnglesTorsions(coord,AtomDict['Bond'],AtomDict['Angle'],AtomDict['Torsion'])[0]# Compute angles and return array of angles
            numatom = len(obs[0]) # get number of unique angles in molecule
            timeser = [obs[:,d] for d in range(numatom)] # re-organize data into timeseries
            A[i] = timeser[obs_ind] # pull out single angle in molecule for test case  
      
        xyznnewtest = [[] for i in trajstest[ind][indparam]] 
        Anewtest = np.zeros([K,N_max],np.float64)
        for i,x in enumerate(trajstest[ind][indparam]):
            coordtest = readtraj(x,indkeep)[1]
            xyznnewtest[i] = coordtest   
            obstest = ComputeBondsAnglesTorsions(coordtest,AtomDict['Bond'],AtomDict['Angle'],AtomDict['Torsion'])[0]# Compute angles and return array of angles
            numatomtest = len(obstest[0]) # get number of unique angles in molecule
            timesertest = [obstest[:,d] for d in range(numatomtest)] # re-organize data into timeseries
            Anewtest[i] = timesertest[obs_ind] # pull out single angle in molecule for test case

              
        # Subsample timeseries and return new number of samples per state
        A_sub, N_kA, xyzn_A_sub, indA  = subsampletimeseries(A, xyznsampled, N_k)
        En_sub, N_kEn, xyzn_En_sub, indEn = subsampletimeseries(energies[0], xyznsampled, N_k) 
        Ennew_sub, N_kEnnew, xyzn_Ennew_sub, indEnnew = subsampletimeseries(energiesnew[0], xyznsampled, N_k)
        A_sub_test,N_kA_test,xyzn_A_test,indAtest = subsampletimeseries(Anewtest,xyznnewtest,N_k)
        
        for a,b,c,d in zip(N_kA,N_kEn,N_kEnnew,N_kA_test):
            N_kF = np.array([min(a,b,c,d)])
           
        A_kn = np.zeros([sum(N_kF)],np.float64)
        A_knnew = np.zeros([sum(N_kF)],np.float64)
        count = 0
        for x1,x2 in zip(A_sub,A_sub_test):
            for y1,y2 in zip(x1,x2):
                A_kn[count] = y1
                A_knnew[count] = y2
                count += 1
                if count > (sum(N_kF)-1):
                    break 
        #--------------------------------------------------------------
        # Re-evaluate potenitals at all subsampled coord and parameters
        #--------------------------------------------------------------
        verbose = False # suppress echos from OEtoolkit functions
        ifs = oechem.oemolistream(mol2en[ind])
        mol = oechem.OEMol()
        # This uses parm@frosst atom types, so make sure to use the forcefield-flavor reader
        flavor = oechem.OEIFlavor_Generic_Default | oechem.OEIFlavor_MOL2_Default | oechem.OEIFlavor_MOL2_Forcefield
        ifs.SetFlavor( oechem.OEFormat_MOL2, flavor)
        oechem.OEReadMolecule(ifs, mol )
        # Perceive tripos types
        oechem.OETriposAtomNames(mol)

        # Load forcefield file
        ffxml = get_data_filename('forcefield/smirff99Frosst.ffxml')
        ff = ForceField(ffxml)

        # Generate a topology
        from smarty.forcefield import generateTopologyFromOEMol
        topology = generateTopologyFromOEMol(mol)


        # Re-calculate energies     
        E_kn = np.zeros([len(K_k[indparam]),sum(N_kEn)],np.float64)
        for inds,s in enumerate(smirkss):
            param = ff.getParameter(smirks=s)
            for indss,vals in enumerate(K_k[indparam]):
                count = 0
                for p in paramtype:
                    for a,b in zip(vals,p):
                        param[b] = str(a)
                    ff.setParameter(param, smirks = s)
                    system = ff.createSystem(topology, [mol], verbose=verbose)  
                    while count < sum(N_kEn):
                        for k_ind, pos in enumerate(xyzn_En_sub):
                            for i,a in enumerate(pos):
                                e = np.float(get_energy(system, a))
                                E_kn[indss,count] = e
                                count += 1 
                             
        E_knnew = np.zeros([len(K_extra[indparam]),sum(N_kEn)],np.float64)
        for inds,s in enumerate(smirkss):
            param = ff.getParameter(smirks=s)
            for indss,vals in enumerate(K_extra[indparam]):
                count = 0
                for p in paramtype:
                    for a,b in zip(vals,p):    
                        param[b] = str(a)
                    ff.setParameter(param, smirks = s)
                    system = ff.createSystem(topology, [mol], verbose=verbose)  
                    while count < sum(N_kEn):
                        for k_ind, pos in enumerate(xyzn_En_sub):
                            for i,a in enumerate(pos):
                                e = np.float(get_energy(system, a))
                                E_knnew[indss,count] = e
                                count += 1
               
        # Post process energy distributions to find expectation values, analytical uncertainties and bootstrapped uncertainties
        #T_from_file = read_col('StateData/data.csv',["Temperature (K)"],100)
        Temp_k = 300.#T_from_file
        T_av = 300.#np.average(Temp_k)

        nBoots = 200

        beta_k = 1 / (kB*T_av)
        bbeta_k = 1 / (kB*Temp_k)

        #################################################################
        # Compute reduced potentials
        #################################################################

        print "--Computing reduced potentials..."

        # Initialize matrices for u_kn/observables matrices and expected value/uncertainty matrices
        u_kn = np.zeros([K, sum(N_kF)], dtype=np.float64)
        E_kn_samp = np.zeros([K,sum(N_kF)],np.float64)
        u_knnew = np.zeros([K,sum(N_kF)], np.float64)
        E_knnew_samp = np.zeros([K,sum(N_kF)], np.float64)
        A_kn_samp = np.zeros([sum(N_kF)],np.float64)
        A_knnew_samp = np.zeros([sum(N_kF)],np.float64)
        A2_kn = np.zeros([sum(N_kF)],np.float64)
        A2_knnew = np.zeros([sum(N_kF)],np.float64)


        nBoots_work = nBoots + 1

        allE_expect = np.zeros([K,nBoots_work], np.float64)
        allA_expect = np.zeros([K,nBoots_work],np.float64)
        allE2_expect = np.zeros([K,nBoots_work], np.float64)
        dE_expect = np.zeros([K], np.float64)
        allE_expectnew = np.zeros([K,nBoots_work], np.float64)
        allE2_expectnew = np.zeros([K,nBoots_work], np.float64)
        dE_expectnew = np.zeros([K], np.float64)
        dA_expect = np.zeros([K],np.float64)
        dA_expectnew = np.zeros([K],np.float64)
        allvarA_expect_samp = np.zeros([K,nBoots_work],np.float64)
        allA_expectnew = np.zeros([K,nBoots_work],np.float64)
        allvarA_expectnew = np.zeros([K,nBoots_work],np.float64)
        allaltvarA_expectnew = np.zeros([K,nBoots_work],np.float64)
        allA_new_mean_samp = np.zeros([nBoots_work],np.float64)
        
        # Begin bootstrapping loop
        for n in range(nBoots_work):
            if (n > 0):
                print "Bootstrap: %d/%d" % (n,nBoots)
            for k in range(K):        
                if N_kF[k] > 0:
	            if (n == 0):
                        booti = np.array(range(N_kF[k]))
	            else:
	                booti = np.random.randint(N_kF[k], size = N_kF[k])
           
                    E_kn_samp[:,sum(N_kF[0:k]):sum(N_kF[0:k+1])] = E_kn[:,booti]
                    E_knnew_samp[:,sum(N_kF[0:k]):sum(N_kF[0:k+1])] = E_knnew[:,booti]
                    A_kn_samp[sum(N_kF[0:k]):sum(N_kF[0:k+1])] = A_kn[booti] 
                    A_knnew_samp[sum(N_kF[0:k]):sum(N_kF[0:k+1])] = A_knnew[booti]
                    
            for k in range(K): 
                u_kn[:,sum(N_kF[0:k]):sum(N_kF[0:k+1])] = beta_k * E_kn_samp[:,sum(N_kF[0:k]):sum(N_kF[0:k+1])]     
                u_knnew[:,sum(N_kF[0:k]):sum(N_kF[0:k+1])] = beta_k * E_knnew_samp[:,sum(N_kF[0:k]):sum(N_kF[0:k+1])]
	        A2_kn[sum(N_kF[0:k]):sum(N_kF[0:k+1])] = A_kn_samp[sum(N_kF[0:k]):sum(N_kF[0:k+1])] 
                A2_knnew[sum(N_kF[0:k]):sum(N_kF[0:k+1])] = A_knnew_samp[sum(N_kF[0:k]):sum(N_kF[0:k+1])]

############################################################################
# Initialize MBAR
############################################################################

# Initialize MBAR with Newton-Raphson
# Use Adaptive Method (Both Newton-Raphson and Self-Consistent, testing which is better)
            if (n==0):
	        initial_f_k = None # start from zero 
            else:
	        initial_f_k = mbar.f_k # start from the previous final free energies to speed convergence
		
            mbar = pymbar.MBAR(u_kn, N_kF, verbose=False, relative_tolerance=1e-12,initial_f_k=initial_f_k)
            
                        
        #------------------------------------------------------------------------
        # Compute Expectations for energy and angle distributions
        #------------------------------------------------------------------------
   
           # print ""
           # print "Computing Expectations for E..."
            E_kn2 = u_kn  # not a copy, we are going to write over it, but we don't need it any more.
            E_knnew2 = u_knnew
            for k in range(K):
                E_kn2[k,:]*=beta_k**(-1)  # get the 'unreduced' potential -- we can't take differences of reduced potentials because the beta is different.
	        E_knnew2[k,:]*=beta_k**(-1)
            (E_expect, dE_expect) = mbar.computeExpectations(E_kn2,state_dependent = True)
            (E_expectnew, dE_expectnew) = mbar.computeExpectations(E_knnew2,state_dependent = True)
            (A_expect, dA_expect) = mbar.computeExpectations(A2_kn,state_dependent = False) 


            allE_expect[:,n] = E_expect[:]
            allE_expectnew[:,n] = E_expectnew[:]
            allA_expect[:,n] = A_expect[:]
    
        # expectations for the differences, which we need for numerical derivatives  
        # To be used once the energy expectations are fixed
            (DeltaE_expect, dDeltaE_expect) = mbar.computeExpectations(E_kn2,output='differences', state_dependent = False)
            (DeltaE_expectnew, dDeltaE_expectnew) = mbar.computeExpectations(E_knnew2,output='differences', state_dependent = False)

        # print "Computing Expectations for E^2..."
            (E2_expect, dE2_expect) = mbar.computeExpectations(E_kn2**2, state_dependent = True)
            allE2_expect[:,n] = E2_expect[:]

            (A_expectnew, dA_expectnew) = mbar.computeExpectations(A2_kn,u_knnew,state_dependent=False)
            allA_expectnew[:,n] = A_expectnew[:]
            
            
	#Variance in unsampled calculated observables using MBAR
            varA_mbar_feed = np.zeros([sum(N_kF)],np.float64)
            for l in range(sum(N_kF)):
                varA_mbar_feed[l] = ((A2_kn[l] - A_expect)**2)
            
            (varA_expectnew,dvarA_expectnew) = mbar.computeExpectations(varA_mbar_feed,u_knnew,state_dependent=False)
            
            allvarA_expectnew[:,n] = varA_expectnew[:]

            #Check against calculating variance of A as <x^2> - <A>^2 (instead of <(x-A)^2>)
            (A2_expectnew,dA2_expectnew) = mbar.computeExpectations(A2_kn**2,u_knnew,state_dependent=False)
            altvarA_expectnew = (A2_expectnew[:] - A_expectnew[:]**2)
               
            allaltvarA_expectnew[:,n] = altvarA_expectnew[:]
            
            #Record mean of sampled observable with bootstrap randomization to get error bars 
            allA_new_mean_samp[n] = np.mean(A2_knnew)
            
            N_eff = mbar.computeEffectiveSampleNumber(verbose = True)

        if nBoots > 0:
            A_bootnew = np.zeros([K],dtype=np.float64)
            E_bootnew = np.zeros([K],dtype=np.float64)
            dE_boot = np.zeros([K],dtype=np.float64)
            dE_bootnew = np.zeros([K],dtype=np.float64)
            dA_boot = np.zeros([K],dtype=np.float64)
            dA_bootnew = np.zeros([K],dtype=np.float64)
            varA_bootnew = np.zeros([K],dtype=np.float64)
            altvarA_bootnew = np.zeros([K],dtype=np.float64)
            dvarA_bootnew = np.zeros([K],dtype=np.float64)
            altdvarA_bootnew = np.zeros([K],dtype=np.float64)
            A_bootnew_samp = np.mean(allA_new_mean_samp)
            dA_bootnew_samp = np.std(allA_new_mean_samp)
            for k in range(K):
    	        dE_boot[k] = np.std(allE_expect[k,1:nBoots_work])
                dE_bootnew[k] = np.std(allE_expectnew[k,1:nBoots_work])
                dA_boot[k] = np.std(allA_expect[k,1:nBoots_work])
                dA_bootnew[k] = np.std(allA_expectnew[k,1:nBoots_work])
                varA_bootnew[k] = np.average(allvarA_expectnew[k,1:nBoots_work])
                altvarA_bootnew[k] = np.average(allaltvarA_expectnew[k,1:nBoots_work])
                dvarA_bootnew[k] = np.std(allvarA_expectnew[k,1:nBoots_work])
                altdvarA_bootnew[k] = np.std(allaltvarA_expectnew[k,1:nBoots_work])
             
            dA_bootnew = dA_expectnew
            varA_bootnew = varA_expectnew
            dvarA_bootnew = dvarA_expectnew
            altvarA_bootnew = altvarA_expectnew
            #altdvarA_bootnew = altdvarA_expectnew
            
            #bins1 = int(np.log2(len(allA_expectnew[0])))
            #bins2 = int(np.sqrt(len(allA_expectnew[0])))
            #binsnum = int((bins1+bins2)/2)
            
            #plt.figure() 
            #plt.hist(allA_expectnew[0], binsnum, normed=1, facecolor='green', alpha=0.75)
            #plt.xlabel('Length (A)')
            #plt.ylabel('Probability')
            #plt.axis([min(allA_expectnew[0])-(bins[1]-bins[0]), max(allA_expectnew[0])-(bins[1]-bins[0]), 0, bins[1]-bins[0]])
            #plt.grid(True)
            #plt.savefig('checkdist.png')
         
            E_mean_samp = np.array([np.average(E) for E in E_kn])
            E_mean_unsamp = np.array([np.average(E) for E in E_knnew]) 
            A_mean_samp = np.array([np.average(A) for A in A_sub])
            A_mean_test = np.array([np.average(A) for A in A_sub_test])
            varAnew_samp = np.array([np.std(A)**2 for A in A_sub_test])
            
            E_expect_mean = np.zeros([K],dtype=np.float64)       
            E_expect_meannew = np.zeros([K],dtype=np.float64)
            A_expect_mean_samp = np.zeros([K],dtype=np.float64)
            A_expect_mean_unsamp = np.zeros([K],dtype=np.float64)
            for k in range(K):
                E_expect_mean[k] = np.average(allE_expect[k,1:nBoots_work])
                E_expect_meannew[k] = np.average(allE_expectnew[k,1:nBoots_work])
                A_expect_mean_samp[k] = np.average(allA_expect[k,1:nBoots_work])
                A_expect_mean_unsamp[k] = np.average(allA_expectnew[k,1:nBoots_work])   
            
            A_expect_mean_unsamp = A_expectnew
            E_expect_meannew = E_expectnew
           
                        
            pctshft = 100.*((np.float(K_k[indparam]) - np.float(K_extra[indparam]))/np.float(K_k[indparam]))


                    
        allvarA_expectnew2 = np.zeros([K,nBoots_work],np.float64)
        allaltvarA_expectnew2 = np.zeros([K,nBoots_work],np.float64)
        for n in range(nBoots_work):
            if (n > 0):
                print "Bootstrap: %d/%d" % (n,nBoots)
            for k in range(K):        
                if N_kF[k] > 0:
	            if (n == 0):
                        booti = np.array(range(N_kF[k]))
	            else:
	                booti = np.random.randint(N_kF[k], size = N_kF[k])
           
                    E_kn_samp[:,sum(N_kF[0:k]):sum(N_kF[0:k+1])] = E_kn[:,booti]
                    E_knnew_samp[:,sum(N_kF[0:k]):sum(N_kF[0:k+1])] = E_knnew[:,booti]
                    A_kn_samp[sum(N_kF[0:k]):sum(N_kF[0:k+1])] = A_kn[booti] 
                    A_knnew_samp[sum(N_kF[0:k]):sum(N_kF[0:k+1])] = A_knnew[booti]
       
            for k in range(K): 
                u_kn[:,sum(N_kF[0:k]):sum(N_kF[0:k+1])] = beta_k * E_kn_samp[:,sum(N_kF[0:k]):sum(N_kF[0:k+1])]     
                u_knnew[:,sum(N_kF[0:k]):sum(N_kF[0:k+1])] = beta_k * E_knnew_samp[:,sum(N_kF[0:k]):sum(N_kF[0:k+1])]
	        A2_kn[sum(N_kF[0:k]):sum(N_kF[0:k+1])] = A_kn_samp[sum(N_kF[0:k]):sum(N_kF[0:k+1])] 
                A2_knnew[sum(N_kF[0:k]):sum(N_kF[0:k+1])] = A_knnew_samp[sum(N_kF[0:k]):sum(N_kF[0:k+1])]
                
############################################################################
# Initialize MBAR
############################################################################

# Initialize MBAR with Newton-Raphson
# Use Adaptive Method (Both Newton-Raphson and Self-Consistent, testing which is better)
            if (n==0):
	        initial_f_k = None # start from zero 
            else:
	        initial_f_k = mbar.f_k # start from the previous final free energies to speed convergence
		
            mbar = pymbar.MBAR(u_kn, N_kF, verbose=False, relative_tolerance=1e-12,initial_f_k=initial_f_k)

            #Variance in unsampled calculated observables using MBAR
            varA_mbar_feed2 = np.zeros([sum(N_kF)],np.float64)
            for l in range(sum(N_kF)):
                varA_mbar_feed2[l] = ((A2_kn[l] - A_expect_mean_unsamp[0])**2)
            
            
            (varA_expectnew2,dvarA_expectnew2) = mbar.computeExpectations(varA_mbar_feed2,u_knnew,state_dependent=False)
            
            allvarA_expectnew2[:,n] = varA_expectnew2[:]

            #Check against calculating variance of A as <x^2> - <A>^2 (instead of <(x-A)^2>)
            (A_expectnew2,dA_expectnew2) = mbar.computeExpectations(A2_kn,u_knnew,state_dependent=False)
            (A2_expectnew2,dA2_expectnew2) = mbar.computeExpectations(A2_kn**2,u_knnew,state_dependent=False)
            altvarA_expectnew2 = (A2_expectnew2[:] - A_expectnew2[:]**2)
               
            allaltvarA_expectnew2[:,n] = altvarA_expectnew2[:] 
            
        if nBoots > 0:
            varA_bootnew2 = np.zeros([K],dtype=np.float64)
            altvarA_bootnew2 = np.zeros([K],dtype=np.float64)
            dvarA_bootnew2 = np.zeros([K],dtype=np.float64)
            altdvarA_bootnew2 = np.zeros([K],dtype=np.float64)
            for k in range(K):
    	        varA_bootnew2[k] = np.average(allvarA_expectnew2[k,1:nBoots_work])
                altvarA_bootnew2[k] = np.average(allaltvarA_expectnew2[k,1:nBoots_work])
                dvarA_bootnew2 = np.std(allvarA_expectnew2[k,1:nBoots_work])
                altdvarA_bootnew2 = np.std(allaltvarA_expectnew2[k,1:nBoots_work])

            varA_bootnew2 = varA_expectnew2
            dvarA_bootnew2 = dvarA_expectnew2
            altvarA_bootnew2 = altvarA_expectnew
            #altdvarA_bootnew2 = altdvarA_expectnew2

            #print allvarA_expectnew2
            #print np.var(allA_expectnew) 
            bins1 = int(np.log2(len(allvarA_expectnew2[0])))
            bins2 = int(np.sqrt(len(allvarA_expectnew2[0])))
            binsnum = int((bins1+bins2)/2)           
            
            plt.figure()
            plt.hist(allvarA_expectnew2[0], binsnum, normed=1, facecolor='green', alpha=0.75)
            plt.xlabel('Length^2 (A^2)')
            plt.ylabel('Probability')
            #plt.axis([min(allA_expectnew[0])-(bins[1]-bins[0]), max(allA_expectnew[0])-(bins[1]-bins[0]), 0, bins[1]-bins[0]])
            plt.grid(True)
            plt.savefig('checkdist2.png')
            #sys.exit()



            print '###############################################################################'
            molnamedf.append(mol2[ind])
            smirksdf.append(smirkss[0])
            obstypedf.append(obstype)
            paramtypedf.append(paramtype[0])
            newparamval.append(K_extra[indparam])
            percentshiftdf.append(pctshft)
            N_subsampleddf.append(N_kF)
            E_expectdf.append(E_expect_mean)
            dE_expectdf.append(dE_expect)
            dE_bootdf.append(dE_boot)
            Enew_expectdf.append(E_expect_meannew)
            dEnew_expectdf.append(dE_expectnew)
            dEnew_bootdf.append(dE_bootnew)
            A_expectdf.append(A_expect_mean_samp)
            dA_expectdf.append(dA_expect)
            dA_bootdf.append(dA_boot)
            Anew_sampleddf.append(A_mean_test)
            Anew_expectdf.append(A_expect_mean_unsamp) 
            dAnew_expectdf.append(dA_expectnew)
            dAnew_bootdf.append(dA_bootnew)
            varAnew_sampdf.append(varAnew_samp)
            varAnew_bootdf.append(varA_bootnew)
            altvarAnew_bootdf.append(altvarA_bootnew)
            dvarAnew_bootdf.append(dvarA_bootnew)
            altdvarAnew_bootdf.append(altdvarA_bootnew)
            varAnew_bootdf2.append(varA_bootnew2)
            altvarAnew_bootdf2.append(altvarA_bootnew2)
            dvarAnew_bootdf2.append(dvarA_bootnew2)
            altdvarAnew_bootdf2.append(altdvarA_bootnew2)
            A_boot_new_sampdf.append(A_bootnew_samp)
            dA_boot_new_sampdf.append(dA_bootnew_samp)
            print("NEXT LOOP")
########################################################################
df = pd.DataFrame.from_dict({'mol_name':[value for value in molnamedf],
                             'smirks':[value for value in smirksdf],
                             'obs_type':[value for value in obstypedf],
                             'param_type':[value for value in paramtypedf],
                             'new_param':[value for value in newparamval],
                             'percent_shift':[value for value in percentshiftdf],
                             'N_subsampled':[value for value in N_subsampleddf],
                             'E_expect':[value for value in E_expectdf],
                             'dE_expect':[value for value in dE_expectdf],
                             'dE_boot':[value for value in dE_bootdf],
                             'Enew_expect':[value for value in Enew_expectdf],
                             'dEnew_expect':[value for value in dEnew_expectdf],
                             'dEnew_boot':[value for value in dEnew_bootdf],
                             'A_expect':[value for value in A_expectdf],
                             'dA_expect':[value for value in dA_expectdf],
                             'dA_boot':[value for value in dA_bootdf],
                             'Anew_sampled':[value for value in Anew_sampleddf],
                             'Anew_expect':[value for value in Anew_expectdf],
                             'dAnew_expect':[value for value in dAnew_expectdf],
                             'dAnew_boot':[value for value in dAnew_bootdf],
                             'varAnew_samp':[value for value in varAnew_sampdf],
                             'varAnew_boot':[value for value in varAnew_bootdf],
                             'altvarAnew_boot':[value for value in altvarAnew_bootdf],
                             'dvarAnew_boot':[value for value in dvarAnew_bootdf],
                             'altdvarAnew_boot':[value for value in altdvarAnew_bootdf],
                             'varAnew_boot2':[value for value in varAnew_bootdf2],
                             'altvarAnew_boot2':[value for value in altvarAnew_bootdf2],
                             'dvarAnew_boot2':[value for value in dvarAnew_bootdf2],
                             'altdvarAnew_boot2':[value for value in altdvarAnew_bootdf2],
                             'A_boot_new_samp':[value for value in A_boot_new_sampdf],
                             'dA_boot_new_samp':[value for value in dA_boot_new_sampdf]})
df.to_csv('mbar_analyses/mbar_analysis_'+sys.argv[1]+'_'+smirkss[0]+'_'+paramtype[0][0]+'_'+obstype+'.csv',sep=';')
df.to_pickle('mbar_analyses/mbar_analysis_'+sys.argv[1]+'_'+smirkss[0]+'_'+paramtype[0][0]+'_'+obstype+'.pkl')


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
