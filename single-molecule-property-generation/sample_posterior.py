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
import math
import scipy as sp
import seaborn as sns
import scipy.special
from scipy.stats import norm
from scipy.stats import multivariate_normal
import sys
from collections import OrderedDict
from itertools import combinations
from sklearn import *
from sklearn.preprocessing import PolynomialFeatures
import subprocess as sb

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
    xyzn = Quantity(xyz[-indkeep:], angstroms)

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
        traj_name = AlkEthOH_IDs[0]+'_'+filename_string+'.nc'
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
     
    integrator = openmm.VerletIntegrator(1.0 * femtoseconds)
    context = openmm.Context(system, integrator)
    context.setPositions(positions)
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy() / kilocalories_per_mole
    return energy

#------------------------------------------------------------------

def new_param_energy(coords, params, T=300.):
    """
    Return potential energies associated with specified parameter perturbations.
    Parameters
    ----------
    coords: coordinates from the simulation(s) ran on the given molecule
    params:  arbitrary length dictionary of changes in parameter across arbitrary number of states. Highest level key is the molecule AlkEthOH_ID,
             second level of keys are the new state, the values of each of these subkeys are a arbitrary length list of length 3 lists where the 
             length 3 lists contain information on a parameter to change in the form: [SMIRKS, parameter type, parameter value]. I.e. :
    
             params = {'AlkEthOH_c1143':{'State 1':[['[6X4:1]-[#1:2]','k','620'],['[6X4:1]-[#6X4:2]','length','1.53'],...],'State 2':[...],...}}
    T: Temperature of the system. By default set to 300 K.
    
    Returns
    -------
    E_kn: a kxN matrix of the dimensional energies associated with the forcfield parameters used as input
    u_kn: a kxN matrix of the dimensionless energies associated with the forcfield parameters used as input
    """

    #-------------------
    # CONSTANTS
    #-------------------
    beta = 1/(kB*T) 

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
    
    E_kn = np.zeros([K,len(coords)],np.float64)
    u_kn = np.zeros([K,len(coords)],np.float64)
    for i,j in enumerate(params):
        AlkEthOH_id = j    
        for k,l in enumerate(params[AlkEthOH_id]):
            for m,n in enumerate(params[AlkEthOH_id][l]):
                newparams = ff.getParameter(smirks=n[0])
                newparams[n[1]]=n[2]
                ff.setParameter(newparams,smirks=n[0]) 
                system = ff.createSystem(topology, [mol])
            for o,p in enumerate(coords):
                e = np.float(get_energy(system,p))
                E_kn[k,o] = e
                u_kn[k,o] = e*beta
              
         
    return E_kn,u_kn

#------------------------------------------------------------------
#ncfiles = glob.glob('*.nc')
#mol2 = 'Mol2_files/AlkEthOH_c1143'
#positions = []
#for i,j in enumerate(ncfiles):
#    data, xyz = read_traj(j)
#    for pos in xyz:
#       positions.append(pos)
#for i,j in enumerate(positions):
#    print j
#    if i > 1:
#        break 
#print len(positions)
#params = {'AlkEthOH_c1143':{'State 1':[['[#6X4:1]-[#1:2]','k','620'],['[#6X4:1]-[#6X4:2]','length','1.53']],
#                            'State 2':[['[#6X4:1]-[#1:2]','k','680'],['[#6X4:1]-[#6X4:2]','length','1.60']]}}

#energies,system = new_param_energy(positions, params)

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

#AtomDict,lst_0,lst_1,lst_2 = get_small_mol_dict(['Mol2_files/AlkEthOH_c1143.mol2'])

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

#ts_sub,N_k_sub,xyz_sub,ind = subsampletimeseries([energies[0],energies[1]],[positions,positions],[10000,10000])
#obs = ComputeBondsAnglesTorsions(positions,AtomDict['Bond'],AtomDict['Angle'],AtomDict['Torsion'])[0]
#----------------------------------------------------------------------

def find_smirks_instance(mol2,smirks):
    """
    Given a mol2 file and a smirks string, match the observable indices of the mol2 (determined by the mol2 connectivity)
    to the smirks string supplied.
    
    Parameters
    ----------
    mol2: the mol2 file for the molecule of interest
    smirks: the smirks string you wish to identify in the molecule
   
    Returns
    -------
    inds: a list of indices which correspond to the bonded observable identified by the smirks string supplied 
    """
    
    AtomDict, lst_0, lst_1, lst_2 = get_small_mol_dict([mol2])

    smirks_list = smirks.split('-')
    
    if len(smirks_list)==2:
        lst = lst_0
        obstype = 'Bond'
    
    if len(smirks_list)==3:
        lst = lst_1
        obstype = 'Angle'
   
    if len(smirks_list)==4:
        lst = lst_2
        obstype = 'Torsion'
 
    mylist = [i[1] for i in lst[0]] 
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
    for sublist in lst[0]:    
        if sublist[1] == smirks:
            Atomdictmatches.append(sublist[0])  
    if not Atomdictmatches:
        print 'No matches found'
        # continue 
    
    Atom_dict_match_inds = []
    for yy in Atomdictmatches:
        for z,y in enumerate(AtomDict[obstype]):
            if yy == str(AtomDict[obstype][z]):
                Atom_dict_match_inds.append(z)
    
        
    return Atom_dict_match_inds

#a = find_smirks_instance('Mol2_files/AlkEthOH_c1143.mol2','[#6X4:1]-[#1:2]')

#------------------------------------------------------------------

def calc_u_kn(energies,params,T=300.,):
    """
    Given a nested list of energies (k lists of length N), calculate full u_kn (kxN) matrix for arbitrary number of new states.
    Where k is the original number of states and N is the length of each trajectory.
    -------------
    energies - len(N) list of all subsampled coordinates representing all configurations visited from the original states
    params - Arbitrary length dictionary of changes in parameter across arbitrary number of states. Highest level key is the molecule AlkEthOH_ID,
             second level of keys are the new state, the values of each of these subkeys are a arbitrary length list of length 3 lists where the 
             length 3 lists contain information on a parameter to change in the form: [SMIRKS, parameter type, parameter value]. I.e. :
    
             params = {'AlkEthOH_c100':{'State 1':[['[6X4:1]-[#1:2]','k','620'],['[6X4:1]-[#6X4:2]','length','1.53'],...],'State 2':[...],...},'AlkEthOH_c1143':                       {...},...}
    
    Returns
    -------------
    
    """
    #Step 1: Calculate the energies of the original configurations so that we can subsample
   
    
    N_k = len(xyzn)
    beta = 1/(kB*T)
    for i in params:
        print i

    return


#N_k = 5000
#files = ['AlkEthOH_c1143_[#6X4:1]-[#6X4:2]-[#8X2H1:3]-[#1:4]_k1_0.08.nc','AlkEthOH_c1143_[#6X4:1]-[#6X4:2]-[#8X2H1:3]-[#1:4]_k1_0.09.nc']
#xyzn_tot = []
#count = 0
#for i in files:
#    data, xyzn = read_traj(i)
#    for i in xyzn:
#        xyzn_tot.append(i)
#        count += 1
#        if count == N_k*len(files):
#            break
#print xyzn_tot        
#print len(xyzn_tot)
#g = timeseries.statisticalInefficiency(xyzn_tot) 
#xyz_sub = [xyzn_tot[timeseries.subsampleCorrelatedData(xyzn_tot,g)]]

ncfiles = glob.glob('traj4ns_c1143/*.nc')
#ncfiles = ncfiles[0:5]
AtomDict,lst_0,lst_1,lst_2 = get_small_mol_dict(['Mol2_files/AlkEthOH_c1143.mol2'])
a = find_smirks_instance('Mol2_files/AlkEthOH_c1143.mol2','[#6X4:1]-[#1:2]')
obs_ind = a[0]

kval = []
lenval = []
bond_len_av = []
bond_len_var = []
bond_len_var_var = []
for ind,name in enumerate(ncfiles):
    print "Analyzing trajectory %s of %s" %(ind+1,len(ncfiles))  
    name_string = name.split('/')[-1].rsplit('.',1)[0].split('_')
    filt_string = filter(lambda x:x.startswith(("length", "k")), name_string) 
    for i in filt_string:
        if i.startswith('k'):
            kval.append(float(i[1:]))
        if i.startswith('length'):
            lenval.append(float(i[6:]))
    if len(filt_string) == 1:
        for i in filt_string:
            if i.startswith('k'):
                lenval.append(1.090)
            if i.startswith('length'):
                kval.append(680.0)
   
    data,xyz = read_traj(name,6250)
    bl = ComputeBondsAnglesTorsions(xyz,AtomDict['Bond'],AtomDict['Angle'],AtomDict['Torsion'])[0]
    num_obs = len(bl[0]) # get number of unique angles in molecule
    timeser = [bl[:,d] for d in range(num_obs)] # re-organize data into timeseries
    A_kn = timeser[obs_ind] # pull out single angle in molecule for test case
    N = np.array([len(A_kn)],int)
    A_av = np.average(A_kn)
    A_var = np.var(A_kn)
    bond_len_av.append(A_av)
    bond_len_var.append(A_var) 
    print "The mean of series A_kn is: %s" %(A_av)
    print "The variance of series A_kn is: %s" %(A_var)
    #implement bootstrapping to get variance of variance estimate
    A_variance_boots = []
    nBoots_work = 200
    for n in range(nBoots_work):
        for k in range(len(N)):
            if N[k] > 0:
                if (n == 0):
                    booti = np.array(range(N[k]),int)
                else:
                    booti = np.random.randint(N[k], size = N[k])
                A_variance_boots.append(np.var(A_kn[booti]))
    #A_variance_boots = np.vstack(np.array(A_variance_boots))
    A_var_var = np.var(A_variance_boots) 
    bond_len_var_var.append(A_var_var) 
    print "The variance of the variance of series A_kn is: %s" %(A_var_var)

if not len(bond_len_av) == len(bond_len_var) == len(bond_len_var_var) == len(kval) == len(lenval):
    print "[len(kval),len(lenval),len(bond_len_av),len(bond_len_var),len(bond_len_var_var)] = [%s,%s,%s,%s,%s]" %(len(kval),len(lenval),len(bond_len_av),len(bond_len_var),len(bond_len_var_var))
    pdb.set_trace()
else:
    print "Storing measurements in pandas dataframe"   
df = pd.DataFrame(
    {'k_values': kval,
     'length_values': lenval,
     'bond_length_average': bond_len_av,
     'bond_length_variance': bond_len_var,
     'bond_length_variance_variance': bond_len_var_var
    })
df.to_csv('AlkEthOH_c1143_C-H_bl_stats.csv',sep=';')
sys.exit()

mol2 = 'Mol2_files/AlkEthOH_c1143'
xyz_orig = []
for i,j in enumerate(ncfiles):
    data, xyz = read_traj(j,4000)
    for pos in xyz:
       xyz_orig.append(pos)

params = {'AlkEthOH_c1143':{'State 1':[['[#6X4:1]-[#1:2]','k','680'],['[#6X4:1]-[#1:2]','length','1.09']],
                            'State 2':[['[#6X4:1]-[#1:2]','k','740'],['[#6X4:1]-[#1:2]','length','1.13']]}}

energies, u_first = new_param_energy(xyz_orig, params)

AtomDict,lst_0,lst_1,lst_2 = get_small_mol_dict(['Mol2_files/AlkEthOH_c1143.mol2'])

ts_sub,N_k_sub,xyz_sub,ind = subsampletimeseries([energies[0]],[xyz_orig],[4000])
obs = ComputeBondsAnglesTorsions(xyz_sub,AtomDict['Bond'],AtomDict['Angle'],AtomDict['Torsion'])[0]

paramsnew = {'AlkEthOH_c1143':{'State 0':[['[#6X4:1]-[#1:2]','k','680'],['[#6X4:1]-[#1:2]','length','1.09']],
                            'State 1':[['[#6X4:1]-[#1:2]','k','650'],['[#6X4:1]-[#1:2]','length','1.09']],
                            'State 2':[['[#6X4:1]-[#1:2]','k','710'],['[#6X4:1]-[#1:2]','length','1.09']],
                            'State 3':[['[#6X4:1]-[#1:2]','k','680'],['[#6X4:1]-[#1:2]','length','1.06']],
                            'State 4':[['[#6X4:1]-[#1:2]','k','650'],['[#6X4:1]-[#1:2]','length','1.06']],
                            'State 5':[['[#6X4:1]-[#1:2]','k','710'],['[#6X4:1]-[#1:2]','length','1.06']],
                            'State 6':[['[#6X4:1]-[#1:2]','k','680'],['[#6X4:1]-[#1:2]','length','1.12']],
                            'State 7':[['[#6X4:1]-[#1:2]','k','650'],['[#6X4:1]-[#1:2]','length','1.12']],
                            'State 8':[['[#6X4:1]-[#1:2]','k','710'],['[#6X4:1]-[#1:2]','length','1.12']]}}


E_kn, u_kn = new_param_energy(xyz_sub, paramsnew)

a = find_smirks_instance('Mol2_files/AlkEthOH_c1143.mol2','[#6X4:1]-[#1:2]')

obs_ind = a[0]
num_obs = len(obs[0]) # get number of unique angles in molecule
timeser = [obs[:,d] for d in range(num_obs)] # re-organize data into timeseries
A_kn = timeser[obs_ind] # pull out single angle in molecule for test case

K,N = np.shape(u_kn)

N_k = np.zeros(K)

N_k[0] = N 

# Initialize MBAR with Newton-Raphson
# Use Adaptive Method (Both Newton-Raphson and Self-Consistent, testing which is better)
initial_f_k = None # start from zero
mbar = pymbar.MBAR(u_kn, N_k, verbose=False, relative_tolerance=1e-12,initial_f_k=initial_f_k)


#------------------------------------------------------------------------
# Compute Expectations for energy and angle distributions
#------------------------------------------------------------------------
print "Computing Expectations for E and A..."
(E_expect, dE_expect) = mbar.computeExpectations(E_kn,state_dependent = True)
(A_expect, dA_expect) = mbar.computeExpectations(A_kn,state_dependent = False)


N_eff = mbar.computeEffectiveSampleNumber(verbose = True)

#from one original sample point ((length,k) combo is (1.09,680)) we're making MBAR estimates at 8 other point to make a 3x3 grid to fit with a plane
#New states are 3% either way in length (1.05 and 1.13) and 5% either way in k (640 and 720)
#See hand-drawn grid for map
#Estimate bond length mean and variance (eventually) for [#6X4:1]-[#1:2] and fit surface to observable over 2d param space

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
def simulate_series_for_posterior(mol_list,SMIRKS_and_params,theta):
    

    mol_filename = ['Mol2_files/'+m+'.mol2' for m in mol_list]
    time_step = 0.8 #Femtoseconds
    temperature = 300 #kelvin
    friction = 1 # per picosecond
    num_steps = 10000 #7500000
    trj_freq = 1000 #steps
    data_freq = 1000 #steps 
    traj_names = []
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
            params[i[1]]=str(theta[ind])
            forcefield.setParameter(params,smirks=i[0])
        system = forcefield.createSystem(topology, [mol])

        filename_string = []
        for ind,i in enumerate(SMIRKS_and_params):
            temp = i[0]+'_'+i[1]+'_'+str(theta[ind])
            filename_string.append(temp)
        filename_string = '_'.join(filename_string)

        #Do simulation
        integrator = mm.LangevinIntegrator(temperature*kelvin, friction/picoseconds, time_step*femtoseconds)
        platform = mm.Platform.getPlatformByName('Reference')
        simulation = app.Simulation(topology, system, integrator)
	simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(temperature*kelvin)
        traj_name = 'traj_posterior/'+mol_list[moldex]+'_'+filename_string+'.nc'
        netcdf_reporter = NetCDFReporter(traj_name, trj_freq)
        simulation.reporters.append(netcdf_reporter)
        simulation.reporters.append(app.StateDataReporter('StateData_posterior/data_'+mol_list[moldex]+'_'+filename_string+'.csv', data_freq, step=True,
                                    potentialEnergy=True, temperature=True, density=True))

        print("Starting simulation")
        start = time.clock()
        simulation.step(num_steps)
        end = time.clock()

        print("Elapsed time %.2f seconds" % (end-start))
        netcdf_reporter.close()
        print("Done!")
        
        traj_names.append(traj_name)
    return traj_names
#------------------------------------------------------------------
def simulate_on_cluster_for_posterior(mol_list,SMIRKS_and_params,theta,slurm_script):
   
    fin = open(slurm_script,'r')
    origlines = fin.readlines()
    fin.close()

    # Keep the header information, but discard the commands that are being submitted to run
    fin = open(slurm_script,'w+')
    fin.writelines(origlines[0:13])
    lines = fin.readlines()
    fin.close()

    for i,j in enumerate(mol_list):
        fout=open(slurm_script,'a')
        newline = """python simulate_for_posterior.py %s "%s" "%s" & \n""" %(j,SMIRKS_and_params,theta)
        fout.write(newline)
        fout.close()

    fout=open(slurm_script,'a')
    newline = 'wait \n'
    fout.write(newline)
    fout.close()

    p = sb.Popen(['sbatch '+slurm_script],shell=True,stdout=sb.PIPE)
    out1 = p.communicate()
    jobid = out1[0][-9:-1]
    print jobid

    for i in range(10000000):
        out = sb.Popen(["squeue", "--job", str(jobid)],
                        stdout=sb.PIPE)
        lines = out.stdout.readlines()
        if len(lines)>1:
            print lines
            print "The job is still running"
        else:
            print "The job is finished!!"
            break
        time.sleep(60)
        
    traj_names = []
    filename_string = []
    for ind,i in enumerate(SMIRKS_and_params):
        temp = i[0]+'_'+i[1]+'_'+str(theta[ind])
        filename_string.append(temp)
    filename_string = '_'.join(filename_string)
    for i in mol_list:
        traj_names.append('traj_posterior/'+i+'_'+filename_string+'.nc')
    
    return traj_names

"""
ax1 = plt.subplot()
x = np.linspace(-1, 1, 500)
posterior_analytical = calc_posterior_analytical(data, x, 0., 1.)
ax1.plot(x, posterior_analytical)
ax1.set(xlabel='mu', ylabel='belief', title='Analytical posterior');
sns.despine()
#------------------------------------------------------------------

df = pd.read_csv('AlkEthOH_c1143_C-H_bl_stats.csv',sep=';')

points_av = df.as_matrix(columns=['k_values','length_values','bond_length_average'])
points_var = df.as_matrix(columns=['k_values','length_values','bond_length_variance'])

def poly_matrix(x, y, order=2):
    # generate Matrix use with lstsq 
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    print ij
    for k, (i, j) in enumerate(ij):
        print i,j
        G[:, k] = x**i * y**j
   
    return G

ordr = 2  # order of polynomial
#x_av_0 = x_av[0]
#y_av_0 = y_av[0]
#x_var_0 = x_var[0]

x_av, y_av, z_av = points_av.T
#x_av, y_av = x_av - x_av[0], y_av - y_av[0]  # this improves accuracy

x_var, y_var, z_var = points_var.T
#x_var, y_var = x_var - x_var[0], y_var - y_var[0]  # this improves accuracy



# make Matrix:
G = poly_matrix(x_av, y_av, ordr)
# Solve for np.dot(G, m) = z:
m_av = np.linalg.lstsq(G, z_av)[0]

# Solve for np.dot(G, m) = z:
m_var = np.linalg.lstsq(G, z_var)[0]


# Evaluate it on a grid...
nx, ny = 100, 100
xx, yy = np.meshgrid(np.linspace(x_av.min(), x_av.max(), nx),
                     np.linspace(y_av.min(), y_av.max(), ny))

GG = poly_matrix(xx.ravel(), yy.ravel(), ordr)
zz_av = np.reshape(np.dot(GG, m_av), xx.shape)
zz_var = np.reshape(np.dot(GG, m_var), xx.shape)

# Plotting (see http://matplotlib.org/examples/mplot3d/custom_shaded_3d_surface.html):

fg, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ls = LightSource(270, 45)
rgb = ls.shade(zz_av, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
#heatmap = ax.pcolor(zz_av, cmap=rgb)                  
#plt.colorbar(mappable=heatmap)    # put the major ticks at the middle of each cell
surf = ax.plot_surface(xx, yy, zz_av, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)

ax.set_xlabel('Bonded force constant - (kcal/mol/A^2)')
ax.set_ylabel('Equilibrium bond length - (A)')
ax.set_zlabel('Average of bond length distribution - (A)')
ax.plot3D(x_av, y_av, z_av, "o")

fg.canvas.draw()
plt.savefig('bond_length_average_vs_k_length_w_fit.png')


zz_comp = m_var[0] + m_var[1]*yy + m_var[2]*yy**2 + m_var[3]*xx + m_var[4]*xx*yy + m_var[5]*xx*(yy**2) + m_var[6]*xx**2 + m_var[7]*(xx**2)*yy + m_var[8]*(xx**2)*(yy**2)

# Plotting (see http://matplotlib.org/examples/mplot3d/custom_shaded_3d_surface.html):
fg, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ls = LightSource(270, 45)
rgb = ls.shade(zz_var, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(xx, yy, zz_var,cmap=cm.gist_earth, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)
#fg.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel('Bonded force constant - (kcal/mol/A^2)')
ax.set_ylabel('Equilibrium bond length - (A)')
ax.set_zlabel('Variance of bond length distribution - (A^2)')
ax.plot3D(x_var, y_var, z_var, "o")

fg.canvas.draw()
plt.savefig('bond_length_variance_vs_k_length_w_fit.png')

x_av,res_av,rank_av,s_av = np.linalg.lstsq(G, z_av)
x_var,res_var,rank_var,s_var = np.linalg.lstsq(G, z_var)

#--------------------------------------------------------------
"""

"""
print "starting to calculate evidence"
SMIRKS = ['[#6X4:1]-[#1:2]','[#6X4:1]-[#6X4:2]']
mol_list = ['AlkEthOH_c1143','AlkEthOH_c1163'] 
current_trajectories = ['traj_evidence/AlkEthOH_c1143_evidence.nc','traj_evidence/AlkEthOH_c1163_evidence.nc']
for index,ii in enumerate(current_trajectories):
    data,xyz = read_traj(ii,2000)
    AtomDict = get_small_mol_dict(['Mol2_files/'+mol_list[index]+'.mol2'])[0]

    observables = ComputeBondsAnglesTorsions(xyz,AtomDict['Bond'],AtomDict['Angle'],AtomDict['Torsion'])
    num_obs = [len(a[0]) for a in observables] # get number of unique observables in molecule
    timeser = [[a[:,d] for d in range(g)] for a,g in zip(observables,num_obs)] # re-organize data into timeseries'

    SMIRKS_instances = []
    for SMIRKS_index,SMIRKS_s in enumerate(SMIRKS):
        SMIRKS_instances.append(find_smirks_instance('Mol2_files/'+mol_list[index]+'.mol2',SMIRKS_s))
    obs_per_smirk=[]
    for inds,SMIRKSS in enumerate(SMIRKS_instances):


        timesers = timeser[0]
        A_kn = np.array([timesers[match] for match in SMIRKSS])
        SMIRKSS_inds = np.array([match for match in SMIRKSS],int)
        N = np.array([len(lst) for lst in A_kn],int)
        A_average = np.array([np.mean(A) for A in A_kn])
        A_variance = np.array([np.var(A) for A in A_kn])

        #implement bootstrapping to get variance of variance estimate
        A_variance_boots = []
        nBoots_work = 200
        for n in range(nBoots_work):
            for k in range(len(A_kn)):
                if N[k] > 0:
                    if (n == 0):
                        booti = np.array(range(N[k]),int)
                    else:
                        booti = np.random.randint(N[k], size = N[k])
                    A_variance_boots.append([np.var(A[booti]) for A in A_kn])
        A_variance_boots = np.vstack(np.array(A_variance_boots))
        A_variance_variance = np.array([np.var(A_variance_boots[:,i]) for i in range(len(SMIRKSS))])


        print [mol_list[index],SMIRKS[inds],SMIRKSS,A_average,A_variance,A_variance_variance]

sys.exit()
"""

def sampler(data, mol_list, samples, theta_init, proposal_width,method, parallel=True):
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
    4)Will wait until surrogate either stops changing or we get 
     
    Notes: Initial conditions should be a dictionary in order to track what SMIRKS are changing and initiate the simulations as needed
    theta_init = OrderedDict([('[#6X4:1]-[#1:2]',OrderedDict([('k',500),('length',0.8)])),('[#6X4:1]-[#6X4:2]',OrderedDict([('k',700),('length',1.6)]))])
    """
    # Begin process by initializing a dataframe where the posterior values will be stored
    posterior_columns = [j+'_'+jj for j in theta_init for jj in theta_init[j]] 
    posterior = pd.DataFrame(columns=posterior_columns)

    SMIRKS_and_params = [[j,jj] for j in theta_init for jj in theta_init[j]]
 
    theta_current = [theta_init[j][jj] for j in theta_init for jj in theta_init[j]]
 
    # The theta list must be the same size as the proposal width. If not, exit()
    if len(theta_current)!=len(proposal_width):
        raise ValueError('Every parameter being changed must have a specified proposal width')
 
    posterior.loc[0] = [theta_init[j][jj] for j in theta_init for jj in theta_init[j]]
    # During sampling we will be switching around between classical mechanics simulations, MBAR and surrogate modelling in order to calculate observables 
    # as a function of parameter state. `obs_mode` will be used to record the method used per state
    obs_mode = []
    # Determine the observable type for each parameter type we're tracking changes for
    SMIRKS = list(set([j[0] for j in SMIRKS_and_params]))
    len_SMIRKS = [len(j.split('-')) for j in SMIRKS]
    obs_types = []
    for j in len_SMIRKS:
        if j==2:
            obs_types.append('Bond')
        if j==3:
            obs_types.append('Angle')
        if j==4:
            obs_types.append('Torsion')
    # Create a dataframe for recording observable data for fitting separate from the posterior
    observables_posterior_columns = []
    for index,mol in enumerate(mol_list):
        SMIRKS_instances = []
        for SMIRKS_index,SMIRKS_s in enumerate(SMIRKS):
            SMIRKS_instances.append(find_smirks_instance('Mol2_files/'+mol_list[index]+'.mol2',SMIRKS_s))      
             
        if obs_types[SMIRKS_index]=='Bond':
            for inds,j in enumerate(SMIRKS_instances):
                for i in j:
                    observables_posterior_columns.append(mol_list[index]+'_'+SMIRKS[inds]+'_'+str(i)+'_length_average')
                    observables_posterior_columns.append(mol_list[index]+'_'+SMIRKS[inds]+'_'+str(i)+'_length_variance')
                    
        if obs_types[SMIRKS_index]=='Angle':
            for inds,j in SMIRKS_instances:
                for i in j:
                    observables_posterior_columns.append(mol_list[index]+'_'+SMIRKS[inds]+'_'+str(i)+'_angle_average')
                    observables_posterior_columns.append(mol_list[index]+'_'+SMIRKS[inds]+'_'+str(i)+'_angle_variance')
                    
        if obs_types[SMIRKS_index]=='Torsion':
            for inds,j in SMIRKS_instances:
                for i in j:
                    observables_posterior_columns.append(mol_list[index]+'_'+SMIRKS[inds]+'_'+str(i)+'_torsion_pmf_fit_c0')
                    observables_posterior_columns.append(mol_list[index]+'_'+SMIRKS[inds]+'_'+str(i)+'_torsion_pmf_fit_c1')
                    observables_posterior_columns.append(mol_list[index]+'_'+SMIRKS[inds]+'_'+str(i)+'_torsion_pmf_fit_c2')
                    observables_posterior_columns.append(mol_list[index]+'_'+SMIRKS[inds]+'_'+str(i)+'_torsion_pmf_fit_c3')
                    observables_posterior_columns.append(mol_list[index]+'_'+SMIRKS[inds]+'_'+str(i)+'_torsion_pmf_fit_c4')
                    observables_posterior_columns.append(mol_list[index]+'_'+SMIRKS[inds]+'_'+str(i)+'_torsion_pmf_fit_c5')  
    observables_posterior = pd.DataFrame(columns=observables_posterior_columns)
     
    # Start sampling loop 
    for ind,i in enumerate(range(samples)): 
        if (ind==0) and (method=='1sim_MBAR_surr'):
            """if parallel==True:
                current_trajectories = simulate_on_cluster_for_posterior(mol_list,SMIRKS_and_params,theta_current,'slurm_posterior.sh')           
            else:
                current_trajectories = simulate_series_for_posterior(mol_list,SMIRKS_and_params,theta_current) """
            current_trajectories = ['traj_posterior/AlkEthOH_c1143_[#6X4:1]-[#1:2]_k_500_[#6X4:1]-[#1:2]_length_0.8_[#6X4:1]-[#6X4:2]_k_700_[#6X4:1]-[#6X4:2]_length_1.6.nc','traj_posterior/AlkEthOH_c1163_[#6X4:1]-[#1:2]_k_500_[#6X4:1]-[#1:2]_length_0.8_[#6X4:1]-[#6X4:2]_k_700_[#6X4:1]-[#6X4:2]_length_1.6.nc']
            # compute observables for current state
            all_xyz = [[] for i in mol_list]
            A_kns = [[] for i in mol_list]
            SMIRKS_inst_per_mol = [[] for i in mol_list]
            for index,ii in enumerate(current_trajectories):
                data,xyz = read_traj(ii,1250)
                for pos in xyz:
                    all_xyz[index].append(pos)
                
                AtomDict = get_small_mol_dict(['Mol2_files/'+mol_list[index]+'.mol2'])[0]

                observables = ComputeBondsAnglesTorsions(xyz,AtomDict['Bond'],AtomDict['Angle'],AtomDict['Torsion'])
                num_obs = [len(a[0]) for a in observables] # get number of unique observables in molecule
                timeser = [[a[:,d] for d in range(g)] for a,g in zip(observables,num_obs)] # re-organize data into timeseries'

                SMIRKS_instances = []
                for SMIRKS_index,SMIRKS_s in enumerate(SMIRKS): 
                    SMIRKS_instances.append(find_smirks_instance('Mol2_files/'+mol_list[index]+'.mol2',SMIRKS_s))
                SMIRKS_inst_per_mol[index].append(SMIRKS_instances)
                obs_per_smirk=[]
                for inds,SMIRKSS in enumerate(SMIRKS_instances):

                    if obs_types[inds]=="Bond":
                        timesers = timeser[0]
                        A_kn = np.array([timesers[match] for match in SMIRKSS])
                        for A in A_kn:
                            A_kns[index].append(A)                       
                        SMIRKSS_inds = np.array([match for match in SMIRKSS],int)
                        N = np.array([len(lst) for lst in A_kn],int)
                        A_average = np.array([np.mean(A) for A in A_kn])
                        A_variance = np.array([np.var(A) for A in A_kn])


                        for jjj,value in enumerate(SMIRKSS):
                            observables_posterior.set_value(ind,mol_list[index]+'_'+SMIRKS[inds]+'_'+str(value)+'_length_average',A_average[jjj])
                            observables_posterior.set_value(ind,mol_list[index]+'_'+SMIRKS[inds]+'_'+str(value)+'_length_variance',A_variance[jjj])

                    if obs_types[inds]=="Angle":
                        timeser = timeser[1]
                        A_kn = np.array([timesers[match] for match in SMIRKSS])
                        for A in A_kn:
                            A_kns[index].append(A)
                        SMIRKSS_inds = np.array([match for match in SMIRKSS],int)
                        N = np.array([len(lst) for lst in A_kn],int)
                        A_average = np.array([np.mean(A) for A in A_kn])
                        A_variance = np.array([np.var(A) for A in A_kn])

                        for jjj,value in enumerate(SMIRKSS):
                            observables_posterior.set_value(ind,mol_list[index]+'_'+SMIRKS[inds]+'_'+str(value)+'_angle_average',A_average[jjj])
                            observables_posterior.set_value(ind,mol_list[index]+'_'+SMIRKS[inds]+'_'+str(value)+'_angle_variance',A_variance[jjj])
               
                    if obs_types[inds]=="Torsion":
                        timeser = timeser[2]
                        #pmf thing
                        A_kn = [timeser[match] for match in SMIRKSS]
                        SMIRKSS_inds = [match for match in SMIRKSS]
                        obs_per_ind = []
                        A_kns[index].append(A_kn)
                        for indexx,A in enumerate(A_kn):
                            num_bins = 100
                            ntotal = len(A)
                            # figure out what the width of the kernel density is.
                            # the "rule-of-thumb" estimator used std, but that is for gaussian.  We should use instead
                            # the stdev of the Gaussian-like features. Playing around with what it looks like, then something
                            # like 12 degress as 2 sigma? So sigma is about  degrees = 3/360 * 2*pi = 0.0524
                            # this gives a relatively smooth PMF, without smoothing too much.
                            # this will of course depend on the temperature the simulation is run at.

                            sd = 1.06*0.0524*ntotal**(-0.2)
                            # create a fine grid
                            ngrid = 10000
                            kT = 300*kB  # units of kcal/mol
                            x = np.arange(-np.pi,np.pi,(2*np.pi)/ngrid)
                            y = np.zeros(ngrid)
                            # Easier to use a von Mises distribution than a wrapped Gaussian.
                            denom = 2*np.pi*scipy.special.iv(0,1/sd)
                            for a in A:
                                y += np.exp(np.cos(x-a)/sd)/denom
                                y /= ntotal


                            pmf = -kT*np.log(y) # now we have the PMF

                            # adapted from http://stackoverflow.com/questions/4258106/how-to-calculate-a-fourier-series-in-numpy
                            # complex fourier coefficients
                            def cn(n,y):
                                c = y*np.exp(-1j*n*x)
                                return c.sum()/c.size

                            def ft(x, cn, Nh):
                                f = np.array([2*cn[i]*np.exp(1j*i*x) for i in range(1,Nh+1)])
                                return f.sum()+cn[0]

                            # generate Fourier series (complex)
                            Ns = 6 # needs to be adjusted
                            cf = np.zeros(Ns+1,dtype=complex)
                            for i in range(Ns+1):
                                cf[i] = cn(i,pmf)

                            y1 = np.array([ft(xi,cf,Ns).real for xi in x])  # plot the fourier series approximation.

                            # OK, Fourier series works pretty well.  But we actually want to do a
                            # linear least square fit to a fourier series, since we want to get
                            # the coefficients out.  Let's use the standard LLS formulation with
                            # normal equations.
                            # http://www.math.uconn.edu/~leykekhman/courses/MATH3795/Lectures/Lecture_9_Linear_least_squares_SVD.pdf
                            # basis functions are 1, sin(x), cos(x), sin(2x), cos(2x), . . .
                            Z = np.ones([len(x),2*Ns+1])
                            for i in range(1,Ns+1):
                                Z[:,2*i-1] = np.sin(i*x)
                                Z[:,2*i] = np.cos(i*x)
                            ZM = np.matrix(Z) # easier to manipulate as a matrix
                            [U,S,V] = np.linalg.svd(ZM)    # perform SVD  - S has an interesting shape, is just 1, sqrt(2), sqrt(2). Probably has
                                                       # to do with the normalization.  Still need V and U, though.
                            Sinv = np.matrix(np.zeros(np.shape(Z))).transpose()  # get the inverse of the singular matrix.
                            for i in range(2*Ns+1):
                                Sinv[i,i] = 1/S[i]
                            cm = V.transpose()*Sinv*U.transpose()*np.matrix(pmf).transpose()  # get the linear constants
                            cl = np.array(cm) # cast back to array for plotting
                            # check that it works by plotting
                            y2 = cl[0]*np.ones(len(x))
                            for i in range(1,Ns+1):
                                y2 += cl[2*i-1]*np.sin(i*x)
                                y2 += cl[2*i]*np.cos(i*x)


                            # determine the covariance matrix for the fitting parameters
                            dev = pmf - np.array(ZM*cm).transpose()
                            residuals = np.sum(dev**2)
                            s2 = residuals /(len(pmf) - 2*Ns+1)
                            cov = s2*(V.transpose()*np.linalg.inv(np.diag(S**2))*V)

                            observables_posterior.set_value(ind,mol_list[index]+'_'+SMIRKS[inds]+'_'+SMIRKS_ind[index]+'_torsion_pmf_fit_c0',cl[0])
                            observables_posterior.set_value(ind,mol_list[index]+'_'+SMIRKS[inds]+'_'+SMIRKS_ind[index]+'_torsion_pmf_fit_c1',cl[1])
                            observables_posterior.set_value(ind,mol_list[index]+'_'+SMIRKS[inds]+'_'+SMIRKS_ind[index]+'_torsion_pmf_fit_c2',cl[2])
                            observables_posterior.set_value(ind,mol_list[index]+'_'+SMIRKS[inds]+'_'+SMIRKS_ind[index]+'_torsion_pmf_fit_c3',cl[3])
                            observables_posterior.set_value(ind,mol_list[index]+'_'+SMIRKS[inds]+'_'+SMIRKS_ind[index]+'_torsion_pmf_fit_c4',cl[4])
                            observables_posterior.set_value(ind,mol_list[index]+'_'+SMIRKS[inds]+'_'+SMIRKS_ind[index]+'_torsion_pmf_fit_c5',cl[5])
            obs_mode.append('simulation')
            A_expectations = [[] for a in all_xyz]
            A_var_expectations = [[] for a in all_xyz]
            A_var_alt_expectations = [[] for a in all_xyz]
            dA_expectations = [[] for a in all_xyz]
            for ii,value in enumerate(all_xyz):                 
                #do reweighting
                #suggest new position
                MBAR_coordinates = [[] for i in theta_current]
                for indexx,vall in enumerate(theta_current):
                    if SMIRKS_and_params[indexx][1] == 'k':
                        MBAR_coordinates[indexx].append(str(float(theta_current[indexx]) - 0.05*float(theta_current[indexx])))
                        MBAR_coordinates[indexx].append(str(float(theta_current[indexx]) + 0.05*float(theta_current[indexx])))
                        MBAR_coordinates[indexx].append(str(theta_current[indexx]))
                    if (SMIRKS_and_params[indexx][1] == 'length') or (SMIRKS_and_params[indexx][1] == 'angle'):
                        MBAR_coordinates[indexx].append(str(float(theta_current[indexx]) - 0.02*float(theta_current[indexx])))
                        MBAR_coordinates[indexx].append(str(float(theta_current[indexx]) + 0.02*float(theta_current[indexx])))
                        MBAR_coordinates[indexx].append(str(theta_current[indexx]))
                MBAR_moves = list(itertools.product(*MBAR_coordinates))
                MBAR_moves[0],MBAR_moves[-1] = MBAR_moves[-1],MBAR_moves[0]
                #theta_current_float = [str(i) for i in theta_current]
                #MBAR_moves.insert(0,theta_current_float)
                D = OrderedDict()
                for i,val in enumerate(MBAR_moves):
                    D['State' + ' ' + str(i)] = [[SMIRKS_and_params[j][0],SMIRKS_and_params[j][1],val[j]] for j in range(len(MBAR_coordinates))] 
                D_mol = {mol_list[ii] : D}
               
                energies, u = new_param_energy(all_xyz[ii], D_mol)
                
                AtomDict,lst_0,lst_1,lst_2 = get_small_mol_dict(['Mol2_files/'+mol_list[ii]+'.mol2'])
                ts_sub,N_k_sub,xyz_sub,ind_subs = subsampletimeseries([energies[0]],[all_xyz[ii]],[len(all_xyz[ii])])     
                E_kn, u_kn = new_param_energy(xyz_sub, D_mol)

                K,N = np.shape(u_kn)

                N_k = np.zeros(K)
                N_k[0] = N

                # Initialize MBAR with Newton-Raphson
                # Use Adaptive Method (Both Newton-Raphson and Self-Consistent, testing which is better)
                initial_f_k = None # start from zero
                mbar = pymbar.MBAR(u_kn, N_k, verbose=False, relative_tolerance=1e-12,initial_f_k=initial_f_k)
                
                  
                for ind_A,A_kn in enumerate(A_kns[ii]):                  
                    A_kn = [A_kn[ind_sub] for ind_sub in ind_subs[0]]
                    A_kn_var1 = [(A - np.average(A))**2 for A in A_kn]
                    #------------------------------------------------------------------------
                    # Compute Expectations for energy and angle distributions
                    #------------------------------------------------------------------------
                    (E_expect, dE_expect) = mbar.computeExpectations(E_kn,state_dependent = True)
                    (A_expect, dA_expect) = mbar.computeExpectations(A_kn,state_dependent = False)
                    
                    A_kn_var = [(A - A_expect[0])**2 for A in A_kn]
                                           
                    (A_var_expect, dA_var_expect) = mbar.computeExpectations(A_kn_var,state_dependent = False)  
                    (A_var_expect_alt,dA_var_expect_alt) = mbar.computeExpectations(A_kn_var1,state_dependent = False)
                    A_expectations[ii].append(A_expect)
                    A_var_expectations[ii].append(A_var_expect)
                    A_var_alt_expectations[ii].append(A_var_expect_alt)
                    dA_expectations[ii].append(dA_expect)
        
            print A_expectations
            print A_var_expectations
            print A_var_alt_expectations
            print MBAR_moves
            print SMIRKS_inst_per_mol
            pdb.set_trace()
        if (ind < 10) or (ind % 20 == 0):
            #Directly simulate the first 10 steps or every 20 after that
            if parallel==True: 
                # pass simulation commands to queue and run all molecules simultaneously at prescribed state  
                proposed_trajectories = simulate_on_cluster_for_posterior(mol_list,SMIRKS_and_params,theta_proposal,'slurm_posterior.sh')
            else:
                proposed_trajectories = simulate_series_for_posterior(mol_list,SMIRKS_and_params,theta_proposal) 
 
        Observables_current = []
        Observables_proposal = []
        if ind==0:
            # compute observables for current state
            for index,ii in enumerate(current_trajectories): 
                data,xyz = read_traj(ii)#,-1000:)
                AtomDict = get_small_mol_dict(['Mol2_files/'+mol_list[index]+'.mol2'])[0]
                 
                observables = ComputeBondsAnglesTorsions(xyz,AtomDict['Bond'],AtomDict['Angle'],AtomDict['Torsion'])
                num_obs = [len(a[0]) for a in observables] # get number of unique observables in molecule
                timeser = [[a[:,d] for d in range(g)] for a,g in zip(observables,num_obs)] # re-organize data into timeseries'
            
                SMIRKS_instances = []
                for SMIRKS_index,SMIRKS_s in enumerate(SMIRKS):
                    SMIRKS_instances.append(find_smirks_instance('Mol2_files/'+mol_list[index]+'.mol2',SMIRKS_s)) 
                obs_per_smirk=[]
                for inds,SMIRKSS in enumerate(SMIRKS_instances):

                    if obs_types[inds]=="Bond":
                        timesers = timeser[0]
                        A_kn = np.array([timesers[match] for match in SMIRKSS])
                        SMIRKSS_inds = np.array([match for match in SMIRKSS],int)
                        N = np.array([len(lst) for lst in A_kn],int)
                        A_average = np.array([np.mean(A) for A in A_kn])
                        A_variance = np.array([np.var(A) for A in A_kn])

                        #implement bootstrapping to get variance of variance estimate
                        A_variance_boots = []
                        nBoots_work = 200
                        for n in range(nBoots_work):
                            for k in range(len(A_kn)):
                                if N[k] > 0:
                                    if (n == 0):
                                        booti = np.array(range(N[k]),int)
                                    else:
                                        booti = np.random.randint(N[k], size = N[k])
                                    A_variance_boots.append([np.var(A[booti]) for A in A_kn])
                        A_variance_boots = np.vstack(np.array(A_variance_boots))
                        A_variance_variance = np.array([np.var(A_variance_boots[:,i]) for i in range(len(SMIRKSS))])
                        
                        for jjj,value in enumerate(SMIRKSS):              
                            observables_posterior.set_value(ind,mol_list[index]+'_'+SMIRKS[inds]+'_'+str(value)+'_length_average',A_average[jjj])
                            observables_posterior.set_value(ind,mol_list[index]+'_'+SMIRKS[inds]+'_'+str(value)+'_length_variance',A_variance[jjj])
                            #observables_posterior.set_value(ind,mol_list[index]+'_'+SMIRKS[inds]+'_'+str(value)+'_length_variance_variance',A_variance_variance[jjj])
                    if obs_types[inds]=="Angle":
                        timeser = timeser[1]
                        A_kn = np.array([timesers[match] for match in SMIRKSS])
                        SMIRKSS_inds = np.array([match for match in SMIRKSS],int)
                        N = np.array([len(lst) for lst in A_kn],int)
                        A_average = np.array([np.mean(A) for A in A_kn])
                        A_variance = np.array([np.var(A) for A in A_kn])
                        #implement bootstrapping to get variance of variance estimate
                        A_variance_boots = []
                        nBoots_work = 200
                        for n in range(nBoots_work):
                            for k in range(len(A_kn)):
                                if N[k] > 0:
                                    if (n == 0):
                                        booti = np.array(range(N[k]),int)
                                    else:
                                        booti = np.random.randint(N[k], size = N[k])
                                    A_variance_boots.append([np.var(A[booti]) for A in A_kn])
                        A_variance_boots = np.vstack(np.array(A_variance_boots))
                        A_variance_variance = np.array([np.var(A_variance_boots[:,i]) for i in range(len(SMIRKSS))])
                        
                        for jjj,value in enumerate(SMIRKSS):
                            observables_posterior.set_value(ind,mol_list[index]+'_'+SMIRKS[inds]+'_'+str(value)+'_angle_average',A_average[jjj])
                            observables_posterior.set_value(ind,mol_list[index]+'_'+SMIRKS[inds]+'_'+str(value)+'_angle_variance',A_variance[jjj])
                            #observables_posterior.set_value(ind,mol_list[index]+'_'+SMIRKS[inds]+'_'+str(value)+'_angle_variance_variance',A_variance_variance[jjj])

                    if obs_types[inds]=="Torsion":
                        timeser = timeser[2]
                        #pmf thing
                        A_kn = [timeser[match] for match in SMIRKSS]
                        SMIRKSS_inds = [match for match in SMIRKSS]
                        obs_per_ind = []
                        for indexx,A in enumerate(A_kn):
                            num_bins = 100
                            ntotal = len(A)
                            # figure out what the width of the kernel density is. 
                            # the "rule-of-thumb" estimator used std, but that is for gaussian.  We should use instead
                            # the stdev of the Gaussian-like features. Playing around with what it looks like, then something 
                            # like 12 degress as 2 sigma? So sigma is about  degrees = 3/360 * 2*pi = 0.0524
                            # this gives a relatively smooth PMF, without smoothing too much.
                            # this will of course depend on the temperature the simulation is run at.
 
                            sd = 1.06*0.0524*ntotal**(-0.2)
                            # create a fine grid
                            ngrid = 10000
                            kT = 300*kB  # units of kcal/mol
                            x = np.arange(-np.pi,np.pi,(2*np.pi)/ngrid)
                            y = np.zeros(ngrid)
                            # Easier to use a von Mises distribution than a wrapped Gaussian.
                            denom = 2*np.pi*scipy.special.iv(0,1/sd)
                            for a in A:
                                y += np.exp(np.cos(x-a)/sd)/denom
                                y /= ntotal


                            pmf = -kT*np.log(y) # now we have the PMF

                            # adapted from http://stackoverflow.com/questions/4258106/how-to-calculate-a-fourier-series-in-numpy
                            # complex fourier coefficients
                            def cn(n,y):
                                c = y*np.exp(-1j*n*x)
                                return c.sum()/c.size
 
                            def ft(x, cn, Nh):
                                f = np.array([2*cn[i]*np.exp(1j*i*x) for i in range(1,Nh+1)])
                                return f.sum()+cn[0]

                            # generate Fourier series (complex)
                            Ns = 6 # needs to be adjusted 
                            cf = np.zeros(Ns+1,dtype=complex)
                            for i in range(Ns+1):
                                cf[i] = cn(i,pmf)

                            y1 = np.array([ft(xi,cf,Ns).real for xi in x])  # plot the fourier series approximation.
  
                            # OK, Fourier series works pretty well.  But we actually want to do a
                            # linear least square fit to a fourier series, since we want to get
                            # the coefficients out.  Let's use the standard LLS formulation with
                            # normal equations.
                            # http://www.math.uconn.edu/~leykekhman/courses/MATH3795/Lectures/Lecture_9_Linear_least_squares_SVD.pdf
                            # basis functions are 1, sin(x), cos(x), sin(2x), cos(2x), . . . 
                            Z = np.ones([len(x),2*Ns+1]) 
                            for i in range(1,Ns+1):
                                Z[:,2*i-1] = np.sin(i*x)
                                Z[:,2*i] = np.cos(i*x)
                            ZM = np.matrix(Z) # easier to manipulate as a matrix
                            [U,S,V] = np.linalg.svd(ZM)    # perform SVD  - S has an interesting shape, is just 1, sqrt(2), sqrt(2). Probably has 
                                                       # to do with the normalization.  Still need V and U, though.
                            Sinv = np.matrix(np.zeros(np.shape(Z))).transpose()  # get the inverse of the singular matrix.
                            for i in range(2*Ns+1):   
                                Sinv[i,i] = 1/S[i]
                            cm = V.transpose()*Sinv*U.transpose()*np.matrix(pmf).transpose()  # get the linear constants
                            cl = np.array(cm) # cast back to array for plotting 
                            # check that it works by plotting
                            y2 = cl[0]*np.ones(len(x))
                            for i in range(1,Ns+1):
                                y2 += cl[2*i-1]*np.sin(i*x)
                                y2 += cl[2*i]*np.cos(i*x)


                            # determine the covariance matrix for the fitting parameters
                            dev = pmf - np.array(ZM*cm).transpose()
                            residuals = np.sum(dev**2)
                            s2 = residuals /(len(pmf) - 2*Ns+1)  
                            cov = s2*(V.transpose()*np.linalg.inv(np.diag(S**2))*V)
                            
                            observables_posterior.set_value(ind,mol_list[index]+'_'+SMIRKS[inds]+'_'+SMIRKS_ind[index]+'_torsion_pmf_fit_c0',cl[0])
                            observables_posterior.set_value(ind,mol_list[index]+'_'+SMIRKS[inds]+'_'+SMIRKS_ind[index]+'_torsion_pmf_fit_c1',cl[1])
                            observables_posterior.set_value(ind,mol_list[index]+'_'+SMIRKS[inds]+'_'+SMIRKS_ind[index]+'_torsion_pmf_fit_c2',cl[2])
                            observables_posterior.set_value(ind,mol_list[index]+'_'+SMIRKS[inds]+'_'+SMIRKS_ind[index]+'_torsion_pmf_fit_c3',cl[3])
                            observables_posterior.set_value(ind,mol_list[index]+'_'+SMIRKS[inds]+'_'+SMIRKS_ind[index]+'_torsion_pmf_fit_c4',cl[4])
                            observables_posterior.set_value(ind,mol_list[index]+'_'+SMIRKS[inds]+'_'+SMIRKS_ind[index]+'_torsion_pmf_fit_c5',cl[5])
        obs_mode.append('simulation')
                           
        if ind < 10 or ind % 20 == 0:
            # And calculate observables for proposed state by simulation
            for index,ii in enumerate(proposed_trajectories):
                data,xyz = read_traj(ii)#,-1000:)
                AtomDict = get_small_mol_dict(['Mol2_files/'+mol_list[index]+'.mol2'])[0]

                observables = ComputeBondsAnglesTorsions(xyz,AtomDict['Bond'],AtomDict['Angle'],AtomDict['Torsion'])
                num_obs = [len(a[0]) for a in observables] # get number of unique observables in molecule
                timeser = [[a[:,d] for d in range(g)] for a,g in zip(observables,num_obs)] # re-organize data into timeseries'

                SMIRKS_instances = []
                for SMIRKS_index,SMIRKS_s in enumerate(SMIRKS):
                    SMIRKS_instances.append(find_smirks_instance('Mol2_files/'+mol_list[index]+'.mol2',SMIRKS_s))
                obs_per_smirk=[]
                for inds,SMIRKSS in enumerate(SMIRKS_instances):

                    if obs_types[inds]=="Bond":
                        timesers = timeser[0]
                        A_kn = np.array([timesers[match] for match in SMIRKSS])
                        SMIRKSS_inds = np.array([match for match in SMIRKSS],int)
                        N = np.array([len(lst) for lst in A_kn],int)
                        A_average = np.array([np.mean(A) for A in A_kn])
                        A_variance = np.array([np.var(A) for A in A_kn])

                        #implement bootstrapping to get variance of variance estimate
                        A_variance_boots = []
                        nBoots_work = 200
                        for n in range(nBoots_work):
                            for k in range(len(A_kn)):
                                if N[k] > 0:
                                    if (n == 0):
                                        booti = np.array(range(N[k]),int)
                                    else:
                                        booti = np.random.randint(N[k], size = N[k])
                                    A_variance_boots.append([np.var(A[booti]) for A in A_kn])
                        A_variance_boots = np.vstack(np.array(A_variance_boots))
                        A_variance_variance = np.array([np.var(A_variance_boots[:,i]) for i in range(len(SMIRKSS))])

                        for jjj,value in enumerate(SMIRKSS):
                            observables_posterior.set_value(ind+1,mol_list[index]+'_'+SMIRKS[inds]+'_'+str(value)+'_length_average',A_average[jjj])
                            observables_posterior.set_value(ind+1,mol_list[index]+'_'+SMIRKS[inds]+'_'+str(value)+'_length_variance',A_variance[jjj])
                            #observables_posterior.set_value(ind+1,mol_list[index]+'_'+SMIRKS[inds]+'_'+str(value)+'_length_variance_variance',A_variance_variance[jjj])
                    if obs_types[inds]=="Angle":
                        timeser = timeser[1]
                        A_kn = np.array([timesers[match] for match in SMIRKSS])
                        SMIRKSS_inds = np.array([match for match in SMIRKSS],int)
                        N = np.array([len(lst) for lst in A_kn],int)
                        A_average = np.array([np.mean(A) for A in A_kn])
                        A_variance = np.array([np.var(A) for A in A_kn])

                        #implement bootstrapping to get variance of variance estimate
                        A_variance_boots = []
                        nBoots_work = 200
                        for n in range(nBoots_work):
                            for k in range(len(A_kn)):
                                if N[k] > 0:
                                    if (n == 0):
                                        booti = np.array(range(N[k]),int)
                                    else:
                                        booti = np.random.randint(N[k], size = N[k])
                                    A_variance_boots.append([np.var(A[booti]) for A in A_kn])
                        A_variance_boots = np.vstack(np.array(A_variance_boots))
                        A_variance_variance = np.array([np.var(A_variance_boots[:,i]) for i in range(len(SMIRKSS))])

                        for jjj,value in enumerate(SMIRKSS):
                            observables_posterior.set_value(ind+1,mol_list[index]+'_'+SMIRKS[inds]+'_'+str(value)+'_angle_average',A_average[jjj])
                            observables_posterior.set_value(ind+1,mol_list[index]+'_'+SMIRKS[inds]+'_'+str(value)+'_angle_variance',A_variance[jjj])
                            #observables_posterior.set_value(ind+1,mol_list[index]+'_'+SMIRKS[inds]+'_'+str(value)+'_angle_variance_variance',A_variance_variance[jjj])

                    if obs_types[inds]=="Torsion":
                        timeser = timeser[2]
                        #pmf thing
                        A_kn = [timeser[match] for match in SMIRKSS]
                        SMIRKSS_inds = [match for match in SMIRKSS]
                        obs_per_ind = []
                        for indexx,A in enumerate(A_kn):
                            num_bins = 100
                            ntotal = len(A)
                            # figure out what the width of the kernel density is.
                            # the "rule-of-thumb" estimator used std, but that is for gaussian.  We should use instead
                            # the stdev of the Gaussian-like features. Playing around with what it looks like, then something
                            # like 12 degress as 2 sigma? So sigma is about  degrees = 3/360 * 2*pi = 0.0524
                            # this gives a relatively smooth PMF, without smoothing too much.
                            # this will of course depend on the temperature the simulation is run at.

                            sd = 1.06*0.0524*ntotal**(-0.2)
                            # create a fine grid
                            ngrid = 10000
                            kT = 300*kB  # units of kcal/mol
                            x = np.arange(-np.pi,np.pi,(2*np.pi)/ngrid)
                            y = np.zeros(ngrid)
                            # Easier to use a von Mises distribution than a wrapped Gaussian.
                            denom = 2*np.pi*scipy.special.iv(0,1/sd)
                            for a in A:
                                y += np.exp(np.cos(x-a)/sd)/denom
                                y /= ntotal


                            pmf = -kT*np.log(y) # now we have the PMF

                            # adapted from http://stackoverflow.com/questions/4258106/how-to-calculate-a-fourier-series-in-numpy
                            # complex fourier coefficients
                            def cn(n,y):
                                c = y*np.exp(-1j*n*x)
                                return c.sum()/c.size

                            def ft(x, cn, Nh):
                                f = np.array([2*cn[i]*np.exp(1j*i*x) for i in range(1,Nh+1)])
                                return f.sum()+cn[0]

                            # generate Fourier series (complex)
                            Ns = 6 # needs to be adjusted
                            cf = np.zeros(Ns+1,dtype=complex)
                            for i in range(Ns+1):
                                cf[i] = cn(i,pmf)

                            y1 = np.array([ft(xi,cf,Ns).real for xi in x])  # plot the fourier series approximation.

                            # OK, Fourier series works pretty well.  But we actually want to do a
                            # linear least square fit to a fourier series, since we want to get
                            # the coefficients out.  Let's use the standard LLS formulation with
                            # normal equations.
                            # http://www.math.uconn.edu/~leykekhman/courses/MATH3795/Lectures/Lecture_9_Linear_least_squares_SVD.pdf
                            # basis functions are 1, sin(x), cos(x), sin(2x), cos(2x), . . .
                            Z = np.ones([len(x),2*Ns+1])
                            for i in range(1,Ns+1):
                                Z[:,2*i-1] = np.sin(i*x)
                                Z[:,2*i] = np.cos(i*x)
                            ZM = np.matrix(Z) # easier to manipulate as a matrix
                            [U,S,V] = np.linalg.svd(ZM)    # perform SVD  - S has an interesting shape, is just 1, sqrt(2), sqrt(2). Probably has
                                                           # to do with the normalization.  Still need V and U, though.
                            Sinv = np.matrix(np.zeros(np.shape(Z))).transpose()  # get the inverse of the singular matrix.
                            for i in range(2*Ns+1):
                                Sinv[i,i] = 1/S[i]
                            cm = V.transpose()*Sinv*U.transpose()*np.matrix(pmf).transpose()  # get the linear constants
                            cl = np.array(cm) # cast back to array for plotting
                            # check that it works by plotting
                            y2 = cl[0]*np.ones(len(x))
                            for i in range(1,Ns+1):
                                y2 += cl[2*i-1]*np.sin(i*x)
                                y2 += cl[2*i]*np.cos(i*x)

                            # determine the covariance matrix for the fitting parameters
                            dev = pmf - np.array(ZM*cm).transpose()
                            residuals = np.sum(dev**2)
                            s2 = residuals /(len(pmf) - 2*Ns+1)
                            cov = s2*(V.transpose()*np.linalg.inv(np.diag(S**2))*V)

                            observables_posterior.set_value(ind+1,mol_list[index]+'_'+SMIRKS[inds]+'_'+SMIRKS_ind[index]+'_torsion_pmf_fit_c0',cl[0])
                            observables_posterior.set_value(ind+1,mol_list[index]+'_'+SMIRKS[inds]+'_'+SMIRKS_ind[index]+'_torsion_pmf_fit_c1',cl[1])
                            observables_posterior.set_value(ind+1,mol_list[index]+'_'+SMIRKS[inds]+'_'+SMIRKS_ind[index]+'_torsion_pmf_fit_c2',cl[2])
                            observables_posterior.set_value(ind+1,mol_list[index]+'_'+SMIRKS[inds]+'_'+SMIRKS_ind[index]+'_torsion_pmf_fit_c3',cl[3])
                            observables_posterior.set_value(ind+1,mol_list[index]+'_'+SMIRKS[inds]+'_'+SMIRKS_ind[index]+'_torsion_pmf_fit_c4',cl[4])
                            observables_posterior.set_value(ind+1,mol_list[index]+'_'+SMIRKS[inds]+'_'+SMIRKS_ind[index]+'_torsion_pmf_fit_c5',cl[5])              
        if (ind >= 10) and (ind % 20 == 1):
            try:
                #Make surrogate model and find area of highest probability. Move there. Going to start with a general additive model and see how that works
                obs_names = observables_posterior.columns.values
                obs_data = np.zeros([ind,len(obs_names)],np.float64)
                for name_ind,name in enumerate(obs_names):
                    obs_data[:,name_ind] = observables_posterior[name]
                
                var_names = posterior.columns.values
                var_data = np.zeros([ind,len(var_names)],np.float64)
                for var_ind,var in enumerate(var_names):
                    var_data[:,var_ind] = posterior[var]
                
                def sk_poly_fit(X,y,X_prop):
                # Fit surrogate model for observables with an N-th order polynomial fit. Order to be determined by testing R-sq of many different
                # fits across a range of orders
                    order = []
                    resids = []
                    sum_resid = []
                    Rsq = []
                    for i in range(30):
                        poly = PolynomialFeatures(i) 

                        X_ = poly.fit_transform(X)
                        
                        reg = linear_model.LinearRegression()
                        reg.fit(X_,y)
                        R2 = reg.score(X_,y)
    
                        order.append(i)
                        sum_resid.append(sum(reg.predict(X_) - y))
                        resids.append(reg.predict(X_) - y)
                        Rsq.append(R2)
                   
                    best_fit_index = Rsq.index(max(Rsq))
                    
                    poly = PolynomialFeatures(order[best_fit_index])
                   
                    X_ = poly.fit_transform(X)

                    reg = linear_model.LinearRegression()
                    reg.fit(X_,y)
                    
                    obs_surr = reg.predict(poly.fit_transform(X_prop))
                    return obs_surr

                for obs_ind,obss in enumerate(obs_data):
                    y = obs_data[:,obs_inds]
                    
                    obs_pred = sk_poly_fit(var_data,y,theta_proposal)
             
                    observables_posterior.set_value(ind+1,obs_names[obs_ind],obs_pred)       
            except ValueError:
                print("Oops!  That was no valid number.  Try again...")
    
        # Compute likelihood by multiplying probabilities of each data point
        likelihood_current = np.prod(np.array([1/(np.sqrt(2*np.pi*data[1][j])) * np.exp(-((data[0][j] - 
                             Observables_posterior[ind][obs_name])**2)/(2*data[1][j])) for j,obs_name in enumerate(observables_posterior.columns.values)]))
        likelihood_proposal = np.prod(np.array([1/(np.sqrt(2*np.pi*data[1][j])) * np.exp(-((data[0][j] -
                              Observables_posterior[ind+1][obs_name])**2)/(2*data[1][j])) for j,obs_name in enumerate(observables_posterior.columns.values)]))


        # Compute prior probability of current and proposed mu    
        prior_current = norm(theta_current[0],theta_current[1]).pdf(theta_current[0])  
        prior_proposal = norm(theta_proposal[0],theta_proposal[1]).pdf(theta_proposal[0])
        
        p_current = likelihood_current * prior_current
        p_proposal = likelihood_proposal * prior_proposal
      
        # Accept proposal?
        p_accept = p_proposal / p_current
         
        # Usually would include prior probability, which we neglect here for simplicity
        accept = np.random.rand() < p_accept

        #if plot:
        #    plot_proposal(mu_current, mu_proposal, mu_prior_mu, mu_prior_sd, data, accept, posterior, i)
        
        if accept:
            # Update position
            theta_current = theta_proposal
            #hits.append(1)
            #print "%s out of %s MCMC steps completed. Prior current = %s" % (i,samples,prior_current)
        else:
            hits.append(0)
        posterior.append(theta_current)
        #probs.append(float(likelihood_current*prior_current))
        posterior.to_csv("posterior_"+'_'.join(SMIRKS_and_params)+".csv",sep=";")
        posterior.to_pickle("posterior_"+'_'.join(SMIRKS_and_params)+".pkl")
        observables_posterior.to_csv("observables_posterior_"+'_'.join(SMIRKS_and_params)+".csv",sep=";")
        observables_posterior.to_pickle("observables_posterior_"+'_'.join(SMIRKS_and_params)+".pkl")
        
    #efficiency = float(sum(hits))/float(samples) 
    
    return #posterior,probs

#-----------------------------------------------------------------
data = []
mol_list = ['AlkEthOH_c1143','AlkEthOH_c1163']
theta_init = OrderedDict([('[#6X4:1]-[#1:2]',OrderedDict([('k',500),('length',0.8)])),('[#6X4:1]-[#6X4:2]',OrderedDict([('k',700),('length',1.6)]))]) 
proposal_width = [10,0.01,10,0.01]
samples=10
sampler(data, mol_list, samples, theta_init, proposal_width,'1sim_MBAR_surr',parallel=True)

x = np.array([a[0] for a in posterior])
y = np.array([a[1] for a in posterior])


fig, ax = plt.subplots()
hb = ax.hexbin(x, y, cmap=cm.jet)
ax.axis([625.0, 725.0, 0.95, 1.20])
ax.set_xlabel('Bonded force constant - (kcal/mol/A^2)')
ax.set_ylabel('Equilibrium bond length - (A)')
ax.set_title('Frequency of parameter combinations sampled from posterior distribution')
cb = fig.colorbar(hb, ax=ax)
cb.set_label('Frequency')

plt.savefig('C-H_2D_posterior.png')
#------------------------------------------------------------------
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
