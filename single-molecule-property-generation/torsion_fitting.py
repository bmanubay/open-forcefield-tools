import matplotlib as mpl

mpl.use('Agg')

import math
import scipy.optimize as sci
import matplotlib.pyplot as plt
import numpy as np
import openeye
from openeye import oechem
from smarty import *
from smarty.utils import *
from smarty.forcefield import *
from simtk import unit
import pandas as pd
import netCDF4 as netcdf
import sys
import pdb

import pymbar # multistate Bennett acceptance ratio analysis (provided by pymbar)
from pymbar import timeseries # timeseries analysis (provided by pymbar)
import commands
import os
import os.path
import time

#-------------------------------------------------------
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
    inputs: the xyz files, an array of length-2 arrays.
    we calculate all three together since the torsions and angles
    require the bond vectors to be calculated anyway.
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
#------------------------------------------------------------------

def readtraj(ncfiles):

    """
    Take multiple .nc files and read in coordinates in order to re-valuate energies based on parameter changes

    ARGUMENTS
    ncfiles - a list of trajectories in netcdf format
    """

    for fname in ncfiles:
        data = netcdf.Dataset(fname)
        xyz = data.variables['coordinates']
        time = data.variables['time']

    return data, xyz, time 
#------------------------------------------------------------------
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
    print "You have specified a search for %s SMIRKS strings" %(obstype)
    print "Other SMIRKS strings of this type in this molecule include:"
    for b,k in enumerate(myset):
        print "%s which occurs %s times" %(k, mylist.count(k))
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
#------------------------------------------------------------------
# Function handle for the fourier series fit of the torsion pdf
P = 4.*np.pi/3.
def fourier(x, *a):
    ret =  (a[0] * np.sin(2*np.pi*x / a[1] + a[2]))**2 + \
           (a[3] * np.sin(2*np.pi*x / a[4] + a[5]))**2 + \
           (a[6] * np.sin(2*np.pi*x / a[7] + a[8]))**2 # \
           #[9]
           #a[6] * (np.sin(2*np.pi*x / a[7] + phase))**2 + \
           #a[8] * (np.sin(2*np.pi*x / a[9] + phase))**2 + \
           #a[10] * (np.sin(2*np.pi*x / a[11] + phase))**2
          
    #for deg in range(2, len(a)/2):
    #    ret += a[2*deg] * np.sin((deg) * 2*np.pi*x / tau + a[2*deg-1])
    return ret
#-----------------------------------------------------------------
ff = ForceField(get_data_filename('/data/forcefield/smirff99Frosst.ffxml'))
mol2 = ['Mol2_files/AlkEthOH_r48.mol2', 'Mol2_files/AlkEthOH_r51.mol2']
traj = ['traj/AlkEthOH_r48_50ns.nc', 'traj/AlkEthOH_r51_50ns.nc']

AtomDict_r48,lst0,lst1,lst2 = get_small_mol_dict([mol2[0]])

smirks = '[#1:1]-[#6X4:2]-[#6X4:3]-[#6X4:4]'

match_inds = find_smirks_instance(mol2[0],smirks)

data_r48, xyz_r48, time_r48 = readtraj([traj[0]])

xyzn_r48 = unit.Quantity(xyz_r48[:], unit.angstroms)

# Compute bond lengths and angles and return array of angles
a = ComputeBondsAnglesTorsions(xyzn_r48,AtomDict_r48['Bond'],AtomDict_r48['Angle'],AtomDict_r48['Torsion'])

# Pull out torsion data
torsions_r48 = a[2]

# Get number of angles in molecule
numtors_r48 = len(torsions_r48[0])

# Re-organize data into timeseries
torstimeser_r48 = [torsions_r48[:,ind] for ind in range(numtors_r48)]

num_bins = 100

match_inds = match_inds[0:2]

fourier_coeff = []
for i in match_inds:
    plt.figure()
    torsion_r48_a = torstimeser_r48[i]
    (n1,bins1,patch1) = plt.hist(torsion_r48_a,num_bins,label='AlkEthOH_r48 histogram',color='green')
    
    # estimate fourier coefficients using scipy's curve_fit function (uses linear least squares by default)
    popt, pcov = sci.curve_fit(fourier, bins1[1:], n1, [100.0,3.0,1.0]*3 , maxfev=100000)
    fourier_coeff.append([popt,pcov,i])

    plt.plot(bins1[1:],fourier(bins1[1:],*popt),label='AlkEthOH_r48 fourier fit')
    plt.ylabel('Number of times configuration is sampled')
    plt.xlabel('Torsion angle (radians)')
    plt.title('Torsion sample in AlkEthOH_r48')
    plt.legend()
    plt.savefig('torsion_histograms/r48/Torsion_likelihood_r48_tors_'+smirks+'_'+str(i)+'.png')

print "Done"

#--------------------------------------------------------------------------
# Fourier series fits to the PMF calculations based on the torsion probability densities
#--------------------------------------------------------------------------
# Calculation methods:
# 1) Standard equation: g(r) = exp(-beta*F(r))
#                       where g(r) is the radial distribution function, beta is 1/(kB*T) and F(r) is the PMF
#                       Thus:
#                            F(r) = -kB*T*ln(g(r))
# Thought: Since g(r) is just a density function could I just use the torsion density function instead?

# 2) Using MBAR: found a script that John made for calculating PMF a la the WHAM method (but translated into MBAR) 
#                
#              Link: https://github.com/choderalab/pymbar/blob/master/examples/constant-force-optical-trap/force-bias-optical-trap.py

