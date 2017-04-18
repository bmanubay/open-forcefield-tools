import matplotlib as mpl

mpl.use('Agg')


import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import scipy as sp
from scipy.interpolate import griddata
import seaborn as sns

from scipy.stats import norm
from scipy.stats import multivariate_normal
import sys

from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('AlkEthOH_c1143_C-H_bl_stats.csv',sep=';')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df.k_values,df.length_values,df.bond_length_average)
ax.set_xlabel('Bonded force constant - (kcal/mol/A^2)')
ax.set_ylabel('Equilibrium bond length - (A)')
ax.set_zlabel('Average of bond length distribution - (A)')
plt.savefig('bond_length_average_vs_k_length.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df.k_values,df.length_values,df.bond_length_variance)
ax.set_xlabel('Bonded force constant - (kcal/mol/A^2)')
ax.set_ylabel('Equilibrium bond length - (A)')
ax.set_zlabel('Variance of bond length distribution - (A^2)')
plt.savefig('bond_length_variance_vs_k_length.png')


grid_x, grid_y = np.mgrid[0:1200:100j, 0:1.75:200j]

z1 = griddata([[i,j] for i,j in zip(df.k_values,df.length_values)], df.bond_length_average, (grid_x, grid_y), method='cubic')
z2 = griddata([[i,j] for i,j in zip(df.k_values,df.length_values)], df.bond_length_variance, (grid_x, grid_y), method='cubic')

plt.figure()
CS = plt.contour(grid_x,grid_y,z1)
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel('Bonded force constant - (kcal/mol/A^2)')
plt.ylabel('Equilibrium bond length - (A)')
plt.title('Average of bond length distribution vs k_Bond and x_0 - (A)')
plt.savefig('bond_length_average_vs_k_length_contour.png')

plt.figure()
CS = plt.contour(grid_x,grid_y,z2)
plt.clabel(CS, inline=1, fontsize=10)
plt.xlabel('Bonded force constant - (kcal/mol/A^2)')
plt.ylabel('Equilibrium bond length - (A)')
plt.title('Variance of bond length distribution vs k_Bond and x_0 - (A^2)')
plt.savefig('bond_length_variance_vs_k_length_contour.png')
