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

import itertools
from matplotlib import cm
from matplotlib.colors import LightSource

df = pd.read_csv('AlkEthOH_c1143_C-H_bl_stats.csv',sep=';')

points_av = df.as_matrix(columns=['k_values','length_values','bond_length_average'])
points_var = df.as_matrix(columns=['k_values','length_values','bond_length_variance'])

def poly_matrix(x, y, order=2):
    """ generate Matrix use with lstsq """
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i, j) in enumerate(ij):
        G[:, k] = x**i * y**j
    return G

ordr = 2  # order of polynomial
x_av, y_av, z_av = points_av.T
x_av, y_av = x_av - x_av[0], y_av - y_av[0]  # this improves accuracy

x_var, y_var, z_var = points_var.T
x_var, y_var = x_var - x_var[0], y_var - y_var[0]  # this improves accuracy



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
surf = ax.plot_surface(xx, yy, zz_av, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)
ax.plot3D(x_av, y_av, z_av, "o")

fg.canvas.draw()
plt.savefig('bond_length_average_vs_k_length_w_fit.png')

# Plotting (see http://matplotlib.org/examples/mplot3d/custom_shaded_3d_surface.html):
fg, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ls = LightSource(270, 45)
rgb = ls.shade(zz_var, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
surf = ax.plot_surface(xx, yy, zz_var, rstride=1, cstride=1, facecolors=rgb,
                       linewidth=0, antialiased=False, shade=False)
ax.plot3D(x_var, y_var, z_var, "o")

fg.canvas.draw()
plt.savefig('bond_length_variance_vs_k_length_w_fit.png')

sys.exit()

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

print grid_x
print z1


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
