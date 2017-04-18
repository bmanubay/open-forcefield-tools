import matplotlib as mpl

mpl.use('Agg')


import numpy as np
import pandas as pd
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import scipy as sp
import seaborn as sns

from scipy.stats import norm
from scipy.stats import multivariate_normal
import sys

from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('AlkEthOH_c1143_C-H_bl_stats.csv',sep=';')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df.k_values,df.length_values,df.bond_length_average)
plt.savefig('bond_length_average_vs_k_length.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df.k_values,df.length_values,df.bond_length_variance)
plt.savefig('bond_length_variance_vs_k_length.png')
