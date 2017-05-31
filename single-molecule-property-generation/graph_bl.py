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
from scipy.stats import uniform
from scipy.stats import multivariate_normal
import sys

from mpl_toolkits.mplot3d import Axes3D

import itertools
from matplotlib import cm
from matplotlib.colors import LightSource
import pdb

from sklearn.utils import resample

df = pd.read_csv('AlkEthOH_c1143_C-H_bl_stats.csv',sep=';')

points_av = df.as_matrix(columns=['k_values','length_values','bond_length_average'])
points_var = df.as_matrix(columns=['k_values','length_values','bond_length_variance'])
points_var_var = df.as_matrix(columns=['k_values','length_values','bond_length_variance_variance'])

def poly_matrix(x, y, order=2):
    """ generate Matrix use with lstsq """
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    print ij
    for k, (i, j) in enumerate(ij):
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

x_var_var, y_var_var, z_var_var = points_var_var.T

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

zz_av_comp = m_av[0] + m_av[1]*yy + m_av[2]*yy**2 + m_av[3]*xx + m_av[4]*xx*yy + m_av[5]*xx*(yy**2) + m_av[6]*xx**2 + m_av[7]*(xx**2)*yy + m_av[8]*(xx**2)*(yy**2)

zz_var_comp = m_var[0] + m_var[1]*yy + m_var[2]*yy**2 + m_var[3]*xx + m_var[4]*xx*yy + m_var[5]*xx*(yy**2) + m_var[6]*xx**2 + m_var[7]*(xx**2)*yy + m_var[8]*(xx**2)*(yy**2)

zz_av_comp_boots = []
zz_var_comp_boots = []
# Bootstrap regression for error bars
nBoots_work = 200
for n in range(nBoots_work):
   if (n == 0):
       booti_av = zz_av_comp
       booti_var = zz_var_comp
   else:
       booti_av = resample(zz_av_comp)
       booti_var = resample(zz_var_comp)
   zz_av_comp_boots.append(booti_av)
   zz_var_comp_boots.append(booti_var)

zz_av_comp_boots = np.array(zz_av_comp_boots)
zz_var_comp_boots = np.array(zz_var_comp_boots)

zz_av_unc = np.apply_along_axis(np.var,0,zz_av_comp_boots)
zz_var_unc = np.apply_along_axis(np.var,0,zz_var_comp_boots)

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

for i in np.arange(0, len(x_av)):
    ax.plot([x_av[i],x_av[i]], [y_av[i],y_av[i]], [z_av[i]-z_var[i], z_av[i]+z_var[i]], marker="_")

xx_arr = np.vstack(xx.flatten()).T[0]
yy_arr = np.vstack(yy.flatten()).T[0]
zz_av_comp_arr = np.vstack(zz_av_comp.flatten()).T[0]
zz_av_unc_arr = np.vstack(zz_av_unc.flatten()).T[0]
zz_var_comp_arr = np.vstack(zz_var_comp.flatten()).T[0]
zz_var_unc_arr = np.vstack(zz_var_unc.flatten()).T[0]


for i in np.arange(0, len(xx_arr)):
    ax.plot([xx_arr[i],xx_arr[i]], [yy_arr[i],yy_arr[i]], zs=[zz_av_comp_arr[i]-zz_av_unc_arr[i],zz_av_comp_arr[i]+zz_av_unc_arr[i]],marker="_")

fg.canvas.draw()
plt.savefig('bond_length_average_vs_k_length_w_fit.png')

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

for i in np.arange(0, len(x_av)):
    ax.plot([x_var[i],x_var[i]], [y_var[i],y_var[i]], [z_var[i]-z_var_var[i], z_var[i]+z_var_var[i]], marker="_")

xx_arr = np.vstack(xx.flatten()).T[0]
yy_arr = np.vstack(yy.flatten()).T[0]
zz_av_comp_arr = np.vstack(zz_av_comp.flatten()).T[0]
zz_av_unc_arr = np.vstack(zz_av_unc.flatten()).T[0]
zz_var_comp_arr = np.vstack(zz_var_comp.flatten()).T[0]
zz_var_unc_arr = np.vstack(zz_var_unc.flatten()).T[0]


for i in np.arange(0, len(xx_arr)):
    ax.plot([xx_arr[i],xx_arr[i]], [yy_arr[i],yy_arr[i]], zs=[zz_var_comp_arr[i]-zz_var_unc_arr[i],zz_var_comp_arr[i]+zz_var_unc_arr[i]],marker="_")

fg.canvas.draw()
plt.savefig('bond_length_variance_vs_k_length_w_fit.png')

x_av,res_av,rank_av,s_av = np.linalg.lstsq(G, z_av)
x_var,res_var,rank_var,s_var = np.linalg.lstsq(G, z_var)

print x_av
print x_var

print rank_av
print rank_var

print res_av
print res_var
sys.exit()
def sampler(data, samples=4, theta_init=[500,0.8], proposal_width=[10,0.05], plot=False, mu_prior_mu=0, mu_prior_sd=1.):
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
    
    theta_current = theta_init
    posterior = [theta_current]
    probs = [np.random.rand()]
    hits = []
    for i in range(samples):
        # suggest new position
        theta_proposal = [norm(theta_current[j],proposal_width[j]).rvs() for j in range(len(theta_current))]    
        

        # Compute observables at proposed theta with surrgates
        O_av_comp_curr = m_av[0] + m_av[1]*theta_current[1] + m_av[2]*theta_current[1]**2 + m_av[3]*theta_current[0] +\
                     m_av[4]*theta_current[0]*theta_current[1] + m_av[5]*theta_current[0]*(theta_current[1]**2) +\
                     m_av[6]*theta_current[0]**2 + m_av[7]*(theta_current[0]**2)*theta_current[1] +\
                     m_av[8]*(theta_current[0]**2)*(theta_current[1]**2)

        O_var_comp_curr = m_var[0] + m_var[1]*theta_current[1] + m_var[2]*theta_current[1]**2 + m_var[3]*theta_current[0] +\
                     m_var[4]*theta_current[0]*theta_current[1] + m_var[5]*theta_current[0]*(theta_current[1]**2) +\
                     m_var[6]*theta_current[0]**2 + m_var[7]*(theta_current[0]**2)*theta_current[1] +\
                     m_var[8]*(theta_current[0]**2)*(theta_current[1]**2)

        O_comp_curr = [O_av_comp_curr,O_var_comp_curr]
 

        O_av_comp_prop = m_av[0] + m_av[1]*theta_proposal[1] + m_av[2]*theta_proposal[1]**2 + m_av[3]*theta_proposal[0] +\
                     m_av[4]*theta_proposal[0]*theta_proposal[1] + m_av[5]*theta_proposal[0]*(theta_proposal[1]**2) +\
                     m_av[6]*theta_proposal[0]**2 + m_av[7]*(theta_proposal[0]**2)*theta_proposal[1] +\
                     m_av[8]*(theta_proposal[0]**2)*(theta_proposal[1]**2)

        O_var_comp_prop = m_var[0] + m_var[1]*theta_proposal[1] + m_var[2]*theta_proposal[1]**2 + m_var[3]*theta_proposal[0] + \
                     m_var[4]*theta_proposal[0]*theta_proposal[1] + m_var[5]*theta_proposal[0]*(theta_proposal[1]**2) + \
                     m_var[6]*theta_proposal[0]**2 + m_var[7]*(theta_proposal[0]**2)*theta_proposal[1] + \
                     m_var[8]*(theta_proposal[0]**2)*(theta_proposal[1]**2)         
        
        O_comp_prop = [O_av_comp_prop,O_var_comp_prop]

    
        # Compute likelihood by multiplying probabilities of each data point
        likelihood_current = np.prod(np.array([1/(np.sqrt(2*np.pi*data[1][j])) * np.exp(- ((data[0][j] - O_comp_curr[j])**2)/(2*data[1][j])) 
                             for j in range(len(data))]))
        likelihood_proposal = np.prod(np.array([1/(np.sqrt(2*np.pi*data[1][j])) * np.exp(- ((data[0][j] - O_comp_prop[j])**2)/(2*data[1][j]))
                              for j in range(len(data))]))


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
            hits.append(1)
            print "%s out of %s MCMC steps completed. Prior current = %s" % (i,samples,prior_current)
        else:
            hits.append(0)
        posterior.append(theta_current)
        probs.append(float(likelihood_current*prior_current))
    efficiency = float(sum(hits))/float(samples) 
    print efficiency
    return posterior,probs
posterior,probs = sampler([[1.0920405895833334,0.00090201196735599997],[0.00090201196735599997,2.8009246152166006e-10]],samples=1000000)

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
