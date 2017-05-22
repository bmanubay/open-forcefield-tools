import matplotlib as mpl
mpl.use('Agg')

import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.special
import pdb
import pickle

np.set_printoptions(threshold='nan')

num_bins = 100
torsion_r48_a = pickle.load(open( "torsion.p", "rb" ))
plt.figure(1)
(n1,bins1,patch1) = plt.hist(torsion_r48_a, num_bins, label='AlkEthOH_r48 histogram', color='green', normed=1)
n1 = np.array(n1)
n1[n1 == 0] += 1e-10



ntotal = len(torsion_r48_a)
# figure out what the width of the kernel density is. 
# the "rule-of-thumb" estimator used std, but that is for gaussian.  We should use instead
# the stdev of the Gaussian-like features. Playing around with what it looks like, then something 
# like 12 degress as 2 sigma? So sigma is about  degrees = 3/360 * 2*pi = 0.0524
# this gives a relatively smooth PMF, without smoothing too much.
# this will of course depend on the temperature the simulation is run at.

sd = 1.06*0.0524*ntotal**(-0.2)
# create a fine grid
ngrid = 10000
kT = 0.6 # Units of kcal/mol (simulation temp was 300 K)
x = np.arange(-np.pi,np.pi,(2*np.pi)/ngrid)
y = np.zeros(ngrid)
# Easier to use a von Mises distribution than a wrapped Gaussian.
denom = 2*np.pi*scipy.special.iv(0,1/sd)
for a in torsion_r48_a:
    y += np.exp(np.cos(x-a)/sd)/denom
y /= ntotal

plt.plot(x,y,label = 'kernel density estimate (KDE)')
plt.title('comparison between histogram and (KDE)')
plt.xlabel('x (radians)')
plt.ylabel('P(x)')
plt.legend()
plt.savefig('KDE.png')

pmf = -kT*np.log(y)
pmf1 = -kT*np.log(n1) # now we have the PMF
bins1 = np.array(bins1)

#(n1,bins1,patch1) = plt.hist(torsion_r48_a, num_bins, label='AlkEthOH_r48 histogram', color='green', normed=1)
plt.figure()
plt.hist(bins1[1:],len(bins1[1:]),weights=pmf1,label='non-smooth pmf')
plt.plot(x,pmf,label='smooth pmf')
plt.xlabel('x (radians)')
plt.ylabel('Potential of Mean Force (kT)')
plt.legend()
plt.title('Comparison of smoothed and unsmoothed pmf')
plt.savefig('PMF_smooth_vs_nonsmooth.png')

pdb.set_trace()
# adapted from http://stackoverflow.com/questions/4258106/how-to-calculate-a-fourier-series-in-numpy
# complex fourier coefficients
def cn(n,y):
   c = y*np.exp(-1j*n*x)
   return c.sum()/c.size

def ft(x, cn, Nh):
   f = np.array([2*cn[i]*np.exp(1j*i*x) for i in range(1,Nh+1)])
   return f.sum()+cn[0]

# generate Fourier series (complex)
Ns = 12 # needs to be adjusted 
cf = np.zeros(Ns+1,dtype=complex)
for i in range(Ns+1):
    cf[i] = cn(i,pmf)

y1 = np.array([ft(xi,cf,Ns).real for xi in x])  # plot the fourier series approximation.
plt.figure(2)
plt.plot(x,pmf, label='pmf')
plt.plot(x,y1, label='Fourier transform')
plt.title('comparison between PMF and Fourier Transform')
plt.legend()
plt.xlabel('x (radians)')
plt.ylabel('Potential of Mean Force (kT)')
plt.savefig('PMFfitFourier.png')

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

#How different are the coeficients by the two methods?
print "Difference between Fourier series and linear fit to finite Fourier"
print "index   Four   Fit  Diff" 
for i in range(2*Ns+1):
    if i==0:
        cfp = cf[i].real
    elif i%2==0:
        cfp = 2*cf[i/2].real
    elif i%2==1:
        cfp = -2*cf[(i+1)/2].imag
    print "{:3d} {:10.5f} {:10.5f} {:10.5f}".format(i,cfp,float(cl[i]),cfp-float(cl[i]))
print "Looks like they are the same!"

plt.figure(3)
plt.plot(x,pmf,label='pmf')
plt.plot(x,y2,label='LLS fit')
plt.title('Comparison between PMF and linear least squares fit')
plt.xlabel('x (radians)')
plt.ylabel('Potential of Mean Force (kT)')
plt.legend()
plt.savefig('PMFfitLLS.png')

#Compare the LLS and the fourier transform directly.
plt.figure(4)
plt.plot(x,y1,label='Fouier')
plt.plot(x,y2,label='LLS fit')
plt.title('Comparison between Fourier and finite linear least squares fit')
plt.xlabel('x (radians)')
plt.ylabel('Potential of Mean Force (kT)')
plt.legend()
plt.savefig('Fourier_vs_LLS.png')
print "the same!"

# determine the covariance matrix for the fitting parameters
dev = pmf - np.array(ZM*cm).transpose()
residuals = np.sum(dev**2)
s2 = residuals /(len(pmf) - 2*Ns+1)  
cov = s2*(V.transpose()*np.linalg.inv(np.diag(S**2))*V)
print "Covariance matrix is:"
print cov
print "seem to be no nonzero off-diagonal elements! Uncorrelated!  Probably because of orthogonality of Fourier series."

'''
conclusion: there is no correlation, and a linear fit to a discrete
fourier series ends up being the same thing as the first N fourier coefficients
'''
