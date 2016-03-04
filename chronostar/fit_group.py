"""This program takes an initial model for a stellar association and uses an affine invariant
Monte-Carlo to fit for the group parameters."""

from __future__ import print_function, division

import emcee
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb
plt.ion()

fp = open('traceback_save.pkl','r')
(bp,times,xyzuvw,xyzuvw_cov)=pickle.load(fp)
fp.close()

#Preliminaries. 
#Create the inverse covariances and other globals.
ns = len(bp)
nt = len(times)
xyzuvw_icov = np.empty( (ns,nt,6,6) )
xyzuvw_icov_det = np.empty( (ns,nt) )
for i in range(ns):
    for j in range(nt):
        xyzuvw_icov[i,j] = np.linalg.inv(xyzuvw_cov[i,j])
#        xyzuvw_icov_det[i,j] = np.linalg.det(xyzuvw_icov[i,j])

def lnprob_group(x, background_density=2e-12,t_ix = 12,return_overlaps=False,return_cov=False, min_axis=2.0,min_v_disp=0.5):
    """The x variables are:
    xyzuvw, then xyzstd, uvw_symmetrical_std, xyz_correlations 
    
    The probability of a model is the product of the probabilities
    overlaps of every star in the group. 
    
    Parameters
    ----------
    background_density:
        The density of a background stellar population, in
        units of pc**(-3)*(km/s)**(-3). 
    
    """
    #See if we have to interpolate in time.
    if len(x)>13:
        if ( (x[13] < 0) | (x[13] >= nt-1)):
            return -np.inf
        ix = np.interp(x[13],times,np.arange(nt))
        ix0 = np.int(ix)
        frac = ix-ix0
        bs     = xyzuvw[:,ix0]*(1-frac) + xyzuvw[:,ix0+1]*frac
        cov    = xyzuvw_cov[:,ix0]*(1-frac) + xyzuvw_cov[:,ix0+1]*frac
        #Bs     = xyzuvw_icov[:,ix0]*(1-frac) + xyzuvw_icov[:,ix0+1]*frac
        Bs     = np.linalg.inv(cov)
        #B_dets = xyzuvw_icov_det[:,ix0]*(1-frac) + xyzuvw_icov_det[:,ix0+1]*frac
        #pdb.set_trace()
    else:
        bs     = xyzuvw[:,t_ix]
        Bs     = xyzuvw_icov[:,t_ix]
#        B_dets = xyzuvw_icov_det[:,t_ix]
    
    #Sanity check inputs for out of bounds...
    if (np.min(x[6:9])<=min_axis):
        return -np.inf
    if (np.min(x[9])<min_v_disp):
        return -np.inf
    if (np.max(np.abs(x[10:13])) >= 1):
        return -np.inf
        
        
    #Create the group_mn and group_cov from x.
    x = np.array(x)
    group_mn = x[0:6]
    group_cov = np.eye( 6 )
    group_cov[np.tril_indices(3,-1)] = x[10:13]
    group_cov[np.triu_indices(3,1)] = x[10:13]
    for i in range(3):
        group_cov[i,:3] *= x[6:9]
        group_cov[:3,i] *= x[6:9]
    for i in range(3,6):
        group_cov[i,3:] *= x[9]
        group_cov[3:,i] *= x[9]

    #Allow this covariance matrix to be returned.
    if return_cov:
        return group_cov

    #Enforce some sanity check limits on prior...
    if (np.min(np.linalg.eigvalsh(group_cov[:3,:3])) < min_axis**2):
        return -np.inf

    #Invert the group covariance matrix and check for negative eigenvalues
    group_icov = np.linalg.inv(group_cov)
    group_icov_eig = np.linalg.eigvalsh(group_icov)
    if np.min(group_icov_eig) < 0:
        return -np.inf
    group_icov_det = np.prod(group_icov_eig)
        

    #Before starting, lets set the prior probability
    #Given the way we're sampling the covariance matrix, I'm
    #really not sure this is correct! But it is pretty close...
    #it looks almost like 1/(product of standard deviations).
    #See YangBerger1998
    lnprob=np.log(np.abs(group_icov_det)**3.5)
    
    overlaps = np.zeros(ns)
    
    #Now loop through stars. 
    for i in range(ns):
        #Compute the integral, including some temporary variables
        #for speed and to match the notes.
        ApB = Bs[i] + group_icov
        AapBb = np.dot(Bs[i],bs[i]) + np.dot(group_icov,group_mn)
        B_det = np.linalg.det(Bs[i])
        ApB_det = np.linalg.det(ApB)
        if (ApB_det < 0) | (B_det<0):
            #!!!This shouldn't ever happen, as the determinants of the sum of positive definite matrices is
            #greater than the sum of their determinants
            pdb.set_trace()
            return -np.inf
        c = np.linalg.solve(ApB, AapBb)
        overlaps[i] = np.exp(-0.5*(np.dot(bs[i]-c,np.dot(Bs[i],bs[i]-c)) + \
            np.dot(group_mn-c,np.dot(group_icov,group_mn-c)))) 
        overlaps[i] *= np.sqrt(B_det*group_icov_det/ApB_det)/(2*np.pi)**3.0
        #print(overlap)
        lnprob += np.log(background_density + overlaps[i])
    #print(lnprob)
    if return_overlaps:
        return overlaps    

    return lnprob
    
nparams = 14
init_mod = np.array([0,30,30,-11,-16,-9,25,25,25,2,0,0,0])
init_mod = np.array([0,30,30,-1,-12,-9,25,25,25,2,0,0,0])
init_mod = np.array([-5.69, 38.4,  23.86, -1.05,-11.71, -7.55,  11, 10,11,1.0,0.3,-0.4,-0.1,12])
init_mod = np.array([ -8.273, 56.576, 21.445, -1.376,-11.310, -6.576, 10.967,  8.535, 11.964,  0.814,  0.913,  0.631,  0.806, 18.552])
init_mod = np.array([ -6.574, 66.560, 23.436, -1.327,-11.427, -6.527, 10.045, 10.319, 12.334,  0.762,  0.932,  0.735,  0.846, 20.589])

#[ -0.16,29.418,32.587,-10.00,-13.7, -8.61,25.926,24.28,24.84,-0.043, -2.0809177  , -0.63003765 ,  2.29294306]
#init_mod = [ -0.1993,25.8383,16.58,0.993,-2.6,-10.0,28.21,24.2,34.14,5.26,-19.22,-23.0,-15.38]
#init_mod = np.array([-28.345, 27.407, 25.891, -0.557,-11.500, -7.225, 17.716, 13.553, 10.122,  0.001,  0.523,  0.175,  0.029])

#test = lnprob_group(init_mod)
#pdb.set_trace()

ndim, nwalkers = nparams, 200
nchain = 5000
nburn  = 1000

p0 = [init_mod + np.random.random(size=ndim) - 0.5 for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_group)

#Burn in...
pos, prob, state = sampler.run_mcmc(p0, nburn)
sampler.reset()

#Run...
sampler.run_mcmc(pos, nchain)
plt.figure(1)
plt.clf()
plt.plot(sampler.lnprobability.T)

#Best Model
best_ix = np.argmax(sampler.flatlnprobability)
print('[' + ",".join(["{0:7.3f}".format(f) for f in sampler.flatchain[best_ix]]) + ']')
overlaps = lnprob_group(sampler.flatchain[best_ix],return_overlaps=True)
group_cov = lnprob_group(sampler.flatchain[best_ix],return_cov=True)
np.sqrt(np.linalg.eigvalsh(group_cov[:3,:3]))
ww = np.where(overlaps < 2e-12)[0]
print(bp[ww]['Name'])

print("Mean acceptance fraction: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))

plt.figure(2)       
plt.clf()         
plt.hist(sampler.chain[:,:,-1].flatten(),20)