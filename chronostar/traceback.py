"""This is a test script for tracing back beta Pictoris stars"""

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import pdb
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from galpy.util import bovy_conversion
from error_ellipse import plot_cov_ellipse
import pickle
import time
plt.ion()

def integrate_xyzuvw(params,ts,lsr_orbit,MWPotential2014):
    """Convenience function. Integrates the motion of a star backwards in time.
    
    
    
    Parameters
    ----------
    params: numpy array 
        Kinematic parameters (RAdeg,DEdeg,Plx,pmRA,pmDE,RV)
        
    ts: times for back propagation
        Needs to be converted to galpy units, e.g. for times in Myr:
        ts = -(times/1e3)/bovy_conversion.time_in_Gyr(220.,8.)
        ts = -(np.array([0,10,20])/1e3)/bovy_conversion.time_in_Gyr(220.,8.)
    lsr_orbit:
        WARNING: messy...
        lsr_orbit= Orbit(vxvv=[1.,0,1,0,0.,0],vo=220,ro=8)
        lsr_orbit.integrate(ts,MWPotential2014,method='odeint')
    MWPotential2014:
        WARNING: messy...
        from galpy.potential import MWPotential2014        
    """
    vxvv = params.copy()
    vxvv[2]=1.0/params[2]
    o = Orbit(vxvv=vxvv, radec=True, solarmotion='schoenrich')
    o.integrate(ts,MWPotential2014,method='odeint')
    xyzuvw = np.zeros( (len(ts),6) )
    xyzuvw[:,0] = 1e3*(o.x(ts)-lsr_orbit.x(ts))
    xyzuvw[:,1] = 1e3*(o.y(ts)-lsr_orbit.y(ts))
    xyzuvw[:,2] = 1e3*(o.z(ts))
    xyzuvw[:,3] = o.U(ts) - lsr_orbit.U(ts)
    xyzuvw[:,4] = o.V(ts) - lsr_orbit.V(ts)
    xyzuvw[:,5] = o.W(ts)
    return xyzuvw
 
def withindist(ra2, de2, d2, maxim, ra1 = 53.45111, de1 = 23.37806, d1 = 0.1362):
    """Helper function - Determines if one object is within a certain distance
    of another object.
    
    Parameters
    ----------
    ra2,de2: float
        distance of one object. 
    d2: float
        distance NB: For RAVE - Distances in kpc
    maxim : float
        The distance limit inkpc: anything outside this is not included
        The other object is hard coded into the function for now i.e Pleiades
    """
    x1 = d1*np.cos(np.deg2rad(de1))*np.cos(np.deg2rad(ra1))
    y1 = d1*np.cos(np.deg2rad(de1))*np.sin(np.deg2rad(ra1))
    z1 = d1*np.sin(np.deg2rad(de1))
    x2 = d2*np.cos(np.deg2rad(de2))*np.cos(np.deg2rad(ra2))
    y2 = d2*np.cos(np.deg2rad(de2))*np.sin(np.deg2rad(ra2))
    z2 = d2*np.sin(np.deg2rad(de2))
    d = np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
    return (d <= maxim) and (d >= 0)
    
class TraceBack():
    """A class for tracing back orbits.
    
    This class is initiated with an astropy Table object that includes
    the columns 'RAdeg', 'DEdeg', 'Plx', 'pmRA', 'pmDE', 'RV', 'Name',
    and the uncertainties: 
    'plx_sig', 'pmRA_sig', 'pmDEC_sig', 'RV_sig'. Optionally, include 
    the correlations between parameters: 'c_plx_pmRA', 'c_plx_pmDEC' and
    'c_pmDEC_pmRA'.
    
    Parameters
    ----------
    stars: astropy Table
        Table of star parameters."""

    # A constant - is this in astropy???
    spc_kmMyr = 1.022712165

    def __init__(self, stars, include_cor=False):
        self.nstars = len(stars)
        self.stars = stars
        
    def traceback(self,times,max_plot_error=50,plotit=False, savefile='', dims=[1,2],xoffset=[],yoffset=[],text_ix=[],axis_range=[]):
        """Trace back stellar orbits
    
        Parameters
        ----------
        times: float array
            Times to trace back, in Myr. Note that positive numbers are going backwards in time.
        max_plot_error: float
            Maximum positional error in pc to allow plotting.
        dims: list with 2 ints
            Dimensions to be plotted (out of xyzuvw)
        xoffset, yoffset: nstars long list
            Offsets for text position in star labels.
        """
        nstars = self.nstars
        stars = self.stars
        #Times in Myr
        ts = -(times/1e3)/bovy_conversion.time_in_Gyr(220.,8.)
        nts = len(times)
        
        #Positions and velocities in the co-rotating solar reference frame.
        xyzuvw = np.zeros( (nstars,nts,6) )
        
        #Derivative of the past xyzuvw with respect to the present xyzuvw
        xyzuvw_jac = np.zeros( (nstars,nts,6,6) )
        
        #Past covariance matrices
        xyzuvw_cov = np.zeros( (nstars,nts,6,6) )
        
        #Trace back the local standard of rest.
        lsr_orbit= Orbit(vxvv=[1.,0,1,0,0.,0],vo=220,ro=8)
        lsr_orbit.integrate(ts,MWPotential2014,method='odeint')
        
        #Delta parameters for numerical derivativd
        dp = 1e-3
    
        if plotit:
            plt.clf()

        dim1=dims[0]
        dim2=dims[1]
        cov_ix1 = [[dim1,dim2],[dim1,dim2]]
        cov_ix2 = [[dim1,dim1],[dim2,dim2]]

        axis_titles=['X (pc)','Y (pc)','Z (pc)','U (km/s)','V (km/s)','W (km/s)']
        #Put some defaults in 
        if len(xoffset)==0:
            xoffset = np.zeros(nstars)
            yoffset = np.zeros(nstars)
        if len(text_ix)==0:
            text_ix = range(nstars)
        if len(axis_range)==0:
            axis_range = [-100,100,-100,100]

        for i in range(nstars):
            star = stars[i]
            if 'RAdeg' in star.columns:
                RAdeg = star['RAdeg']
            else:
                RAdeg = star['RAhour']*15.0
            if 'HRV' in star.columns:
                RV = star['HRV']
                e_RV = star['e_HRV']
            else:
                RV = star['RV']
                e_RV = star['e_RV']
            if 'plx' in star.columns:
                Plx = star['plx']
                e_Plx = star['e_plx']
            else:
                e_Plx = star['e_Plx']
                Plx = star['Plx']
            if 'pmRAU4' in star.columns:
                pmRA = star['pmRAU4']
                e_pmRA = star['e_pmRAU4']
            else:
                pmRA = star['pmRA']
                e_pmRA = star['e_pmRA']
            if 'pmDEU4' in star.columns:
                pmDE = star['pmDEU4']
                e_pmDE = star['e_pmDEU4']  
            else:
                pmDE = star['pmDE']
                e_pmDE = star['e_pmDE']
            params = np.array([RAdeg,star['DEdeg'],Plx,pmRA,pmDE,RV])
            xyzuvw[i] = integrate_xyzuvw(params,ts,lsr_orbit,MWPotential2014)
            
            #Create numerical derivatives
            for j in range(6):
                params_plus = params.copy()
                params_plus[j] += dp
                xyzuvw_plus = integrate_xyzuvw(params_plus,ts,lsr_orbit,MWPotential2014)
                for k in range(nts):
                    xyzuvw_jac[i,k,j,:] = (xyzuvw_plus[k] - xyzuvw[i,k])/dp 
                    
            #Now that we've got the jacobian, find the modified covariance matrix.
            cov_obs = np.zeros( (6,6) )
            cov_obs[0,0] = 1e-1**2 #Nominal 0.1 degree. !!! There is almost a floating underflow in fit_group due to this
            cov_obs[1,1] = 1e-1**2 #Nominal 0.1 degree. !!! Need to think about this more
            cov_obs[2,2] = e_Plx**2
            cov_obs[3,3] = e_pmRA**2
            cov_obs[4,4] = e_pmDE**2
            cov_obs[5,5] = e_RV**2
            
            #Create the covariance matrix from the Jacobian. See e.g.:
            #https://en.wikipedia.org/wiki/Covariance#A_more_general_identity_for_covariance_matrices
            #Think of the Jacobian as just a linear transformation between the present and the past in
            #"local" coordinates. 
            for k in range(nts):
                xyzuvw_cov[i,k] = np.dot(np.dot(xyzuvw_jac[i,k].T,cov_obs),xyzuvw_jac[i,k])
                
            #Plot beginning and end points, plus their the uncertainties from the
            #covariance matrix.
        #    plt.plot(xyzuvw[i,0,0],xyzuvw[i,0,1],'go')
        #    plt.plot(xyzuvw[i,-1,0],xyzuvw[i,-1,1],'ro')
            #Only plot if uncertainties are low...
            if plotit:
                cov_end = xyzuvw_cov[i,-1,cov_ix1,cov_ix2]
                if (np.sqrt(cov_end.trace()) < max_plot_error):
                    if i in text_ix:
                        plt.text(xyzuvw[i,0,dim1]*1.1 + xoffset[i],xyzuvw[i,0,dim2]*1.1 + yoffset[i],star['Name'],fontsize=11)
                    plt.plot(xyzuvw[i,:,dim1],xyzuvw[i,:,dim2],'b-')
                    plot_cov_ellipse(xyzuvw_cov[i,0,cov_ix1,cov_ix2], [xyzuvw[i,0,dim1],xyzuvw[i,0,dim2]],color='g',alpha=1)
                    plot_cov_ellipse(cov_end, [xyzuvw[i,-1,dim1],xyzuvw[i,-1,dim2]],color='r',alpha=0.2)
        
        if plotit:            
            plt.xlabel(axis_titles[dim1])
            plt.ylabel(axis_titles[dim2])
            plt.axis(axis_range)

        if len(savefile)>0:
            fp = open(savefile,'w')
            pickle.dump((stars,times,xyzuvw,xyzuvw_cov),fp)
            fp.close()

def traceback2(params,times):
    """Trace forward a cluster. First column of returned array is the position of the cluster at a given age.

    Parameters
    ----------
    times: float array
        Times to trace forward, in Myr. Note that positive numbers are going forward in time.
    params: float array
        xyzuvw for a cluster
    age: Age of cluster, in Myr
    """
    #Times in Myr
    ts = -(times/1e3)/bovy_conversion.time_in_Gyr(220.,8.)
    nts = len(times)

    #Positions and velocities in the co-rotating solar reference frame.
    xyzuvw = np.zeros( (1,nts,6) )

    #Trace forward the local standard of rest.
    lsr_orbit= Orbit(vxvv=[1.,0,1,0,0.,0],vo=220,ro=8)
    lsr_orbit.integrate(ts,MWPotential2014,method='odeint')

    xyzuvw = integrate_xyzuvw(params,ts,lsr_orbit,MWPotential2014)
    return xyzuvw

#Instead of a __main__ block below, please use test routines in the main directory.
