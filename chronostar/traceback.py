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

# A constant - is this in astropy???
spc_kmMyr = 1.022712165

#Read in numbers for beta Pic. For HIPPARCOS, we have *no* radial velocities in general.
#
bp=Table.read('betaPic.csv')
#Remove bad stars. "bp" stands for Beta Pictoris.
bp = bp[np.where([ (n.find('6070')<0) & (n.find('12545')<0) & (n.find('Tel')<0) for n in bp['Name']])[0]]
nstars = len(bp)
nts = 201
error_lim=50
saveit=False
ts = np.linspace(0,-1/bovy_conversion.time_in_Gyr(220.,8.),nts)

#Times in Myr
times = -ts*bovy_conversion.time_in_Gyr(220.,8.)*1e3

#The next two lines are actually in the co-rotating solar reference frame.
xyzuvw = np.zeros( (nstars,nts,6) )
xyzuvw_jac = np.zeros( (nstars,nts,6,6) )
xyzuvw_cov = np.zeros( (nstars,nts,6,6) )

plt.clf()

lsr_orbit= Orbit(vxvv=[1.,0,1,0,0.,0],vo=220,ro=8)
lsr_orbit.integrate(ts,MWPotential2014,method='odeint')

dp = 1e-3

dim1=1
dim2=2
cov_ix1 = [[dim1,dim2],[dim1,dim2]]
cov_ix2 = [[dim1,dim1],[dim2,dim2]]

axis_range = [-100,100,-100,100]

def integrate_xyzuvw(params,ts,lsr_orbit,MWPotential2014):
    """Convenience function"""
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
    
axis_titles=['X (pc)','Y (pc)','Z (pc)','U (km/s)','V (km/s)','W (km/s)']
xoffset = np.zeros(nstars)
yoffset = np.zeros(nstars)
text_ix = range(nstars)

if (dim1==0) & (dim2==1):
    yoffset[0:10] = [6,-8,-6,2,0,-4,0,0,0,-4]
    yoffset[10:] = [0,-8,0,0,6,-6,0,0,0]
    xoffset[10:] = [0,-4,0,0,-15,-10,0,0,-20]
    axis_range = [-70,60,-40,120]
    
if (dim1==1) & (dim2==2):
    axis_range = [-40,120,-30,100]
    text_ix = [0,1,4,7]
    xoffset[7]=-15

tic=time.time()
for i in range(nstars):
    star = bp[i]
    params = np.array([star['Radeg'],star['Dedeg'],star['plx'],star['pmRA'],star['pmDEC'],star['RV']])
    xyzuvw[i] = integrate_xyzuvw(params,ts,lsr_orbit,MWPotential2014)
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
    cov_obs[2,2] = star['plx_sig']**2
    cov_obs[3,3] = star['pmRA_sig']**2
    cov_obs[4,4] = star['pmDEC_sig']**2
    cov_obs[5,5] = star['RV_sig']**2
    for k in range(nts):
        xyzuvw_cov[i,k] = np.dot(np.dot(xyzuvw_jac[i,k].T,cov_obs),xyzuvw_jac[i,k])
    #Plot beginning and end points, plus their the uncertainties from the
    #covariance matrix.
#    plt.plot(xyzuvw[i,0,0],xyzuvw[i,0,1],'go')
#    plt.plot(xyzuvw[i,-1,0],xyzuvw[i,-1,1],'ro')
    #Only plot if uncertainties are low...
    cov_end = xyzuvw_cov[i,-1,cov_ix1,cov_ix2]
    if (np.sqrt(cov_end.trace()) < error_lim):
        if i in text_ix:
            plt.text(xyzuvw[i,0,dim1]*1.1 + xoffset[i],xyzuvw[i,0,dim2]*1.1 + yoffset[i],star['Name'],fontsize=11)
        plt.plot(xyzuvw[i,:,dim1],xyzuvw[i,:,dim2],'b-')
        plot_cov_ellipse(xyzuvw_cov[i,0,cov_ix1,cov_ix2], [xyzuvw[i,0,dim1],xyzuvw[i,0,dim2]],color='g',alpha=1)
        plot_cov_ellipse(cov_end, [xyzuvw[i,-1,dim1],xyzuvw[i,-1,dim2]],color='r',alpha=0.2)
#    w,v = np.linalg.eig(cov_xz)    
#    pdb.set_trace()
print(time.time()-tic)
    
plt.xlabel(axis_titles[dim1])
plt.ylabel(axis_titles[dim2])
plt.axis(axis_range)

xyz_cov = np.array([[  34.25840977,   35.33697325,   56.24666544],
       [  35.33697325,   46.18069795,   66.76389275],
       [  56.24666544,   66.76389275,  109.98883853]])
xyz = [ -6.221, 63.288, 23.408]
plot_cov_ellipse(xyz_cov[cov_ix1,cov_ix2],[xyz[dim1],xyz[dim2]],alpha=0.5,color='k')

#[ -6.221, 63.288, 23.408, -0.853,-11.392, -6.570,  9.884,  8.434, 10.356,  0.697,  0.938,  0.886,  0.871, 19.833]
#
if saveit:
    fp = open('traceback_save.pkl','w')
    pickle.dump((bp,times,xyzuvw,xyzuvw_cov),fp)
    fp.close()