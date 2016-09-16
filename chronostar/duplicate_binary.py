# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 00:39:03 2016

@author: jhansen
"""

from astropy.table import Table
import numpy as np

#Read in table
t = Table.read('filename.csv', format = 'ascii')
#Sort table according to RA and DE
t.sort(['RAdeg', 'DEdeg'])

#Choose the maximum difference in angular position for binary stars (deg)
bin_dist = 0.01
arcs = 0.000278

new_t=t
i=0

while i < len(t):
    d_b = []
    l = 1
    
    #Determines if a star and the next in the table are within an arcsec of 
    #each other. That is, they are identical.
    if abs(t['RAdeg'][i+1] - t['RAdeg'][i]) < arcs and abs(t['DEdeg'][i+1] - t['DEdeg'][i]) < arcs:
        d_b = np.append(d_b,t[i:i+2], axis = 0)
        n= i+1
        #Searches for other identical stars
        while abs(t['RAdeg'][n+1] - t['RAdeg'][n]) < arcs and abs(t['DEdeg'][n+1] - t['DEdeg'][n]) < arcs:
            d_b = np.append(d_b,t[n+1], axis = 0)
            n += 1
        l = len(d_b)
        
        #Averages the coordinates and finds the weighted average RV
        new_t['RA'][i] = np.mean(d_b['RAdeg'])
        new_t['DE'][i] = np.mean(d_b['DEdeg'])
        sum_w = np.sum(1/(d_b['RVsig'])**2)
        new_t['RV'][i] = np.sum(d_b['RV']*(1/(d_b['RVsig'])**2))/sum_w
        new_t['RVsig'][i] = 1/np.sqrt(sum_w)
        new_t = np.delete(new_t, range(i+1,i+(len(d_b))), axis = 0)
        
    #Determines if a star and the next one are within a certain max distance
    #of each other (Designed to locate binary systems)
    elif abs(t['RAdeg'][i+1] - t['RAdeg'][i]) < bin_dist and abs(t['DEdeg'][i+1] - t['DEdeg'][i]) < bin_dist:
        d_b = np.append(d_b,(t[i:i+2]), axis = 0)
        l = len(d_b)
        
        #Averages the RV
        new_t['RV'][i:i+2] = np.mean(d_b['RV'])
        
        #Performs the RVerr calc, as described on Slack
        w_err = 1/np.sqrt(np.sum(1/(d_b['RVsig'])**2))
        
        #Updates both binary stars with new RVs, but same coordinates
        new_t['RVsig'][i:i+2] = np.sqrt(w_err**2+(np.diff(d_b['RVsig'])/4)**2)
        
    i += l

#Writes the modified table to either the same, or new file
new_t.write('filename.csv', format='ascii')