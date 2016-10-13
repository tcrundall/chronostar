# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 00:39:03 2016

@author: jhansen
"""

from astropy.table import Table
import numpy as np
import numpy.core.defchararray as npstr
import math

#Read in table
t = Table.read('gdoc_better.csv', format = 'ascii.csv')
#Sort table according to RA and DE
t.sort(['RAdeg', 'DEdeg'])

arcsec = 1./3600.

#Choose the maximum difference in angular position for binary stars (deg)
bin_dist = 50*arcsec
#Choose the maximum difference in angular position to identify duplicates eg arcsecond (deg)
dup_dist = 1*arcsec

#Initialise Variables
i=0
t['n_obs'] = np.zeros(len(t)).astype('int32')
t['wide_binary'] = np.empty(len(t)).astype('S8')
rmv = []
ref = []

#Remove Duplicates
while i <= (len(t)-1):
    l = 1
  
    #Determines if a star and the next in the table are within an arcsec of 
    #each other. That is, they are identical.
    if i < (len(t)-1) and ((math.fabs(t['RAdeg'][i+1] - t['RAdeg'][i]) < dup_dist and math.fabs(t['DEdeg'][i+1] - t['DEdeg'][i]) < dup_dist) or t['Name1'][i+1] == t['Name1'][i]):
        duplicates = t[i:i+2]
        n= i+1
        #Searches for other identical stars
        while math.fabs(t['RAdeg'][n+1] - t['RAdeg'][n]) < dup_dist and math.fabs(t['DEdeg'][n+1] - t['DEdeg'][n]) < dup_dist:
            duplicates.add_row(t[n+1])
            n += 1
        l = len(duplicates)
        
        #Averages the coordinates and finds the weighted average RV
        t['RAdeg'][i] = np.mean(duplicates['RAdeg'])
        t['DEdeg'][i] = np.mean(duplicates['DEdeg'])
        sum_w = np.sum(1/(duplicates['RV error'])**2)
        t['RV'][i] = np.sum(duplicates['RV']*(1/(duplicates['RV error'])**2))/sum_w
        t['RV error'][i] = 1/np.sqrt(sum_w)

        rmv = np.append(rmv, range(i+1,i+l))
    
    #Combine References
    new_ref = np.unique([t['Reference'][j] for j in range(i,i+l)])
    joined_ref = ', '.join(new_ref)   
    ref = np.append(ref,joined_ref)   
    t['n_obs'][i] = l  
       
    i += l

t.remove_rows(rmv)
t.remove_column('Reference')
t['Reference'] = ref
i=0

#Find Binaries and average RVs
while i < (len(t)-1):
    l = 1    
    #Determines if a star and the next one are within a certain max distance
    #of each other (Designed to locate binary systems)
    if math.fabs(t['RAdeg'][i+1] - t['RAdeg'][i]) < bin_dist and math.fabs(t['DEdeg'][i+1] - t['DEdeg'][i]) < bin_dist:
        binaries = t[i:i+2]
        l = len(binaries)
        
        #Averages the RV
        t['RV'][i:i+2] = np.mean(binaries['RV'])
        
        #Performs the RVerr calc, as described on Slack
        w_err = 1/np.sqrt(np.sum(1/(binaries['RV error'])**2))
        
        #Updates both binary stars with new RVs, but same coordinates
        t['RV error'][i:i+2] = np.sqrt(w_err**2+(np.diff(binaries['RV error'])/4)**2)
        t['wide_binary'][i:i+2] = 'True'

    else:
        t['wide_binary'][i] = 'False'
        
    i += l

#Writes the modified table to either the same, or new file
t.write('gdoc_mod.csv', format='ascii')
