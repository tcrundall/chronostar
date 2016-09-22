# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 00:39:03 2016

@author: jhansen
"""

from astropy.table import Table
import numpy as np
import math

#Read in table
t = Table.read('gdoc_Torres.csv', format = 'ascii')
#Sort table according to RA and DE
t.sort(['RAdeg', 'DEdeg'])

#Choose the maximum difference in angular position for binary stars (deg)
bin_dist = 0.01
#Choose the maximum difference in angular position to identify duplicates eg arcsecond (deg)
arcs = 0.0001

i=0
rmv = []

#Remove Duplicates
while i < (len(t)-1):
    l = 1
  
    #Determines if a star and the next in the table are within an arcsec of 
    #each other. That is, they are identical.
    if (math.fabs(t['RAdeg'][i+1] - t['RAdeg'][i]) < arcs and math.fabs(t['DEdeg'][i+1] - t['DEdeg'][i]) < arcs) or t['Name1'][i+1] == t['Name1'][i]:
        duplicates = t[i:i+2]
        n= i+1
        #Searches for other identical stars
        while math.fabs(t['RAdeg'][n+1] - t['RAdeg'][n]) < arcs and math.fabs(t['DEdeg'][n+1] - t['DEdeg'][n]) < arcs:
            duplicates.add_row(t[n+1])
            n += 1
        l = len(duplicates)
        
        #Averages the coordinates and finds the weighted average RV
        t['RAdeg'][i] = np.mean(duplicates['RAdeg'])
        t['DEdeg'][i] = np.mean(duplicates['DEdeg'])
        sum_w = np.sum(1/(duplicates['RV error'])**2)
        t['RV'][i] = np.sum(duplicates['RV']*(1/(duplicates['RV error'])**2))/sum_w
        t['RV error'][i] = 1/np.sqrt(sum_w)
        rmv = np.append(rmv, range(i+1,i+(len(duplicates))))
                
    i += l

t.remove_rows(rmv)

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
        
    i += l

#Writes the modified table to either the same, or new file
t.write('gdoc_mod.csv', format='ascii')
