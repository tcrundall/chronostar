# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 01:48:18 2016

@author: jhansen
"""

#NOTE - Column names are placeholders - don't know the actual names on the table

from astropy.table import Table

#Read in table
t = Table.read('filename.fits', format='ascii')


for i in range(0,len(t)):
    
    #First attempts to use GALAH RV
    if t['GALAHRV'][i] > 0:
        t['bestRV'][i] = t['GALAHRV'][i]
        
    #Then tries RAVE        
    elif t['RAVERV'][i] > 0:
        t['bestRV'][i] = t['RAVERV'][i]
        
    #Gontcharov...
    elif t['GontcharovRV'][i] > 0:
        t['bestRV'][i] = t['GontcharovRV'][i]
        
    #RVs from the google sheet
    elif t['gSheetRV'][i] > 0:
        t['bestRV'][i] = t['gSheetRV'][i]
        
    #Then resorts to using TGAS data
    else:
        t['bestRV'][i] = t['TGASRV'][i]   

#Writes table
t.write('filename2.fits', format = 'fits')
