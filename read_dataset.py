from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import Table
import numpy as np

#Import GAIA/HIPPARCOS data
data = Table.read(GAIA/HIPPARCOS)
#Import RAVE, GALAH and CRVAD2 DATASETS
rave = Table.read(RAVE) #RAVE DATA
galah = Table.read(GALAH) #GALAH DATA
crvad2 = Table.read('crvad2.dat', readme='crvad2.ReadMe', format='ascii.cds')

#Limit of accuracy in kpc
limit = 0.005 #5pc

rv_data = data['RV']
e_data = data['e_RV']

def crossmatch(RA,DE,Dist):
    """
    Matches a set of given coordinates with that of the main dataset. Returns
    an index for the matching catalog, as well as the distance between the 
    given coordinate and the matched coordinate
    
    PARAMETERS
    --------------
    RA,DE, Dist - Array of Right Ascensions, Declinations and Distances respectively
    """
    cat1_ra = data['RAdec']
    cat1_dec = data['DEdec']
    cat1_dis = 1.0/data['Dist']
    cat2_ra = RA
    cat2_dec = DE
    cat2_dis = Dist
    cat1 = SkyCoord(cat1_ra*u.degree,cat1_dec*u.degree,cat1_dis*u.kpc,frame='icrs')
    cat2 = SkyCoord(cat2_ra*u.degree,cat2_dec*u.degree,cat2_dis*u.kpc,frame='icrs')
    index,dist2d,dist3d = cat1.match_to_catalog_sky(cat2)
    return index, dist3d

#Calls the crossmatch function on the three datasets
index_rve,dist_rve = crossmatch(rave['RAdec'],rave['DEdec'])
index_ga,dist_ga = crossmatch(galah['RAdec'],galah['DEdec'])
index_cr,dist_cr = crossmatch((crvad2['RAhour']*15.0),crvad2['DEdec'])

#Goes through the data and replaces RV if a suitable match in one of the other
#catalogs is found
for i in range(len(data)):
    if dist_ga[i] < limit:
        rv_data[i] = galah['RV'][index_ga[i]]
        e_data[i] = galah['e_RV'][index_ga[i]]
    elif dist_rve[i] < limit:
        rv_data[i] = rave['HRV'][index_rve[i]]
        e_data[i] = rave['e_HRV'][index_rve[i]]
    elif dist_cr[i] < limit:
        rv_data[i] = crvad2['RV'][index_cr[i]]
        e_data[i] = crvad2['e_RV'][index_cr[i]]
            
#save data tables        
