"""
I've downloaded 1.5M stars from Gaia with parallax being the only limit: ALL STARS BETWEEN 80 and 200 pc, and parallax_error limit
SELECT ... WHERE (gaiadr2.gaia_source.parallax>=5 AND gaiadr2.gaia_source.parallax<=12.5 AND gaiadr2.gaia_source.parallax_error<=0.3)

Background overlap computations are too slow to process that many stars, so narrow down the sample:
- UVW velocities

"""

import numpy as np
from astropy.table import Table
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from chronostar import tabletool

#tab = Table.read('../data/ScoCen_box_result.fits')
tab = Table.read('../data/ScoCen_box_result_with_kinematics.fits')
print('All candidates', len(tab))
print(tab)

def remove_stars_at_very_different_positions_in_the_sky(tab):
    mask = (tab['b']>-20) & (tab['b']<40) & np.logical_or(tab['l']>250, tab['l']<20)
    mask = mask & (tab['ra']>120) & (tab['ra']<280)
    mask = mask & (tab['dec']>-80) & (tab['dec']<0)

    return tab[mask]

def distance_cut(tab):
    distance = 1.0/tab['parallax']*1000.0
    mask = (distance>100.0) & (distance<160.0)
    return tab[mask]

def add_UVW_chronostar(tab):
    tabletool.convert_table_astro2cart(
        table=tab,
        main_colnames=None,
        error_colnames=None,
        corr_colnames=None,
        return_table=True)

    tab.write('ScoCen_box_result_with_kinematics.fits')

def kinematic_cut(tab):
    maskNaN = np.isnan(tab['U'])

    tab1=tab[~maskNaN]


    b=100

    #"""
    print(tab['U'])
    import matplotlib.pyplot as plt
    fig=plt.figure()
    ax=fig.add_subplot(311)
    ax.hist(tab1['U'], bins=np.linspace(-100, 100, b))

    ax=fig.add_subplot(312)
    ax.hist(tab1['V'], bins=np.linspace(-100, 100, b))

    ax=fig.add_subplot(313)
    ax.hist(tab1['W'], bins=np.linspace(-50, 50, b))

    plt.show()
    #"""

    maskU = (tab['U']>-25) & (tab['U']<10)
    maskV = (tab['V']>-20) & (tab['V']<5)
    maskW = (tab['W']>-10) & (tab['W']<10)
    mask = maskU & maskV & maskW

    # UVW within limits or no RV
    mask = np.logical_or(mask, maskNaN)

    print len(tab[maskU])
    print len(tab[maskV])
    print len(tab[maskW])

    tab=tab[mask]

    print(len(tab))

    return tab


#tab = remove_stars_at_very_different_positions_in_the_sky(tab)
#print(len(tab))

#tab = distance_cut(tab)
#print(len(tab))


#tab = add_UVW(tab)
#tab.write('../data/ScoCen_box_result_with_kinematics.fits', format='fits', overwrite=True)

#add_UVW_chronostar(tab)

tab=kinematic_cut(tab)
print(len(tab))
tab.write('scocen_100k_candidates.fits')

print('Days needed', 13.5*len(tab)/3600.0/24.0)
print('Hours needed', 13.5*len(tab)/3600.0)
#What about excluding giants?