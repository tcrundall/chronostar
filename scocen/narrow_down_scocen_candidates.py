"""
I've downloaded 1.5M stars from Gaia with parallax being the only limit: ALL STARS BETWEEN 80 and 200 pc, and parallax_error limit
SELECT ... WHERE (gaiadr2.gaia_source.parallax>=5 AND gaiadr2.gaia_source.parallax<=12.5 AND gaiadr2.gaia_source.parallax_error<=0.3)

Background overlap computations are too slow to process that many stars, so narrow down the sample:
- UVW velocities

"""

import numpy as np
from astropy.table import Table

tab = Table.read('../data/ScoCen_box_result.fits')
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

def add_UVW(tab):
    '''
    Determine UVW.
    Need:
    parallax
    pmra, pmdec
    rv
    '''

    import astropy.coordinates as coord
    import astropy.units as u

    print 'Determine UVW for %d stars.' % len(tab)

    # ~ tab=tab[:10]

    result = []
    U = []
    V = []
    W = []
    i = 0
    for x in tab:
        # ~ print x
        try:
            c1 = coord.ICRS(ra=x['ra'] * u.degree, dec=x['dec'] * u.degree,
                            distance=(x['parallax'] * u.mas).to(u.pc, u.parallax()),
                            pm_ra_cosdec=x['pmra'] * u.mas / u.yr,
                            pm_dec=x['pmdec'] * u.mas / u.yr,
                            radial_velocity=x['radial_velocity'] * u.km / u.s)

            gc1 = c1.transform_to(coord.Galactocentric)
            i += 1
            if i % 100 == 0:
                print i, len(tab), gc1.v_x, gc1.v_y, gc1.v_z
            result.append([x['source_id'], gc1.v_x.value, gc1.v_y.value, gc1.v_z.value])
            U.append(gc1.v_x.value)
            V.append(gc1.v_y.value)
            W.append(gc1.v_z.value)
        except:
            U.append(np.nan)
            V.append(np.nan)
            W.append(np.nan)
    print U

    tab['U'] = U
    tab['V'] = V
    tab['W'] = W
    tab['U'].unit = 'km/s'
    tab['V'].unit = 'km/s'
    tab['W'].unit = 'km/s'

    return tab

tab = remove_stars_at_very_different_positions_in_the_sky(tab)
print(len(tab))

tab = distance_cut(tab)
print(len(tab))


tab = add_UVW(tab)
tab.write('../data/ScoCen_box_result_with_kinematics.fits', format='fits', overwrite=True)

print('Days needed', 13.5*len(tab)/3600.0/24.0)
#What about excluding giants?