"""
Some of Tim's stars are missing in my dataset. Add them, and also add missing columns.
"""

import numpy as np
from astropy.table import Table, join, vstack, unique
import sys
sys.path.insert(0, '..')
from chronostar import tabletool

def what_Tims_stars_are_missing_in_my_dataset():
    usco = Table.read('usco_res/usco_run_subset.fit')
    ucl = Table.read('ucl_res/ucl_run_subset.fit')
    lcc = Table.read('lcc_res/lcc_run_subset.fit')

    tim = vstack([usco, ucl, lcc])
    tim = unique(tim, keys='source_id')

    #tim = join(usco, ucl, keys='source_id')
    #tim = join(tim, lcc, keys='source_id')

    print len(tim), len(usco), len(ucl), len(lcc)

    d=Table.read('data_table_cartesian_with_bg_ols.fits')
    print len(d)


    # source_id
    sd=set(d['source_id'])
    st=set(tim['source_id'])

    print 'Stars in Tims set and not in mine', len(st.difference(sd))
    for x in st.difference(sd):
        print x

    ct = set(tim.colnames)
    cd = set(d.colnames)
    print 'Colnames in mine and not in Tims set', cd.difference(ct)
    print 'Colnames in Tims and not in my set', ct.difference(cd)

def add_missing_tims_stars_to_my_set():
    # My table
    d = Table.read('data_table_cartesian_with_bg_ols.fits')

    # Existing Tim's table
    usco = Table.read('usco_res/usco_run_subset.fit')
    ucl = Table.read('ucl_res/ucl_run_subset.fit')
    lcc = Table.read('lcc_res/lcc_run_subset.fit')
    tim = vstack([usco, ucl, lcc])
    tim_existing = unique(tim, keys='source_id')

    tim_missing = Table.read('missing_columns_for_tims_stars.fits')

    tim = join(tim_existing, tim_missing, keys='source_id')
    print tim.colnames

    tim.remove_columns(['c_VW', 'astrometric_primary_flag', 'c_XZ', 'c_XY', 'c_ZU', 'c_ZV', 'c_ZW', 'c_XV', 'c_XW', 'c_XU', 'c_UW', 'c_UV',
     'c_YU', 'c_YW', 'c_YV', 'c_YZ', 'dZ', 'dX', 'dY', 'dV', 'dW', 'dU'])

    both = vstack([d, tim])
    print len(both)
    both = unique(both, keys='source_id')
    print len(both)

    # This table is masked. Unmask:
    both = both.filled()

    tabletool.convert_table_astro2cart(table=both, return_table=True)

    # WRITE
    both.write('data_table_cartesian_including_tims_stars_with_bg_ols.fits', format='fits', overwrite=True)

    ct = set(tim.colnames)
    cd = set(d.colnames)
    print
    print 'Colnames in mine and not in Tims set', cd.difference(ct)
    print
    print 'Colnames in Tims and not in my set', ct.difference(cd)


if __name__ == '__main__':
    #what_Tims_stars_are_missing_in_my_dataset()
    add_missing_tims_stars_to_my_set()