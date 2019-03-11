from __future__ import print_function, division
"""
Takes a catalogue of compiled radial velocities from literature
along with Gaia DR2 astrometry (and RVs where available),
with spectral types and approximate masses (Kraus & Hillenbrand 2007)
and takes the weighted average of binaries to get centre of mass

Assumes that the Gaia astrometry is in one contiguous block
"""

import csv
import numpy as np
import pdb
import shutil
from tempfile import NamedTemporaryFile


def isFloat(string):
    try:
        myfloat = float(string)
        return (not np.isnan(myfloat) and not np.isinf(myfloat))
    except ValueError:
        return False

def condit_weighted_average(v1, v2, w1, w2):
    """
    Takes the weighted average of two arrays, but can handle missing elements

    Parameters
    ----------
    v1 : [n] array
        kinematic data of star 1
    v2 : [n] array
        kinematic data of star 2
    w1 : float
        mass of star 1
    w2 : float
        mass of star 2

    Returns
    -------
    v : [n] array
        kinematic data of centre of mass
    """
    v1 = np.array(v1).copy()
    v2 = np.array(v2).copy()
    missing1 = np.where((v1 == '') | (v1 == 'nan'))
    missing2 = np.where((v2 == '') | (v2 == 'nan'))
    both_pres = np.where((v1 != '') & (v1 != 'nan') &
                         (v2 != 'nan') & (v2 != ''))

    res = np.zeros(v1.shape)

    res[missing1] = v2[missing1].astype(np.float)
    res[missing2] = v1[missing2].astype(np.float)

    res[both_pres] = (w1**0.5 * v1.astype(np.float)
                    + w2**0.5 * v2.astype(np.float))\
                    / (w1**0.5 + w2**0.5)

    return res


data_file = "../data/bpmg_cand_w_gaia_dr2_astrometry.csv"
tempfile = NamedTemporaryFile(delete=False)
res_file = "../data/bpmg_cand_w_gaia_dr2_astrometry_comb_binars.csv"

wide_binary_name = "wide_binary"
main_rv_name = "RV"
main_erv_name = "RV error"
approx_mass_name = "approx_mass"
name_name = "Name1"
gaia_ra_name = "ra" # this column is start of the gaia contiguous data block
gaia_end_name = "pmra_pmdec_corr"

with open(data_file, 'rw') as cf, tempfile:
    rd = csv.reader(cf)
    wt = csv.writer(tempfile)

    header = rd.next()
    print(header)

    main_rv_ix = header.index(main_rv_name)
    main_erv_ix = header.index(main_erv_name)
    wide_binary_ix = header.index(wide_binary_name)
    approx_mass_ix = header.index(approx_mass_name)
    name_ix = header.index(name_name)
    gaia_astr_start = header.index(gaia_ra_name)
    gaia_astr_end = header.index(gaia_end_name) + 1

    wt.writerow(header)

    while True:
        # read row if possible
        try:
            row = rd.next()
        except:
            break

        # ensure gaia astrometry and main radial vel is present
        if row[wide_binary_ix] == 'FALSE':
            if row[gaia_astr_start] and row[main_erv_ix]:
                print("just copying...")
                wt.writerow(row)

        # take weighted average
        elif row[wide_binary_ix] == 'TRUE':
            pair_row = rd.next()
            # only bother if data is present
            if (isFloat(row[gaia_astr_start])
                    and isFloat(pair_row[gaia_astr_start])
                    and isFloat(row[main_erv_ix])
                    and isFloat(pair_row[main_erv_ix])):

                # handle case where a mass is missing
                try:
                    mass1 = float(row[approx_mass_ix])
                    mass2 = float(pair_row[approx_mass_ix])
                except ValueError:
                    mass1 = mass2 = 1.
                    print("Warning: star {} or its pair has no mass".\
                          format(row[name_ix]))
                new_row = np.zeros(len(row))

                # average all gaia astrometry
                new_row[gaia_astr_start:gaia_astr_end] =\
                    condit_weighted_average(
                        row[gaia_astr_start:gaia_astr_end],
                        pair_row[gaia_astr_start:gaia_astr_end],
                        mass1, mass2)

                # average main rv and error
                new_row[main_rv_ix:main_rv_ix+2] = \
                    condit_weighted_average(
                        row[main_rv_ix:main_rv_ix+2],
                        pair_row[main_rv_ix:main_rv_ix+2],
                        mass1, mass2)

                new_row_str = new_row.astype(np.str)

                # replace '0.0' with the empty string
                new_row_str[np.where(new_row_str == '0.0')] = ''
                new_row_str[name_ix] = row[name_ix] + " and " + pair_row[name_ix]
                print(new_row_str)
                wt.writerow(new_row_str)

        # intentionally break if no flag present (TODO: maybe change this)
        else:
            print("ERROR, no wide binary flag")
            pdb.set_trace()

shutil.move(tempfile.name, res_file)
