from __future__ import print_function, division
"""
Takes a catalogue of compiled radial velocities from literature
along with Gaia DR2 astrometry (and RVs where available), and
uses monte carlo sampling to combine measurements
"""

import csv
import numpy as np
import pdb
import shutil
from tempfile import NamedTemporaryFile

def gaussian(x, mu, sig):
    amp = 1 / np.sqrt(2. * np.pi * np.power(sig, 2.))
    return amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def combine_ervs_inv_quad(e_rvs):
    """
    Taken from https://en.wikipedia.org/wiki/Inverse-variance_weighting
    """
    e_rvs = np.array(e_rvs)
    sig_tot = (np.sum(e_rvs**-2))**-0.5
    return sig_tot


def combine_ervs_quad(e_rvs):
    e_rvs = np.array(e_rvs)
    sig_tot = np.sum(e_rvs**2)**0.5
    return sig_tot


def combine_rvs(rvs, e_rvs):
    """
    Take the mean of rvs, weighted by inverse sigma
    Taken from https://en.wikipedia.org/wiki/Inverse-variance_weighting
    """
    rvs = np.array(rvs)
    e_rvs = np.array(e_rvs)
    return np.sum(rvs/e_rvs**2) / np.sum(1/e_rvs**2)


def normed_error(rvs, e_rvs):
    """
    Calculates a normalised error.
    If the actual spread of measurements is more than 3 sigma of
    the combined uncertainty, then something is up...
    """
    rvs = np.array(rvs)
    e_rvs = np.array(e_rvs)
    ref_std = np.std(rvs)
    sig_tot = combine_ervs_quad(e_rvs)
    err = ref_std / sig_tot
    return err


# how many points to sample from a measurement
# NGAIA_SAMPLES = 1000
# NOTHER_SAMPLES = 100
DEF_ERR_STR = '4.0'

data_file = "../data/bpmg_cand_w_gaia_dr2_astrometry.csv"
tempfile = NamedTemporaryFile(delete=False)
res_file = data_file

main_rv_name = "RV"
main_erv_name = "RV error"
rv1_name = "RV1"
erv1_name = "eRV1"
rv2_name = "RV2"
erv2_name = "eRV2"
gaia_rv_name = "radial_velocity"
gaia_erv_name = "radial_velocity_error"
src_cnt_name = "n_obs"
normed_erv_name = "normed e_RV"
incons_flag_name = "incons_RV_flag"

extra_columns_present = False
nextra_columns = 0

with open(data_file, 'rw') as cf, tempfile:
    rd = csv.reader(cf)
    wt = csv.writer(tempfile)

    header = rd.next()
    print(header)
    main_rv_ix = header.index(main_rv_name)
    main_erv_ix = header.index(main_erv_name)
    rv1_ix = header.index(rv1_name)
    erv1_ix = header.index(erv1_name)
    rv2_ix = header.index(rv2_name)
    erv2_ix = header.index(erv2_name)
    gaia_rv_ix = header.index(gaia_rv_name)
    gaia_erv_ix = header.index(gaia_erv_name)
    src_cnt_ix = header.index(src_cnt_name)

    try:
        normed_erv_ix = header.index(normed_erv_name)
    except ValueError:
        header += [normed_erv_name]
        normed_erv_ix = header.index(normed_erv_name)
        nextra_columns += 1
    try:
        incons_flag_ix = header.index(incons_flag_name)
    except ValueError:
        header += [incons_flag_name]
        incons_flag_ix = header.index(incons_flag_name)
        nextra_columns += 1
    wt.writerow(header)

    for row in rd:
        row += nextra_columns*['']
        try:
            # handle the case with only one lit rv and no gaia
            incons_flag = False
            normed_erv = 0.
            if row[src_cnt_ix] == '1' and row[gaia_rv_ix] == '':
                print("Just copying...")
                row[main_rv_ix] = row[rv1_ix]
                if row[erv1_ix] == '':
                    row[main_erv_ix] = DEF_ERR_STR
                else:
                    row[main_erv_ix] = row[erv1_ix]
                incons_flag = False
                normed_erv = 1.

            # handle all other cases with monte carlo sampling, will append samples
            # to a single array
            else:
                print("Combining")
                samples = np.array([])

                rvs = []
                e_rvs = []
                # assumes errors immediately follow measurement
                for src_ix in [gaia_rv_ix, rv1_ix, rv2_ix]:
                    # handle empty string values
                    try:
                        rvs.append(float(row[src_ix]))
                        e_rvs.append(float(row[src_ix+1]))
                    except ValueError:
                        pass

                comb_rv = combine_rvs(rvs, e_rvs)
                comb_erv = combine_ervs_inv_quad(e_rvs)
                normed_erv = normed_error(rvs, e_rvs)

                incons_flag = normed_erv > 3.

                row[main_rv_ix] = comb_rv
                if incons_flag:
                    row[main_erv_ix] = np.std(rvs)
                else:
                    row[main_erv_ix] = comb_erv

            row[normed_erv_ix] = str(normed_erv)
            row[incons_flag_ix] = str(incons_flag)
            wt.writerow(row)
        except ValueError:
            pdb.set_trace()

shutil.move(tempfile.name, res_file)
