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

# how many points to sample from a measurement
NGAIA_SAMPLES = 1000
NOTHER_SAMPLES = 100
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

    wt.writerow(header)

    for row in rd:
        try:
            # handle the case with only one lit rv and no gaia
            if row[src_cnt_ix] == '1' and row[gaia_rv_ix] == '':
                print("Just copying...")
                row[main_rv_ix] = row[rv1_ix]
                if row[erv1_ix] == '':
                    row[main_erv_ix] = DEF_ERR_STR
                else:
                    row[main_erv_ix] = row[erv1_ix]

            # handle all other cases with monte carlo sampling, will append samples
            # to a single array
            else:
                print("Sampling")
                samples = np.array([])

                if row[gaia_rv_ix] != '':
                    gaia_samples = np.random.randn(NGAIA_SAMPLES)\
                                    * float(row[gaia_erv_ix])\
                                    + float(row[gaia_rv_ix])
                    samples = np.append(samples, gaia_samples)

                # Now do any literature RVs, if error not included, we use
                # some large default uncertainty
                if row[rv1_ix] != '':
                    other_samples = np.random.randn(NOTHER_SAMPLES)\
                                     * float(row[erv1_ix] or DEF_ERR_STR)\
                                     + float(row[rv1_ix])
                    samples = np.append(samples, other_samples)

                if row[rv2_ix] != '':
                    other_samples = np.random.randn(NOTHER_SAMPLES) \
                                    * float(row[erv2_ix] or DEF_ERR_STR) \
                                    + float(row[rv2_ix])
                    samples = np.append(samples, other_samples)

                row[main_rv_ix] = np.mean(samples)
                row[main_erv_ix] = np.std(samples)

            wt.writerow(row)
        except ValueError:
            pdb.set_trace()

shutil.move(tempfile.name, res_file)
