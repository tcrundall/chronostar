"""
Little plotting script. Designed to be run from the main
directory of the EM results
"""
# coding: utf-8
import os
import logging
import numpy as np
import sys
try:
    import matplotlib as mpl
    # prevents displaying plots from generation from tasks in background
    mpl.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not imported")
    pass


def temp():

    ngroups = 4
    iter_count = 0
    tfem.plot_all(star_pars, means, covs, ngroups, iter_count)
    covs
    covs['origin_then']
    covs_dict = {
        'origin_then':covs[0],
        'origin_now': covs[1],
        'fitted_then':covs[2],
        'fitted_now' :covs[3],
    }
    covs.shape
    covs = np.load("prev_covs.npy")
    covs_dict = {
        'origin_then':covs[0],
        'origin_now': covs[1],
        'fitted_then':covs[2],
        'fitted_now' :covs[3],
    }
    covs.shape
    covs = np.load("prev_covs.npy")
    covs.shape
    get_ipython().magic(u'pwd ')
    covs = np.load("prev_covs.npy")
    covs.shape
    covs
    covs.keys()
    covs[0].keys()
    covs.item().keys()
    tfem.plot_all(star_pars, means.item(), covs.item(), ngroups, iter_count)
    plt.savefig("myXY.pdf", format='pdf')
    import matplotlib.pyplot as plt
    plt.savefig("myXY.pdf", format='pdf')
    get_ipython().magic(u'ls ')
    get_ipython().magic(u'pwd ')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("Not enough arguments")
        raise UserWarning

    chr_path = sys.argv[1]
    sys.path.insert(0, chr_path)
    import chronostar.tfgroupfitter as tfgf
    import chronostar.tfexpectmax as tfem
    import chronostar.hexplotter as hp

    iter_count = 0
    while True:
        try:
            os.chdir("iter{}".format(iter_count))
            logging.info("In iter{} directory".format(iter_count))

            covs = np.load("covs.npy").item()
            means = np.load("means.npy").item()
            origins = np.load("../origins.npy")
            star_pars = tfgf.read_stars("../perf_tb_file.pkl")

            ngroups = covs.values()[0].shape[0]

            hp.plot_hexplot(star_pars, means, covs, iter_count)
            # tfem.plot_all(star_pars, means, covs, ngroups, iter_count)

            os.chdir("..")

        except OSError:
            logging.info("No iter{} directory".format(iter_count))
            break
        iter_count += 1
        # break

