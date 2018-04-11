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

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    if len(sys.argv) == 1:
        print(" ----- Not enough arguments --------------------------\n"
              "Must provide relative path to chronostar main directory")
        raise UserWarning

    chr_path = sys.argv[1]
    sys.path.insert(0, chr_path)
    import chronostar.tfgroupfitter as tfgf
    #import chronostar.tfexpectmax as tfem
    import chronostar.hexplotter as hp

    logging.info("Imports complete")

    precs = ['perf', 'half', 'gaia', 'double']
    for prec in precs:
        try:
            os.chdir(prec)
            logging.info("In directory".format(prec))

            if not os.path.isfile("hexplot0.pdf"): 
                covs = np.load("covs.npy").item()
                means = np.load("means.npy").item()
                star_pars = tfgf.read_stars("tb_data.pkl")
                ngroups = covs.values()[0].shape[0]
                hp.plot_hexplot(star_pars, means, covs, iter_count=0, prec=prec,
                                save_dir=prec+'-')
                # tfem.plot_all(star_pars, means, covs, ngroups, iter_count)
                logging.info("finished plot for {}".format(prec))
            os.chdir("..")

        except OSError:
            logging.info("No {} directory".format(prec))

