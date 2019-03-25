from __future__ import division, print_function
"""
Use this script to gather the results of an em fit
"""
import numpy as np
import sys
sys.path.insert(0, '..')


def calc_best_fit(flat_samples):
    """
    Given a set of aligned (converted?) samples, calculate the median and
    errors of each parameter
    """
    return np.array( map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                     zip(*np.percentile(flat_samples, [16,50,84], axis=0))))

if __name__ == '__main__':
    main_dir = '../results/'
    chain_filename = "final_chain.npy"

    try:
        run_name = sys.argv[1].strip('/')
        final_iter_cnt = int(sys.argv[2])
        group_cnt = int(sys.argv[3])
    except:
        print("USAGE: python em_results_gatherer.py [run_name] [iter_cnt]"
              " [group_cnt]")

    rdir = main_dir + run_name + '/iter{}/'.format(final_iter_cnt)

    fits_w_errs = np.zeros((group_cnt, 9, 3))
    for group_ix in range(group_cnt):
        gdir = rdir + "group{}/".format(group_ix)
        flat_chain = np.load(gdir + chain_filename).reshape(-1,9)
        conv_chain = np.copy(flat_chain)
        conv_chain[:,6:8] = np.exp(conv_chain[:,6:8])
        fits_w_errs[group_ix] = calc_best_fit(conv_chain)

