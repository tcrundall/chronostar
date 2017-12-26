from __future__ import division, print_function

import numpy as np



def plot_quadplots(infile, bayes_fits, init_conditions=None, dir=''):
    """
    Generates many quad plots in the provided directory

    Parameters
    ----------
    infile : str
        The traceback file of a set of stars
    bayes_fits : ???
        Some means of conveying the bayesian fit to each age step
    """

    stars, times, xyzuvw, xyzuvw_cov = pickle.load(open(infile, 'r'))
    nstars = len(xyzuvw)

    naive_spreads = get_naive_spreads(xyzuvw)
    bayes_spreads = get_bayes_spreads(infile)

    return 0
