from __future__ import division, print_function

import logging
import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits

import error_ellipse as ee
import synthesiser as syn
import analyser as al
import traceorbit as torb
import transform as tf


COLORS = ['xkcd:orange', 'xkcd:cyan',
          'xkcd:sun yellow', 'xkcd:shit', 'xkcd:bright pink']*12
#COLORS = ['xkcd:cyan'] * 60
color_codes = ['0xF97306','0x00FFFF', '0xFFDF22',   '0x7F5F00','0xFE01B1',    '0xBC13FE',]
color_names = ['orange',  'cyan',     'sun yellow', 'shit',    'bright pink', 'neon purple',]

HATCHES = ['|', '/',  '+', '.', '*'] * 10


def plot_age_hist(ages, ax, init_conditions=None):
    """Plot a histogram of the ages

    A histogram marginalises over all the parameters yielding a
    bayesian statistical description of the best fitting age

    Parameters
    -----------
    chain : [nsteps, nwalkers, npars] array
        the chain of samples
    ax : pyplot axes object
        the axes on which to plot
    init_conditions : [15] array {None}
        group parameters that initialised the data - external encoding
    """
    logging.info("In plot_age_hist")
    ngroups = ages.shape[0]
    max_age = np.max(ages)
    bins = np.linspace(0,max_age,100)

    for i in range(ngroups):
        ax.hist(ages[i], bins=bins, label="Group {}".format(i),
                color=COLORS[i], histtype='step')
    ax.set_xlabel("Ages [Myr]")
    ax.set_ylabel("Number of samples")
    ax.legend()

#    if init_conditions is not None:
#        init_age = init_conditions[13]
#        ax.axvline(
#            init_age, ax.get_ylim()[0], ax.get_ylim()[1], color='r', ls='--'
#        )

def plot_fit(star_pars, means, covs, ngroups, iter_count, ax, dim1=0, dim2=1):
    try:
        means['origin_then']
        origins_inc = True
    except KeyError:
        origins_inc = False

    dim_label = 'XYZUVW'
    units = ['pc']*3 + ['km/s']*3
    xyzuvw = star_pars['xyzuvw'][:, 0]
    xyzuvw_cov = star_pars['xyzuvw_cov'][:, 0]
    ax.plot(xyzuvw[:, dim1], xyzuvw[:, dim2], 'b.')
    for mn, cov in zip(xyzuvw, xyzuvw_cov):
        ee.plot_cov_ellipse(cov[np.ix_([dim1,dim2],[dim1,dim2])],
                            mn[np.ix_([dim1,dim2])],
                            ax=ax, color='b', alpha=0.1)
    for i in range(ngroups):
        if origins_inc:
            ee.plot_cov_ellipse(
                covs['origin_then'][i][np.ix_([dim1,dim2],[dim1,dim2])],
                means['origin_then'][i][np.ix_([dim1,dim2])],
                ax=ax, color="xkcd:neon purple", alpha=0.3, ls='--', #hatch='|',
            )
        ee.plot_cov_ellipse(
            covs['fitted_then'][i][np.ix_([dim1,dim2],[dim1,dim2])],
            means['fitted_then'][i][np.ix_([dim1,dim2])],
            ax=ax, color=COLORS[i], alpha=0.3, ls='-.', #hatch='/',
        )
        ee.plot_cov_ellipse(
            covs['fitted_now'][i][np.ix_([dim1,dim2],[dim1,dim2])],
            means['fitted_now'][i][np.ix_([dim1,dim2])],
            ax=ax, color=COLORS[i], ec=COLORS[i],
            fill=False, alpha=0.3, hatch=HATCHES[i], ls='-.',
        )
    min_means = np.min(np.array(means.values()).reshape(-1,6), axis=0)
    max_means = np.max(np.array(means.values()).reshape(-1,6), axis=0)

    xmin = min(min_means[dim1], np.min(xyzuvw[:,dim1]))
    xmax = max(max_means[dim1], np.max(xyzuvw[:,dim1]))
    ymin = min(min_means[dim2], np.min(xyzuvw[:,dim2]))
    ymax = max(max_means[dim2], np.max(xyzuvw[:,dim2]))

    buffer = 30

    if dim1 == 0 or dim1 == 3:
        ax.set_xlim(xmax+buffer, xmin-buffer)
    else:
        ax.set_xlim(xmin-buffer, xmax+buffer)
    ax.set_ylim(ymin-buffer, ymax+buffer)

    ax.set_xlabel("{} [{}]".format(dim_label[dim1], units[dim1]))
    ax.set_ylabel("{} [{}]".format(dim_label[dim2], units[dim2]))

    logging.info("Iteration {}: {}{} plot plotted".\
                 format(iter_count, dim_label[dim1], dim_label[dim2]))


def get_age_samples(ngroups):
    """
    Uses the number of groups to explore (and hopefully find) final chains

    Returns None if any of the chains aren't there.

    Returns
    -------
    age_samples : [ngroups, nsteps*nwalkers] array
    """
    logging.info("In get_age_samples")
    age_samples = []
    for group in range(ngroups):
        # extract final burnin chain through trial and error XD
        burnin_cnt = 0
        try:
            # tfgroupfitter stores in this format
            final_chain = np.load("final_chain.npy")
            age_samples.append(final_chain[:,:,-1].flatten())
        except IOError:
            return None
    return np.array(age_samples)

def plot_hexplot(star_pars, means, covs, iter_count, prec=None,
                  save_dir=''):
    """
    Generates hex plot in the provided directory

    Paramters
    ---------
    star_pars : dict
        'xyzuvw'
        'xyzuvw_cov'
        'times'
        'something else...'
    means : dict
        'fitted_now'
        'fitted_then'
        'origin_now'  - optional (currently not in use)
        'origin_then' - optional
    covs : dict
        'fitted_now'
        'fitted_then'
        'origin_now'  - optional (currently not in use)
        'origin_then' - optional
    iter_count : integer
    """
    logging.info("In plot_hexplot, iter {}".format(iter_count))
    ngroups = covs['fitted_then'].shape[0]

    # INITIALISE PLOT
    plt.clf()
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    f.set_size_inches(30, 20)
#    f.suptitle(
#        "Iteration {}, precision {}".\
#               format(iter_count, prec)
#    )

    # PLOT THE OVAL PLOTS
    plot_fit(star_pars, means, covs, ngroups, iter_count, ax1, 0, 1)
    plot_fit(star_pars, means, covs, ngroups, iter_count, ax2, 3, 4)
    plot_fit(star_pars, means, covs, ngroups, iter_count, ax4, 0, 3)
    plot_fit(star_pars, means, covs, ngroups, iter_count, ax5, 1, 4)
    plot_fit(star_pars, means, covs, ngroups, iter_count, ax6, 2, 5)

    # PLOT THE HISTOGRAMS
    age_samples = get_age_samples(ngroups)
    if age_samples is not None:
        plot_age_hist(age_samples, ax3)

    f.savefig(save_dir+"hexplot{:02}.pdf".format(iter_count),
              bbox_inches='tight', format='pdf')
    f.clear()

def dataGatherer(res_dir=''):
    """
    Provided with a results directory, tries to find all she needs, then plots

    Parameters
    ----------
    """
    covs = {}
    means = {}
    star_pars = {}

    chain = np.load(res_dir+"final_chain.npy")
    lnprob = np.load(res_dir+"final_lnprob.npy")
    origins = np.load(res_dir+"origins.npy").item()
    best_group = al.getBestSample(chain, lnprob)

    star_pars['xyzuvw'] = fits.getdata(res_dir+"xyzuvw_now.fits", 1)
    star_pars['xyzuvw_cov'] = fits.getdata(res_dir+"xyzuvw_now.fits", 2)

    means['origin_then'] = origins.mean
    means['fitted_then'] = best_group.mean
    means['fitted_now']  = torb.traceOrbitXYZUVW(best_group.mean, best_group.age)

    covs['origin_then'] = origins.generateCovMatrix()
    covs['fitted_then'] = best_group.generateCovMatrix()
    covs['fitted_now'] = tf.transform_cov(
        covs['fitted_then'], torb.traceOrbitXYZUVW, means['fitted_then'],
        args=(best_group.age,)
    )

    plot_hexplot(star_pars, means, covs, iter_count=0, save_dir=res_dir)

