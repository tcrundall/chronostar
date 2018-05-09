from __future__ import division, print_function

import logging
import matplotlib.pyplot as plt
import numpy as np
import pdb

from astropy.io import fits

import errorellipse as ee
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
    xyzuvw = star_pars['xyzuvw']
    xyzuvw_cov = star_pars['xyzuvw_cov']
    ax.plot(xyzuvw[:, dim1], xyzuvw[:, dim2], 'b.')
    for mn, cov in zip(xyzuvw, xyzuvw_cov):
        ee.plotCovEllipse(cov[np.ix_([dim1,dim2],[dim1,dim2])],
                            mn[np.ix_([dim1,dim2])],
                            ax=ax, color='b', alpha=0.1)
    for i in range(ngroups):
        if origins_inc:
            ee.plotCovEllipse(
                covs['origin_then'][i][np.ix_([dim1,dim2],[dim1,dim2])],
                means['origin_then'][i][np.ix_([dim1,dim2])],
                with_line=True,
                ax=ax, color="xkcd:neon purple", alpha=0.3, ls='--', #hatch='|',
            )
        ee.plotCovEllipse(
            covs['fitted_then'][i][np.ix_([dim1,dim2],[dim1,dim2])],
            means['fitted_then'][i][np.ix_([dim1,dim2])],
            with_line=True,
            ax=ax, color=COLORS[i], alpha=0.3, ls='-.', #hatch='/',
        )
        # I plot a marker in the middle for scenarios where the volume
        # collapses to a point
        ax.plot(means['fitted_then'][i][dim1], means['fitted_then'][i][dim2],
                 color=COLORS[i], marker='x')
        ee.plotCovEllipse(
            covs['fitted_now'][i][np.ix_([dim1,dim2],[dim1,dim2])],
            means['fitted_now'][i][np.ix_([dim1,dim2])],
            with_line=True,
            ax=ax, color=COLORS[i], ec=COLORS[i],
            fill=False, alpha=0.3, hatch=HATCHES[i], ls='-.',
        )
#    min_means = np.min(np.array(means.values()).reshape(-1,6), axis=0)
#    max_means = np.max(np.array(means.values()).reshape(-1,6), axis=0)
#
#    xmin = min(min_means[dim1], np.min(xyzuvw[:,dim1]))
#    xmax = max(max_means[dim1], np.max(xyzuvw[:,dim1]))
#    ymin = min(min_means[dim2], np.min(xyzuvw[:,dim2]))
#    ymax = max(max_means[dim2], np.max(xyzuvw[:,dim2]))
#
#    buffer = 30
#
#    ax.set_xlim(xmin-buffer, xmax+buffer)
#    ax.set_ylim(ymin-buffer, ymax+buffer)

    ax.set_xlabel("{} [{}]".format(dim_label[dim1], units[dim1]))
    ax.set_ylabel("{} [{}]".format(dim_label[dim2], units[dim2]))

    logging.info("Iteration {}: {}{} plot plotted".\
                 format(iter_count, dim_label[dim1], dim_label[dim2]))


def get_age_samples(ngroups, final_chain):
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
            age_samples.append(final_chain[0][:,:,-1].flatten())
        except IOError:
            return None
    return np.array(age_samples)

def plot_hexplot(star_pars, means, covs, chain, iter_count, prec=None,
                  save_dir='', file_stem='hexplot', title=''):
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
    f.suptitle(title)

    # PLOT THE OVAL PLOTS
    plot_fit(star_pars, means, covs, ngroups, iter_count, ax1, 0, 1)
    plot_fit(star_pars, means, covs, ngroups, iter_count, ax2, 3, 4)
    plot_fit(star_pars, means, covs, ngroups, iter_count, ax4, 0, 3)
    plot_fit(star_pars, means, covs, ngroups, iter_count, ax5, 1, 4)
    plot_fit(star_pars, means, covs, ngroups, iter_count, ax6, 2, 5)

    # PLOT THE HISTOGRAMS
    age_samples = get_age_samples(ngroups, chain)
    if age_samples is not None:
        plot_age_hist(age_samples, ax3)

    f.savefig(save_dir+file_stem+"{:02}.pdf".format(iter_count),
              bbox_inches='tight', format='pdf')
    f.clear()

def dataGatherer(res_dir='', save_dir='', data_dir='', xyzuvw_file='',
                 title='', file_stem='hexplot'):
    """
    Provided with a results directory, tries to find all she needs, then plots

    Parameters
    ----------
    """
    covs = {}
    means = {}
    star_pars = {}

    chain_file = res_dir + "final_chain.npy"
    lnprob_file = res_dir + "final_lnprob.npy"
    origin_file = res_dir + "origins.npy"
    if not xyzuvw_file:
        logging.info("No xyzuvw filename provided. Must be a synth fit yes?")
        xyzuvw_file = res_dir + "xyzuvw_now.fits"


    chain = np.load(chain_file)
    chain = np.array([chain])
    lnprob = np.load(lnprob_file)
    best_group = al.getBestSample(chain, lnprob)

    star_pars['xyzuvw'] = fits.getdata(xyzuvw_file, 1)
    star_pars['xyzuvw_cov'] = fits.getdata(xyzuvw_file, 2)

    try:
        origins = np.load(origin_file).item()
        means['origin_then'] = np.array([origins.mean])
        covs['origin_then'] = np.array([origins.generateCovMatrix()])
    except IOError:
        logging.info("No origins file: {}".format(origin_file))

    means['fitted_then'] = np.array([best_group.mean])
    means['fitted_now']  =\
        np.array([torb.traceOrbitXYZUVW(best_group.mean, best_group.age)])

    covs['fitted_then'] = np.array([best_group.generateCovMatrix()])
    covs['fitted_now']  =\
        np.array([
            tf.transform_cov(covs['fitted_then'][0], torb.traceOrbitXYZUVW,
                             means['fitted_then'][0], args=(best_group.age,True)
                             )
        ])

    plot_hexplot(star_pars, means, covs, chain, iter_count=0, save_dir=save_dir,
                 file_stem=file_stem, title=title)
