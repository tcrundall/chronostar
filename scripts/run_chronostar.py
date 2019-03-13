"""
Primary Chronostar script.

Perform a kinematic fit to data as described in Crundall et al. (2019).

Run this script with either simple argument
line inputs (call with --help to see options) or with a single input
config file. Unzip contents of config_examples.zip to see some examples.
"""

from __future__ import print_function, division, unicode_literals

import argparse
import numpy as np
import os
import sys
import logging
import imp      # TODO: address deprecation of imp
from distutils.dir_util import mkpath
import random

sys.path.insert(0, os.path.abspath('..'))
from chronostar.synthdata import SynthData
from chronostar import tabletool
from chronostar import expectmax


def dummy_trace_orbit_func(loc, times=None):
    """
    Dummy trace orbit func to skip irrelevant computation
    A little constraint on age (since otherwise its a free floating
    parameter)
    """
    if times is not None:
        if np.all(times > 1.):
            return loc + 10.
    return loc


def log_message(msg, symbol='.', surround=False):
    """Little formatting helper"""
    res = '{}{:^40}{}'.format(5*symbol, msg, 5*symbol)
    if surround:
        res = '\n{}\n{}\n{}'.format(50*symbol, res, 50*symbol)
    logging.info(res)


# Check if single input is provided, and treat as config file
# at the moment config file needs to be in same directory...?
if len(sys.argv) == 2:
    config_name = sys.argv[1]
    config = imp.load_source(config_name.replace('.py', ''), config_name)
    # config = importlib.import_module(config_name.replace('.py', ''), config_name)

# Import suitable component class
if config.special['component'].lower() == 'sphere':
    from chronostar.component import SphereComponent as Component
elif config.special['component'].lower() == 'ellip':
    from chronostar.component import EllipComponent as Component
else:
    raise UserWarning('Unknown (or missing) component parametrisation')


# Check results directory is valid
# If path exists, make a new results_directory with a random int
if os.path.exists(config.config['results_dir']) and\
        not config.config['overwrite_prev_run']:
    rdir = '{}_{}'.format(config.config['results_dir'].rstrip('/'),
                          random.randint(0,1000))
else:
    rdir = config.config['results_dir']
rdir = rdir.rstrip('/') + '/'
mkpath(rdir)

# Now that results directory is set up, can set up log file
logging.basicConfig(filename=rdir+'log.log', level=logging.INFO)
log_message('Beginning Chronostar run',
            symbol='_', surround=True)

assert os.access(rdir, os.W_OK)

# Construct synthetic data if required
datafile = config.config['datafile']
if config.synth is not None:
    log_message('Getting synthetic data')
    if not os.path.exists(datafile) and config.config['pickup_prev_run']:
        synth_data = SynthData(pars=config.synth['pars'],
                               starcounts=config.synth['starcounts'],
                               Components=Component)
        synth_data.synthesise_everything(filename=datafile,
                                         overwrite=True)
    else:
        log_message('Synthetic data already exists')
assert os.path.exists(datafile)

# Set up some filename constants
final_comps_file = 'final_comps.npy'
final_med_and_spans_file = 'final_med_and_spans.npy'
final_memb_probs_file = 'final_membership.npy'


# By the end of this, data will be either a astropy table or
# the path to an astropy table, with cartesian data written in
# columns in default way. (Unless cartesian data was already
# provided in non default way - handle this side-case later)
if config.config['convert_to_cartesian']:
    data = tabletool.convertTableAstroToXYZUVW(
            table=datafile,
            main_colnames=config.astro_colnames.get('main_colnames', None),
            error_colnames=config.astro_colnames.get('error_colnames', None),
            corr_colnames=config.astro_colnames.get('corr_colnames', None),
            return_table=True)
    if config.config['overwrite_datafile']:
        data.write(datafile)
    elif config.config['cartesian_savefile'] != '':
        data.write(config.config['cartesian_savefile'])
else:
    data = datafile

# Set up trace_orbit_func
if config.config['dummy_trace_orbit_function']:
    trace_orbit_func = dummy_trace_orbit_func
else:
    trace_orbit_func = None

# TODO: Implmenet background overlap calculations.

STARTING_NCOMPS = 1
MAX_COMPS = 20          # Set a ceiling on how long code can run for

# Set up initial values
ncomps = STARTING_NCOMPS

# Fit the first component
log_message(msg='FITTING {} COMPONENT'.format(ncomps),
            symbol='*', surround=True)
run_dir = rdir + '{}/'.format(ncomps)

# Try and recover any results from previous run
try:
    prev_comps = Component.load_components(run_dir + 'final/'
                                           + final_comps_file)
    prev_med_and_spans = np.load(run_dir + 'final/'
                            + final_med_and_spans_file)
    prev_memb_probs = np.load(run_dir + 'final/' + final_memb_probs_file)
    # new_comps = Component.load_components(run_dir + 'final/final_comps.npy')
    # new_med_and_spans = np.load(run_dir + 'final/final_med_and_spans.npy')
    # new_memb_probs = np.load(run_dir + 'final/final_membership.npy')
    logging.info('Loaded from previous run')
except IOError:
    prev_comps, prev_med_and_spans, prev_memb_probs = \
        expectmax.fitManyGroups(data=data, ncomps=ncomps, rdir=run_dir,
                                trace_orbit_func=trace_orbit_func,
                                burnin=config.advanced['burnin_steps'],
                                sampling_steps=config.advanced['sampling_steps'],
                                )

# Calculate global score of fit for comparison with future fits with different
# component counts
prev_lnlike = expectmax.getOverallLnLikelihood(data, prev_comps,
                                       # bg_ln_ols=bg_ln_ols,
                                              )
prev_lnpost = expectmax.getOverallLnLikelihood(data, prev_comps,
                                       # bg_ln_ols=bg_ln_ols,
                                       inc_posterior=True)
prev_bic = expectmax.calcBIC(data, ncomps, prev_lnlike)

ncomps += 1

# Begin iterative loop, each time trialing the incorporation of a new component
while ncomps < MAX_COMPS:
    if ncomps >= MAX_COMPS:
        log_message(msg='REACHED MAX COMP LIMIT', symbol='+', surround=True)
        break

    log_message(msg='FITTING {} COMPONENT'.format(ncomps),
                symbol='*', surround=True)

    best_fits = []
    lnlikes = []
    lnposts = []
    bics = []
    all_med_and_spans = []
    all_memb_probs = []

    # Iteratively try subdividing each previous component
    for i, target_comp in enumerate(prev_comps):
        run_dir = rdir + '{}/{}/'.format(ncomps, chr(ord('A') + i))
        mkpath(run_dir)

        assert isinstance(target_comp, Component)
        # Decompose and replace the ith component with two new components
        # by using the 16th and 84th percentile ages from previous run
        split_comps = target_comp.splitGroup(lo_age=prev_med_and_spans[i,-1,1],
                                             hi_age=prev_med_and_spans[i,-1,2])
        init_comps = list(prev_comps)
        init_comps.pop(i)
        init_comps.insert(i, split_comps[1])
        init_comps.insert(i, split_comps[0])

        # Run em fit
        # First try and find any previous runs
        try:
            comps = Component.load_components(run_dir + 'final/'
                                              + final_comps_file)
            med_and_spans = np.load(run_dir + 'final/'
                                    + final_med_and_spans_file)
            memb_probs = np.load(run_dir + 'final/' + final_memb_probs_file)
            logging.info('Fit loaded from previous run')
        except IOError:
            comps, med_and_spans, memb_probs = \
            expectmax.fitManyGroups(data=data, ncomps=ncomps, rdir=run_dir,
                                    # bg_ln_ols=bg_ln_ols,
                                    init_comps=init_comps,
                                    trace_orbit_func=trace_orbit_func,
                                    burnin=config.advanced['burnin_steps'],
                                    sampling_steps=config.advanced['sampling_steps'],
                                    )

        best_fits.append(comps)
        all_med_and_spans.append(med_and_spans)
        all_memb_probs.append(memb_probs)
        lnlikes.append(
               expectmax.getOverallLnLikelihood(data, comps, bg_ln_ols=None,)
        )
        lnposts.append(
                expectmax.getOverallLnLikelihood(data, comps, bg_ln_ols=None,
                                                 inc_posterior=True)
        )
        bics.append(expectmax.calcBIC(data, ncomps, lnlikes[-1]))
        logging.info('Decomposiiton finished with \nBIC: {}\nlnlike: {}\n'
                     'lnpost: {}'.format(
                bics[-1], lnlikes[-1], lnposts[-1],
        ))

    # identify the best performing decomposition
    # best_split_ix = np.argmax(lnposts)
    best_split_ix = np.argmin(bics)
    new_comps, new_meds, new_z, new_lnlike, new_lnpost, new_bic = \
        zip(best_fits, all_med_and_spans, all_memb_probs,
            lnlikes, lnposts, bics)[best_split_ix]
    logging.info("Selected {} as best decomposition".format(best_split_ix))
    logging.info("Turned\n{}".format(prev_comps[best_split_ix].get_pars()))
    logging.info("into\n{}\n&\n{}".format(
            new_comps[best_split_ix].get_pars(),
            new_comps[best_split_ix + 1].get_pars(),
    ))

    # Check if the fit has improved
    if new_bic < prev_bic:
        logging.info("Extra component has improved BIC...")
        logging.info("New BIC: {} < Old BIC: {}".format(new_bic, prev_bic))
        logging.info("lnlike: {} | {}".format(new_lnlike, prev_lnlike))
        logging.info("lnpost: {} | {}".format(new_lnpost, prev_lnpost))
        prev_comps, prev_meds, prev_z, prev_lnlike, prev_lnpost, \
        prev_bic = \
            (new_comps, new_meds, new_z, new_lnlike, new_lnpost, new_bic)
        ncomps += 1
    else:
        logging.info("Extra component has worsened BIC...")
        logging.info("New BIC: {} > Old BIC: {}".format(new_bic, prev_bic))
        logging.info("lnlike: {} | {}".format(new_lnlike, prev_lnlike))
        logging.info("lnpost: {} | {}".format(new_lnpost, prev_lnpost))
        logging.info("... saving previous fit as best fit to data")
        np.save(rdir + final_comps_file, prev_comps)
        np.save(rdir + final_med_and_spans_file, prev_meds)
        np.save(rdir + final_memb_probs_file, prev_z)
        np.save(rdir + 'final_likelihood_post_and_bic',
                [prev_lnlike, prev_lnpost,
                 prev_bic])
        logging.info('Final best fits:')
        [logging.info(g.getSphericalPars()) for g in prev_comps]
        logging.info('Final age med and span:')
        [logging.info(row[-1]) for row in prev_meds]
        logging.info('Membership distribution: {}'.format(prev_z.sum(axis=0)))
        logging.info('Final membership:')
        logging.info('\n{}'.format(np.round(prev_z * 100)))
        logging.info('Final lnlikelihood: {}'.format(prev_lnlike))
        logging.info('Final lnposterior:  {}'.format(prev_lnpost))
        logging.info('Final BIC: {}'.format(prev_bic))
        break

    logging.info("Best fit:\n{}".format(
            [group.getInternalSphericalPars() for group in prev_comps]))
