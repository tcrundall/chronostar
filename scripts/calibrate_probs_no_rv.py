"""
Primary Chronostar script.

Perform a kinematic fit to data as described in Crundall et al. (2019).

Run this script with either simple argument
line inputs (call with --help to see options) or with a single input
config file. Unzip contents of config_examples.zip to see some examples.
"""

from __future__ import print_function, division, unicode_literals

# prevent plots trying to display (and breaking runs on servers)
try:
    import matplotlib as mpl
    mpl.use('Agg')
except ImportError:
    pass

import matplotlib.pyplot as plt

import argparse
import numpy as np
import os
import sys
import logging
import imp      # TODO: address deprecation of imp
from distutils.dir_util import mkpath
import random

from get_association_region import get_region
sys.path.insert(0, os.path.abspath('..'))
from chronostar.synthdata import SynthData
from chronostar import tabletool
from chronostar import compfitter
from chronostar import expectmax
from chronostar.component import SphereComponent


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

log_message('Setting up', symbol='.', surround=True)

assert os.access(rdir, os.W_OK)

# ------------------------------------------------------------
# -----  SETTING UP ALL DATA PREP  ---------------------------
# ------------------------------------------------------------

# Set up some filename constants (INPUT files) FOR COMPARISON and expectmax
final_comps_file_with_rv = '../data/no_rv_paper/beta_pic_sphere_component.npy'
final_memb_probs_file_with_rv = '../data/no_rv_paper/beta_memb_probs.npy'

bp_comp_with_rv = SphereComponent.load_components(final_comps_file_with_rv)
bp_probs_with_rv = np.load(final_memb_probs_file_with_rv)
print(bp_probs_with_rv)

# Set up some filename constants (OUTPUT files)
final_comps_file = 'final_comps.npy'
final_med_and_spans_file = 'final_med_and_spans.npy'
final_memb_probs_file = 'final_membership.npy'

# TODO: data_loadfile: mask out rows with nans

# First see if a data savefile path has been provided, and if
# so, then just assume this script has already been performed
# and the data prep has already been done
if (config.config['data_savefile'] != '' and
        os.path.isfile(config.config['data_savefile'])):
    log_message('Loading pre-prepared data (with background)')
    data_table = tabletool.load(config.config['data_savefile'])
    historical = 'c_XU' in data_table.colnames

# Otherwise, perform entire process
else:
    # Construct synthetic data if required
    datafile = config.config['data_loadfile']
    assert os.path.exists(datafile)

    # Read in data as table
    log_message('Read data into table')
    data_table = tabletool.read(datafile)

    # historical = 'c_XU' in data_table.colnames # column names...
    historical = False


    log_message('Data table has {} rows'.format(len(data_table)))

    # data_table['radial_velocity'] = data_table['radial_velocity_best']
    # data_table['radial_velocity_error'] = data_table['radial_velocity_error_best']
    #
    # By the end of this, data will be a astropy table
    # with cartesian data written in
    # columns in default way.
    if config.config['convert_to_cartesian']:
        print('Converting to cartesian')
        # Performs conversion in place (in memory) on `data_table`
        tabletool.convert_table_astro2cart(
                table=data_table,
                main_colnames=config.astro_colnames.get('main_colnames', None),
                error_colnames=config.astro_colnames.get('error_colnames', None),
                corr_colnames=config.astro_colnames.get('corr_colnames', None),
                return_table=True)


    # Calculate background overlaps, storing in data
    bg_lnol_colname = 'background_log_overlap'
    if config.config['include_background_distribution']:
        print("Calculating background overlaps")
        # Only calculate if missing
        if bg_lnol_colname not in data_table.colnames:
            log_message('Calculating background densities')
            # background_means = tabletool.build_data_dict_from_table(
            #         config.config['kernel_density_input_datafile'],
            #         only_means=True,
            # )
            # star_means = tabletool.build_data_dict_from_table(
            #         data_table, only_means=True,
            # )
            typical_overlap = -17.3
            ln_bg_ols = np.array(len(data_table)*[typical_overlap])

                # expectmax.get_kernel_densities(background_means,
                #                                       star_means, )

            # If allowed, save to original file path
            if config.config['overwrite_datafile']:
                tabletool.insert_column(data_table, col_data=ln_bg_ols,
                                         col_name=bg_lnol_colname, filename=datafile)
            else:
                tabletool.insert_column(data_table, col_data=ln_bg_ols,
                                        col_name=bg_lnol_colname)

if config.config['overwrite_datafile']:
    data_table.write(datafile, overwrite=True)
elif config.config['data_savefile'] != '':
    data_table.write(config.config['data_savefile'], overwrite=True)

# LOAD DATA DICT FROM ORIGINAL DATA
# LOAD DATA DICT FROM MANIPULATED DATA
# THEN CALCULATE MEMB PROBS BOTH TIMES

# Convert data table into numpy arrays of mean and covariance matrices
log_message('Building data dictionary')
data_dict = tabletool.build_data_dict_from_table(
        data_table,
        get_background_overlaps=config.config['include_background_distribution'],
        historical=historical,
)


# Save data_dict of the original data
# np.save('data_dict_original.npy', data_dict)

# MEMB PROBS for data WITH RV available
data_dict_original=np.load('data_dict_original.npy')
memb_probs_with_rv = expectmax.expectation(data=data_dict_original, comps=bp_comp_with_rv)
print(memb_probs_with_rv)


# Some values are nan. Mask out all stars that have any of values in their covariance matrices
# equal to nan.
mask=[~np.any(np.isnan(x)) for x in data_dict['covs']]
for k, v in data_dict.items():
    v2=v[mask]
    data_dict[k]=v2


memb_probs_no_rv = expectmax.expectation(data=data_dict, comps=bp_comp_with_rv, old_memb_probs=bp_probs_with_rv)
print(memb_probs_no_rv)

# Mask: compare only members
members=memb_probs_with_rv[:,0]>0.5

diff=memb_probs_no_rv-memb_probs_with_rv
diff=diff[members]

fig=plt.figure()
ax=fig.add_subplot(111)
ax.hist(diff[:,0], bins=20)
plt.savefig('dummy.pdf')