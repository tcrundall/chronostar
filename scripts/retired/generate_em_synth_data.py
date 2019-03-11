#! /usr/bin/env python
from __future__ import print_function, division

import chronostar.synthdata

try:
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    can_plot = True
except ImportError:
    can_plot = False

from distutils.dir_util import mkpath
from distutils.errors import DistutilsFileError
import logging
import numpy as np
import sys
sys.path.insert(0, '..')
import chronostar.synthdata as syn
import chronostar.traceorbit as torb
import chronostar.retired2.converter as cv
import chronostar.fitplotter as fp
import chronostar.retired2.datatool as dt

def plotInitState(origins, star_pars):
    for dim1, dim2 in ('xy', 'xu', 'yv', 'zw', 'uv'):
        plt.clf()
        true_memb = dt.getZfromOrigins(origins, star_pars)
        fp.plotPaneWithHists(dim1,dim2,groups=origins,
                             weights=[origin.nstars for origin in origins],
                             star_pars=star_pars,
                             group_now=True,
                             membership=true_memb,
                             true_memb=true_memb)
        plt.savefig(rdir + 'pre_plot_{}{}.pdf'.format(dim1, dim2))

# extra_pars = np.array([
#     #  dX,  dV, age, nstars
#     [ 10.,  5.,  3.,   50.],
#     [  7.,  3.,  7.,   30.],
#     [ 10.,  2., 10.,   40.],
#     [ 20.,  5., 13.,   80.],
# ])
extra_pars = np.array([
    #  dX,  dV, age, nstars
    [ 10.,  1.,  12., 50.],
    [ 10.,  0.7, 25., 40],
])

ngroups = extra_pars.shape[0]
# offsets = np.zeros((ngroups, 6))
# four assoc notes:
# AB and CD overlapping in ZW
# AC overlapping in YV
# All overlapping in XY
# AB and AD overlapping in XU
# CD, AD, AB overlapping in UV
# offsets = np.array([
#     #   X,  Y,  Z,  U,  V, W
#     [-10.,  0.,  0.,-10.,  0.,-10.], # yellow / blue
#     [ 10.,  0.,-10.,  0., 10., -5.], # blue   / orange
#     [  0.,-40., 80.,  5., -5., 10.], # brown  / green
#     [ 50.,  5.,  0.,  5.,-10., 10.], # orange / red
# ])

offsets = np.array([
    #   X,  Y,  Z,  U,  V, W
    [50.,  0.,  0., 0.,  0., 0.],
    [ 0.,  0.,  0., 0.,  0., 0.],
])

try:
    run_name = sys.argv[1]
except:
    run_name = 'dummy'

try:
    rdir = "/data/mash/tcrun/em_fit/{}/".format(run_name)
    path_msg = "Storing data on mash data server"
    mkpath(rdir)
except (IOError, DistutilsFileError):
    path_msg = ("I'm guessing you're not Tim Crundall..."
                "or not on an RSAA server")
    rdir = "../results/em_fit/{}/".format(run_name)
    mkpath(rdir)

logging.basicConfig(
    level=logging.INFO,
    filename=rdir + 'data_generation.log',
    filemode='a'
)

logging.info(path_msg)
print("Should be working in the directory:\n{}".format(rdir))

# Setting up standard filenames
# Partial data generation saved in results/em_fit/[run_name]/synth_data/
sd_dir = rdir + 'synth_data/'
mkpath(sd_dir)
xyzuvw_perf_file     = sd_dir + 'perf_xyzuvw.npy'
groups_savefile      = sd_dir + 'origins.npy'
xyzuvw_init_savefile = sd_dir + 'xyzuvw_init.npy'
astro_savefile       = sd_dir + 'astro_table.txt'
bg_savefile = sd_dir + 'bg_density.npy'

# Final XYZUVW data file stored in chronostar/data/ to replicate
# treatment of real data
xyzuvw_conv_savefile = '../data/{}_xyzuvw.fits'.format(run_name)

# Calculate the initial parameters for each component that correspond
# to the current day mean of mean_now
logging.info("---------- Generating synthetic data...")
ERROR = 1.0
BG_DENS = 1.0e-7    # background density for synth bg stars
np.save(bg_savefile, BG_DENS)

logging.info("  with error fraction {}".format(ERROR))
logging.info("  and background density {}".format(BG_DENS))
# Set a current-day location around which synth stars will end up
mean_now = np.array([50., -100., -0., -10., -20., -5.])

logging.info("Mean (now):\n{}".format(mean_now))
logging.info("Extra pars:\n{}".format(extra_pars))
logging.info("Offsets:\n{}".format(offsets))

try:
    #all_xyzuvw_now_perf = np.load(xyzuvw_perf_file)
    np.load(xyzuvw_perf_file)
    #origins = dt.loadGroups(groups_savefile)
    dt.loadGroups(groups_savefile)
    #star_pars = dt.loadXYZUVW(xyzuvw_conv_savefile)
    dt.loadXYZUVW(xyzuvw_conv_savefile)
    logging.info("Synth data exists! .....")
    print("Synth data exists")
    raise UserWarning
except IOError:
    all_xyzuvw_init = np.zeros((0,6))
    all_xyzuvw_now_perf = np.zeros((0,6))
    origins = []
    for i in range(ngroups):
        logging.info(" generating from group {}".format(i))
        # MANUALLY SEPARATE CURRENT DAY DISTROS IN DIMENSION X
        mean_now_w_offset = mean_now.copy()
        # mean_now_w_offset[0] += i * 50
        mean_now_w_offset += offsets[i]
    
        mean_then = torb.traceOrbitXYZUVW(mean_now_w_offset, -extra_pars[i,-2],
                                          single_age=True)
        group_pars = np.hstack((mean_then, extra_pars[i]))
        xyzuvw_init, origin = syn.synthesiseXYZUVW(group_pars, form='sphere',
                                                   return_group=True,
                                                   internal=False)
        origins.append(origin)
        all_xyzuvw_init = np.vstack((all_xyzuvw_init, xyzuvw_init))
        xyzuvw_now_perf = torb.traceManyOrbitXYZUVW(xyzuvw_init,
                                                    times=origin.age,
                                                    single_age=True)
        all_xyzuvw_now_perf = np.vstack((all_xyzuvw_now_perf, xyzuvw_now_perf))

    # insert 'background stars' with density `BG_DENS` [pc km/s]^-3
    ubound = np.max(all_xyzuvw_now_perf, axis=0)
    lbound = np.min(all_xyzuvw_now_perf, axis=0)
    margin = 0.5 * (ubound - lbound)
    ubound += margin
    lbound -= margin
    nbg_stars = int(BG_DENS * np.prod(ubound - lbound))

    # centre bg stars on mean of assoc stars
    # centre = np.mean(all_xyzuvw_now_perf, axis=0)
    # centre = 0.5 * (ubound + lbound)
    # spread = ubound - lbound
    bg_stars_xyzuvw_perf =\
        np.random.uniform(lbound,ubound,size=(nbg_stars, 6))
    logging.info("Using background density of {}".format(BG_DENS))
    logging.info("Generated {} background stars".format(nbg_stars))
    logging.info("Spread from {}".format(lbound))
    logging.info("         to {}".format(ubound))

    all_xyzuvw_now_perf = np.vstack((all_xyzuvw_now_perf, bg_stars_xyzuvw_perf))

    np.save(groups_savefile, origins)
    np.save(xyzuvw_perf_file, all_xyzuvw_now_perf)
    astro_table = chronostar.synthdata.measureXYZUVW(all_xyzuvw_now_perf, error_frac=ERROR,
                                                     savefile=astro_savefile)
    star_pars = cv.convertMeasurementsToCartesian(
        astro_table, savefile=xyzuvw_conv_savefile,
    )
    logging.info("Synthesis complete")

    # make sure stars are initialised as expected
    if can_plot:
        plotInitState(origins, star_pars)


