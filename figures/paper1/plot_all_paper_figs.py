from __future__ import print_function, division

"""
A script which gathers all plotting of all relevant figures into
one spot to facilitate quick and simple replotting as needed.
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '../..')

import chronostar.fitplotter as fp
import chronostar.datatool as dt
import chronostar.synthesiser as syn
import chronostar.traceorbit as torb


# plotting federrath stars
synth_fit = 'fed_stars'
rdir = '../../results/new_fed_stars_20/gaia/'
origins_file = rdir + 'origins.npy'
chain_file = rdir + 'final_chain.npy'
lnprob_file = rdir + 'final_lnprob.npy'
star_pars_file = rdir + 'xyzuvw_now.fits'
init_xyzuvw_file = '../../data/sink_init_xyzuvw.npy'
perf_xyzuvw_file = rdir + '../perf_xyzuvw.npy'
# star_pars_file = '../../data/fed_stars_20_xyzuvw.fits'

chain = np.load(chain_file).reshape(-1,9)
lnprobs = np.load(lnprob_file)
# best_fit_pars = np.load(chain_file)[np.unravel_index(np.argmax(lnprobs), lnprobs.shape)]
best_fit_pars = chain[np.argmax(lnprobs)]
groups = [syn.Group(best_fit_pars, internal=True, starcount=False)]
origins = dt.loadGroups(origins_file)
raw_init_xyzuvw = np.load(init_xyzuvw_file)
perf_xyzuvw = np.load(perf_xyzuvw_file)
init_xyzuvw = torb.traceManyOrbitXYZUVW(perf_xyzuvw, -origins[0].age,
                                        single_age=True)


for dim1, dim2 in ['xy', 'xu', 'yv']:
    plt.clf()
    fp.plotPane(
        dim1,
        dim2,
        groups=groups,
        star_pars={'xyzuvw':init_xyzuvw},
        group_then=True,
        savefile='{}_then_half_{}{}.pdf'.format(synth_fit, dim1, dim2)
    )



for dim1, dim2 in ['xy', 'xu', 'yv']:
    fp.plotPaneWithHists(
        dim1,
        dim2,
        groups=groups,
        star_pars=star_pars_file,
        group_now=True,
        group_then=True,
        star_orbits=True,
        group_orbit=True,
        membership=None,
        true_memb=None,
        savefile='{}_half_{}{}.pdf'.format(synth_fit, dim1, dim2)
    )

# plotting Multi-component synth fits
synth_fits = [
    'four_assocs',
    'assoc_in_field',
    'same_centroid',
]

planes = {
    'four_assocs':['xy', 'yv'],
    'assoc_in_field':['uv', 'xu'],
    'same_centroid':['xu', 'yv'],
}

if False:
    for synth_fit in synth_fits:
        rdir = '../../results/em_fit/{}_res/'.format(synth_fit)
        groups_file = rdir + 'final_best_groups.npy'
        # star_pars_file = rdir + '{}_xyzuvw.fits'.format(synth_fit)
        star_pars_file = '../../data/{}_xyzuvw.fits'.format(synth_fit)
        memb_file = rdir + 'final_membership.npy'
        origins_file = rdir + 'synth_data/origins.npy'
        true_memb = dt.getZfromOrigins(origins_file, star_pars_file)
        for dim1, dim2 in planes[synth_fit]:
            fp.plotPaneWithHists(
                dim1,
                dim2,
                groups=groups_file,
                star_pars=star_pars_file,
                group_now=True,
                # group_then=True,
                # star_orbits=True,
                # group_orbit=True,
                membership=memb_file,
                true_memb=true_memb,
                savefile='{}_{}{}.pdf'.format(synth_fit, dim1, dim2)
            )

