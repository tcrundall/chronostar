from __future__ import print_function, division

import numpy as np
import sys

import chronostar.likelihood

sys.path.insert(0,'..')

from chronostar import expectmax as em
from chronostar.synthdata import SynthData
from chronostar.component import SphereComponent
from chronostar import tabletool
import chronostar.synthdata as syn
# import chronostar.retired2.measurer as ms
# import chronostar.retired2.converter as cv

#
# def test_calcMedAndSpan():
#     """
#     Test that the median, and +- 34th percentiles is found correctly
#     """
#     dx = 10.
#     dv = 5.
#     dummy_mean = np.array([10,10,10, 5, 5, 5,np.log(dx),np.log(dv),20])
#     dummy_std =  np.array([1.,1.,1.,1.,1.,1.,0.5,       0.5,       3.])
#     assert len(dummy_mean) == len(dummy_std)
#     npars = len(dummy_mean)
#
#     nsteps = 10000
#     nwalkers = 18
#
#     dummy_chain = np.array([np.random.randn(nsteps)*std + mean
#                             for (std, mean) in zip(dummy_std, dummy_mean)]).T
#     np.repeat(dummy_chain, 18, axis=0).reshape(nwalkers,nsteps,npars)
#
#     med_and_span = em.calcMedAndSpan(dummy_chain)
#     assert np.allclose(dummy_mean, med_and_span[:,0], atol=0.1)
#     approx_stds = 0.5*(med_and_span[:,1] - med_and_span[:,2])
#     assert np.allclose(dummy_std, approx_stds, atol=0.1)

def test_calcMembershipProbs():
    """
    Even basicer. Checks that differing overlaps are
    correctly mapped to memberships.
    """
    # case 1
    star_ols = [10, 10]
    assert np.allclose([.5,.5], em.calcMembershipProbs(np.log(star_ols)))

    # case 2
    star_ols = [10, 30]
    assert np.allclose([.25,.75], em.calcMembershipProbs(np.log(star_ols)))

    # case 3
    star_ols = [10, 10, 20]
    assert np.allclose([.25, .25, .5],
                       em.calcMembershipProbs(np.log(star_ols)))


def test_expectation():
    """
    Super basic, generates some association stars along
    with some background stars and checks membership allocation
    is correct
    """

    age = 1e-5
    ass_pars1 = np.array([0, 0, 0, 0, 0, 0, 5., 2., age])
    comp1 = SphereComponent(ass_pars1)
    ass_pars2 = np.array([100., 0, 0, 20, 0, 0, 5., 2., age])
    comp2 = SphereComponent(ass_pars2)
    starcounts = [100,100]
    synth_data = SynthData(pars=[ass_pars1, ass_pars2],
                           starcounts=starcounts)
    synth_data.synthesise_everything()
    tabletool.convert_table_astro2cart(synth_data.table)

    true_memb_probs = np.zeros((np.sum(starcounts), 2))
    true_memb_probs[:starcounts[0], 0] = 1.
    true_memb_probs[starcounts[0]:, 1] = 1.

    # star_means, star_covs = tabletool.buildDataFromTable(synth_data.astr_table)
    # all_lnols = em.getAllLnOverlaps(
    #         synth_data.astr_table, [comp1, comp2]
    # )

    fitted_memb_probs = em.expectation(
            tabletool.build_data_dict_from_table(synth_data.table),
            [comp1, comp2]
    )

    assert np.allclose(true_memb_probs, fitted_memb_probs, atol=1e-10)

