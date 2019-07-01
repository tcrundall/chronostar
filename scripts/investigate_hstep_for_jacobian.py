'''
Investigate the impact (if any) the choice in h step has on
calculating the Jacobian.

The Jacobian is used to transfrom the covariance matrix from
two coordinate systesm. In our case, that's taking the cov
matrix from the initial distribution of stars (at "birth")
through to their current day distribution.
'''

import numpy as np

import sys
sys.path.insert(0, '..')
from chronostar import transform
from chronostar.component import SphereComponent
from chronostar.traceorbit import trace_cartesian_orbit

def get_stable_expons(comp, rtol=1e-2, atol=1e-6, lo_exp=-10, hi_exp=2):
    expons = np.arange(lo_exp,hi_exp)
    # ref_cov_now = comp.get_covmatrix_now()
    ref_cov_now = transform.transform_covmatrix(comp.get_covmatrix(),
                                         comp.trace_orbit_func,
                                         loc=comp.get_mean(),
                                         h=1. * 10. ** -3,
                                         args=(comp.get_age(),))
    covs_now = []
    for expon in expons:
        covs_now.append(
            transform.transform_covmatrix(comp.get_covmatrix(),
                                          comp.trace_orbit_func,
                                          loc=comp.get_mean(),
                                          h=1. * 10. ** expon,
                                          args=(comp.get_age(),))
        )
    return expons[np.where([np.allclose(cov_mat, ref_cov_now,
                                        rtol=rtol, atol=atol)
                            for cov_mat in covs_now])]


mean = np.zeros(6)          # centre at LSR
mean = np.array([20.,  -80., 25., -1.9, 11.76, 2.25])
dx = 10.
dv = 2.
age = 30.
pars = np.hstack((mean, dx, dv, age))

comp = SphereComponent(pars=pars)

results = {}
import matplotlib.pyplot as plt

for label, rtol in zip(['e-4', 'e-3', 'e-2', 'e-1'], [1e-4, 1e-3, 1e-2, 1e-1]):
    all_stable_expons = []

    lo_age = 0
    hi_age = 100
    lo_expon = -10
    hi_expon = 2
    for age in range(lo_age, hi_age):
        pars = np.hstack((mean, dx, dv, age))
        stable_expons = get_stable_expons(SphereComponent(pars),
                                          rtol=rtol,
                                          lo_exp=lo_expon,
                                          hi_exp=hi_expon)
        all_stable_expons.append(stable_expons)
        print('{:5} : {}'.format(age, stable_expons))

    result = np.array([np.isin(np.arange(lo_expon,hi_expon), se)
                       for se in all_stable_expons])
    results[label] = result

    plt.clf()
    plt.imshow(result.T, origin='lower',
               extent=[lo_age-0.5, hi_age-0.5, lo_expon-0.5, hi_expon-0.5])
    plt.xlabel('age [Myr]')
    plt.ylabel('exponent')
    plt.savefig('../plots/h_dependence_{}.png'.format(label))

summed_result = np.sum(results.values(), axis=0)
plt.clf()
plt.imshow(summed_result.T, origin='lower',
           extent=[lo_age-0.5, hi_age-0.5,
                   lo_expon-0.5, hi_expon-0.5])
plt.xlabel('age [Myr]')
plt.ylabel('exponent')
plt.axhline(y=-5, color='red', ls=':', alpha=0.75)
plt.savefig('../plots/h_dependence_all.png')

