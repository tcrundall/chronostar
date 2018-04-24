from __future__ import division, print_function

import numpy as np

# not sure if this is right, needed to change to this so I could run
# investigator
from _overlap import get_lnoverlaps
#from chronostar._overlap import get_lnoverlaps
import corner
import emcee
import logging
import matplotlib.pyplot as plt
import pickle
from chronostar import transform as tf

from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from galpy.util import bovy_conversion
#from galpy.util import bovy_coords

try:
    import astropy.io.fits as pyfits
except ImportError:
    import pyfits

# ----- UTILITY FUNCTIONS -----------

def generate_cov(pars):
    """Generate covariance matrix from standard devs and correlations

    Parameters
    ----------
    pars : [9] array
        [X,Y,Z,U,V,W,dX,dV,age]

    Returns
    -------
    cov
        [6, 6] array : covariance matrix for group model or stellar pdf
    """
    dX, dV = np.exp(np.array(pars[6:8]))
    cov = np.eye(6)
    cov[0:3] *= dX**2
    cov[3:6] *= dV**2
    return cov

# ------- DUPLICATED FROM TRACEBACK CAUSE OF ISSUES --------
def get_lsr_orbit(times, Potential=MWPotential2014):
    """Integrate the local standard of rest backwards in time, in order to provide
    a reference point to our XYZUVW coordinates.

    Parameters
    ----------
    times: numpy float array
        Times at which we want the orbit of the local standard of rest.

    Potential: galpy potential object.
    """
    ts = -(times / 1e3) / bovy_conversion.time_in_Gyr(220., 8.)
    lsr_orbit = Orbit(vxvv=[1., 0, 1, 0, 0., 0], vo=220, ro=8,
                      solarmotion='schoenrich')
    lsr_orbit.integrate(ts, Potential)  # ,method='odeint')
    return lsr_orbit

def trace_forward(xyzuvw, time_in_past,
                  Potential=MWPotential2014, solarmotion=None):
    """Trace forward one star in xyzuvw coords

    Parameters
    ----------
    xyzuvw: numpy float array
        xyzuvw relative to the local standard of rest at some point in the past.

    time_in_past: float
        Time in the past that we want to trace forward to the present.
    """
    if solarmotion == None:
        xyzuvw_sun = np.zeros(6)
    elif solarmotion == 'schoenrich':
        xyzuvw_sun = [0, 0, 25, 11.1, 12.24, 7.25]
    else:
        raise UserWarning
    if time_in_past == 0:
        time_in_past = 1e-5
        # print("Time in past must be nonzero: {}".format(time_in_past))
        # raise UserWarning
    times = np.linspace(0, time_in_past, 2)

    # Start off with an LSR orbit...
    lsr_orbit = get_lsr_orbit(times)

    # Convert times to galpy units:
    ts = -(times / 1e3) / bovy_conversion.time_in_Gyr(220., 8.)

    # Add on the lsr_orbit to get the times in the past.
    xyzuvw_gal = np.zeros(6)
    xyzuvw_gal[0] = xyzuvw[0] / 1e3 + lsr_orbit.x(ts[-1]) - lsr_orbit.x(ts[0])
    xyzuvw_gal[1] = xyzuvw[1] / 1e3 + lsr_orbit.y(ts[-1]) - lsr_orbit.y(ts[0])
    # Relative to a zero-height orbit, excluding the solar motion.
    xyzuvw_gal[2] = xyzuvw[2] / 1e3
    xyzuvw_gal[3] = xyzuvw[3] + lsr_orbit.U(ts[-1]) - lsr_orbit.U(ts[0])
    xyzuvw_gal[4] = xyzuvw[4] + lsr_orbit.V(ts[-1]) - lsr_orbit.V(ts[0])
    xyzuvw_gal[5] = xyzuvw[5] + lsr_orbit.W(ts[-1]) - lsr_orbit.W(ts[0])

    # Now convert to units that galpy understands by default.
    # FIXME: !!! Reverse [0] sign here.
    l = np.degrees(np.arctan2(xyzuvw_gal[1], -xyzuvw_gal[0]))
    b = np.degrees(
        np.arctan2(xyzuvw_gal[2], np.sqrt(np.sum(xyzuvw_gal[:2] ** 2))))
    dist = np.sqrt(np.sum(xyzuvw_gal[:3] ** 2))
    # As we are already in LSR co-ordinates, put in zero for solar motion and
    #  zo.
    o = Orbit([l, b, dist, xyzuvw_gal[3], xyzuvw_gal[4], xyzuvw_gal[5]],
              solarmotion=[0, 0, 0], zo=0, lb=True, uvw=True, vo=220, ro=8)

    # Integrate this orbit forwards in time.
    o.integrate(-ts, Potential)  # ,method='odeint')

    # Add in the solar z and the solar motion.
    xyzuvw_now = np.zeros(6)
    xyzuvw_now[0] = 1e3 * (o.x(-ts[-1]) - lsr_orbit.x(0))
    xyzuvw_now[1] = 1e3 * o.y(-ts[-1])
    xyzuvw_now[2] = 1e3 * o.z(-ts[-1]) - xyzuvw_sun[2]
    xyzuvw_now[3] = o.U(-ts[-1]) - xyzuvw_sun[3]
    xyzuvw_now[4] = o.V(-ts[-1]) - xyzuvw_sun[4]
    xyzuvw_now[5] = o.W(-ts[-1]) - xyzuvw_sun[5]

    # pdb.set_trace()

    return xyzuvw_now

# ------- MAIN FUNCTIONS -----------

def read_stars(tb_file):
    """Read stars from traceback file into a dictionary.

    The input is an error ellipse in 6D (X,Y,Z,U,V,W) of a list of stars at
    a bucn of times in the past.

    TODO: MODIFY THIS SO ONLY READING IN STARS AT t=now

    Parameters
    ----------
    tb_file: string
        input file with values to be wrapped as dictionary

    Returns
    -------
    star_dictionary : dict
        stars: (nstars) high astropy table including columns as
                    documented in the Traceback class.
        times: (ntimes) numpy array, containing times that have
                    been traced back, in Myr
        xyzuvw: (nstars,ntimes,6) numpy array, XYZ in pc and UVW in km/s
        xyzuvw_cov: (nstars,ntimes,6,6) numpy array, covariance of xyzuvw
    """
    if len(tb_file) == 0:
        print("Input a filename...")
        raise UserWarning

    # Stars is an astropy.Table of stars
    if tb_file[-3:] == 'pkl':
        with open(tb_file, 'r') as fp:
            (stars, times, xyzuvw, xyzuvw_cov) = pickle.load(fp)
    elif (tb_file[-3:] == 'fit') or (tb_file[-4:] == 'fits'):
        stars = pyfits.getdata(tb_file, 1)
        times = pyfits.getdata(tb_file, 2)
        xyzuvw = pyfits.getdata(tb_file, 3)
        xyzuvw_cov = pyfits.getdata(tb_file, 4)
    else:
        print("Unknown File Type!")
        raise UserWarning

    return dict(stars=stars, times=times, xyzuvw=xyzuvw, xyzuvw_cov=xyzuvw_cov)


def lnprior(pars, star_pars):
    """Computes the prior of the group models constraining parameter space

    Parameters
    ----------
    pars
        Parameters describing the group model being fitted
    star_pars
        traceback data being fitted to
    (retired) z
        array of weights [0.0 - 1.0] for each star, describing how likely
        they are members of group to be fitted.

    Returns
    -------
    lnprior
        The logarithm of the prior on the model parameters

    TODO: Incorporate star determinants
    """
    # fetch maximum allowed age
    max_age = 200

    means = pars[0:6]
    stds = np.exp(pars[6:8])
    age = pars[8]

    if np.min(means) < -1000 or np.max(means) > 1000:
        return -np.inf
    if np.min(stds) <= 0.0 or np.max(stds) > 1000.0:
        return -np.inf
    if age < 0.0 or age > max_age:
        return -np.inf
    return 0.0


def lnlike(pars, star_pars, z=None, return_lnols=False):
    """Computes the log-likelihood for a fit to a group.

    The emcee parameters encode the modelled origin point of the stars. Using
    the parameters, a mean and covariance in 6D space are constructed as well as
    an age. The kinematics are then projected forward to the current age and
    compared with the current stars' XYZUVW values (and uncertainties)

    Parameters
    ----------
    pars
        Parameters describing the group model being fitted
    star_pars
        traceback data being fitted to
    (retired) z
        array of weights [0.0 - 1.0] for each star, describing how likely
        they are members of group to be fitted.

    Returns
    -------
    lnlike
        the logarithm of the likelihood of the fit
    """
    # convert pars into covariance matrix
    mean_then = pars[0:6]
    cov_then = generate_cov(pars)
    age = pars[8]

    mean_now = trace_forward(mean_then, age)
    cov_now = tf.transform_cov(
        cov_then, trace_forward, mean_then, dim=6, args=(age,)
    )

    star_covs = star_pars['xyzuvw_cov'][:,0]
    star_mns  = star_pars['xyzuvw'][:,0]
    nstars = star_mns.shape[0]

    lnols = get_lnoverlaps(
        cov_now, mean_now, star_covs, star_mns, nstars
    )
    if return_lnols:
        return lnols

    # prior on covariance matrix incorporated into parametrisation of dX and dV
    return np.sum(lnols * z)


def lnprobfunc(pars, star_pars, z):
    """Computes the log-probability for a fit to a group.

    Parameters
    ----------
    pars
        Parameters describing the group model being fitted
        0,1,2,3,4,5, 6, 7,  8
        X,Y,Z,U,V,W,dX,dV,age
    star_pars
        traceback data being fitted to
    (retired) z
        array of weights [0.0 - 1.0] for each star, describing how likely
        they are members of group to be fitted.

    Returns
    -------
    logprob
        the logarithm of the posterior probability of the fit
    """
    #global N_FAILS
    #global N_SUCCS

    lp = lnprior(pars, star_pars)
    if not np.isfinite(lp):
        #N_FAILS += 1
        return -np.inf
    #N_SUCCS += 1
    return lp + lnlike(pars, star_pars, z)

def burnin_convergence(lnprob, tol=0.1, slice_size=100, cutoff=0):
    """Checks early lnprob vals with final lnprob vals for convergence

    Parameters
    ----------
    lnprob : [nwalkers, nsteps] array

    tol : float
        The number of standard deviations the final mean lnprob should be within
        of the initial mean lnprob

    slice_size : int
        Number of steps at each end to use for mean lnprob calcultions

    cuttoff : int
        Step number at which to start the analysis. i.e., a value of 0 would
        mean to include the whole chain, whereas a value of 500 would be to
        only confirm no deviation in the steps [500:END]
    """
    # take a chunk the smaller of 100 or half the chain
    if lnprob.shape[1] < slice_size:
        slice_size = int(round(0.5*lnprob.shape[1]))

    start_lnprob_mn = np.mean(lnprob[:,:slice_size])
    start_lnprob_std = np.std(lnprob[:,:slice_size])

    end_lnprob_mn = np.mean(lnprob[:, -slice_size:])
    end_lnprob_std = np.std(lnprob[:, -slice_size:])

    return np.isclose(start_lnprob_mn, end_lnprob_mn, atol=tol*end_lnprob_std)

def fit_group(tb_file, z=None, burnin_steps=1000, plot_it=False, pool=None,
              init_pars=None, convergence_tol=0.1, tight=False, init_pos=None):
    """Fits a single gaussian to a weighted set of traceback orbits.

    Parameters
    ----------
    tb_file : string
        a '.pkl' or '.fits' file containing traceback orbits as a dictionary:
            stars: (nstars) high astropy table including columns as
                        documented in the Traceback class.
            times: (ntimes) numpy array, containing times that have
                        been traced back, in Myr
            xyzuvw (nstars,ntimes,6) numpy array, XYZ in pc and UVW in km/s
            xyzuvw_cov (nstars,ntimes,6,6) numpy array, covariance of xyzuvw
    z
        array of weights [0.0 - 1.0] for each star, describing how likely
        they are members of group to be fitted.

    burnin_steps : int {1000}

    plot_it : bool {False}

    pool : MPIPool object {None}
        pool of threads to execute walker steps concurrently

    Returns
    -------
    best_sample
        The parameters of the group model which yielded the highest posterior
        probability

    chain
        [nwalkers, nsteps, npars] array of all samples

    probability
        [nwalkers, nsteps] array of probabilities for each sample
    """
    #global N_FAILS
    #global N_SUCCS

    #            X,Y,Z,U,V,W,lndX,lndV,age
    if tight:
        INIT_SDEV = [5,5,5,2,2,2, 0.2, 0.2,0.5]
    else:
        INIT_SDEV = [20,20,20,5,5,5, 0.5, 0.5,1]
    star_pars = read_stars(tb_file)

    # initialise z if needed as array of 1s of length nstars
    if z is None:
        z = np.ones(star_pars['xyzuvw'].shape[0])

    # Initialise the fit
#    if init_pars is None:
    init_pars = [0,0,0,0,0,0,3.,2.,10.0]

    NPAR = len(init_pars)
    NWALKERS = 2 * NPAR

    # Since emcee linearly interpolates between walkers to determine next step
    # by initialising each walker to the same age, the age never varies
#    if fixed_age is not None:
#        init_pars[-1] = fixed_age
#        INIT_SDEV[-1] = 0.0


    # Whole emcee shebang
    sampler = emcee.EnsembleSampler(
        NWALKERS, NPAR, lnprobfunc, args=[star_pars, z], pool=pool,
    )
    # initialise walkers, note that INIT_SDEV is carefully chosen such that
    # all generated positions are permitted by lnprior
    if init_pos is None:
        pos = [
            init_pars + (np.random.random(size=len(INIT_SDEV)) - 0.5) * INIT_SDEV
            for _ in range(NWALKERS)
        ]
    else:
        pos = init_pos

    # Perform burnin
    state = None
    converged = False
    cnt = 0
    logging.info("Beginning burnin loop")
    burnin_lnprob_res = np.zeros((NWALKERS,0))
    while not converged:
        logging.info("Burning in cnt: {}".format(cnt))
        sampler.reset()
        pos, lnprob, state = sampler.run_mcmc(pos, burnin_steps, state)
        converged = burnin_convergence(sampler.lnprobability, tol=convergence_tol)

        # Help out the struggling walkers
        best_ix = np.argmax(lnprob)
        poor_ixs = np.where(lnprob < np.percentile(lnprob, 33))
        for ix in poor_ixs:
            pos[ix] = pos[best_ix]

        if plot_it:
            #plt.clf()
            #plt.plot(sampler.lnprobability)
            #plt.savefig("burnin_lnprob{}.png".format(cnt))
            plt.clf()
            plt.plot(sampler.lnprobability.T)
            plt.savefig("burnin_lnprobT{}.png".format(cnt))

        burnin_lnprob_res = np.hstack((
            burnin_lnprob_res, sampler.lnprobability
        ))
        cnt += 1

    logging.info("Burnt in, with convergence: {}\n"
          "Taking final burnin segment as sampling stage".format(converged))
    # save the chain for later inspection
    np.save("final_chain.npy", sampler.chain)
    np.save("final_lnprob.npy", lnprob)
    if plot_it:
#        plt.clf()
#        plt.plot(burnin_lnprob_res)
#        plt.savefig("burnin_lnprob.png")
        plt.clf()
        plt.plot(burnin_lnprob_res.T)
        plt.savefig("burnin_lnprobT.png")

#    print("Sampling")
#    pos, final_lnprob, rstate = sampler.run_mcmc(
#        pos, sampling_steps, rstate0=state,
#    )
#    print("Sampled")
    #    print("Number of failed priors after sampling:\n{}".format(N_FAILS))
    #    print("Number of succeeded priors after sampling:\n{}".format(N_SUCCS))
    if plot_it:
#        plt.clf()
#        plt.plot(sampler.lnprobability)
#        plt.savefig("lnprob.png")
        plt.clf()
        plt.plot(sampler.lnprobability.T)
        plt.savefig("lnprobT.png")

    # sampler.lnprobability has shape (NWALKERS, SAMPLE_STEPS)
    # yet np.argmax takes index of flattened array
    final_best_ix = np.argmax(sampler.lnprobability)

    best_sample = sampler.flatchain[final_best_ix]

    # corner plotting is heaps funky on my laptop....
    # am able to make it plot a nice square plot, but it affects
    # the dimensions of ALL plots that come afterwards....
    if False:
        logging.info("Making corner plot")
        plt.clf()
        fig1, axes = plt.subplots(9,9)
        fig1.set_size_inches(20,20)
        corner.corner(sampler.flatchain, truths=best_sample, fig=fig1)
        fig1.savefig("corner.png")
        logging.info("Made corner plot")

    return best_sample, sampler.chain, sampler.lnprobability
