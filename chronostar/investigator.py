"""Encapsualtes the results (opt. synthesis and) fitting of data into a
single object for ease of analysis and investigation"""

from __future__ import division, print_function

import numpy as np

import synthesiser as syn
import traceback as tb
import groupfitter as gf
import utils
import pdb

import logging

#mock_group_pars = [20,20,5,-22,-10,-3,5,5,5,3,0.5,0.5,0.3,10,10]

class SynthFit():
   # fixed_age_fits = None
   # free_age_fit = None
   # perf_data = None
   # perf_tb   = None

    def __init__(self, init_group_pars, save_dir='', times=None,
                 nfixed_fits=21, prec=1.0):
        logging.info("Initialising SynthFit object")
        self.init_group_pars_ex = init_group_pars
        self.save_dir = save_dir
        if times is None:
            self.times = np.linspace(0,20,41)
        else:
            self.times = times

        self.ntimes = len(self.times)
        self.nfixed_fits = nfixed_fits
        self.fixed_ages = np.linspace(
            np.min(self.times), np.max(self.times), self.nfixed_fits
            )

        self.fixed_age_fits = []
        self.prec = prec

        self.perf_data_file = save_dir + 'perf_data.dat'
        self.gaia_data_file = save_dir + 'gaia_data.dat'
        self.perf_tb_file = save_dir + 'perf_tb.pkl'
        self.gaia_tb_file = save_dir + 'gaia_tb.pkl'

        self.perf_data = None
        self.gaia_data = None

        self.true_pos_radius = utils.approx_spread_from_sample(
            utils.internalise_pars(self.init_group_pars_ex)
        )

    def synthesise_data(self):
        logging.info("Synthesising data...")
        self.perf_data = syn.synthesise_data(
            1, self.init_group_pars_ex, 1e-5, self.perf_data_file,
            return_data=True
        )
        self.gaia_data = syn.synthesise_data(
            1, self.init_group_pars_ex, self.prec, self.gaia_data_file,
            return_data=True
        )

    def calc_tracebacks(self):
        tb.traceback(
            self.perf_data, self.times, savefile=self.perf_tb_file,
        )
        tb.traceback(
            self.gaia_data, self.times, savefile=self.gaia_tb_file,
        )

    def perform_free_fit(self, plot_it=False, period=1000):
        logging.info("Performing free age fit..")
        self.free_age_fit = FreeFit(self.gaia_tb_file)
        self.free_age_fit.run_fit(
            plot_it=plot_it, burnin_steps=period, sampling_steps=period
        )
        logging.info("..completed free age fit")

    def analyse_free_fit(self):
        logging.info("Analysing free age fit..")
        if self.free_age_fit is None:
            print("Need to first run 'perform_free_fit'...")
        # do something with the free fit internal inspection etc
        self.free_age_fit.analyse()
        logging.info("..completed analysis of free age fit")

    def perform_fixed_fits(self, period=1000):
        logging.info("Performing fixed age fits")
        self.fixed_age_fits = []
        for fixed_age in self.fixed_ages:
            logging.info("Initialising fit object for {}".format(fixed_age))
            self.fixed_age_fits.append(
                FixedFit(self.gaia_tb_file, fixed_age)
            )
            #pdb.set_trace()
        for fixed_fit in self.fixed_age_fits:
            fixed_fit.run_fit(burnin_steps=period, sampling_steps=period)

    def analyse_fixed_fits(self):
        for fixed_fit in self.fixed_age_fits:
            logging.info(
                "Analysing fixed age {}".format(fixed_fit.fixed_age)
            )
            fixed_fit.analyse()

    def gather_results(self):
        self.bayes_spreads = []
        self.naive_spreads = []
        for fixed_age_fit in self.fixed_age_fits:
            self.bayes_spreads.append(fixed_age_fit.mean_radius)
            self.naive_spreads.append(fixed_age_fit.naive_spread)

    def investigate(self, plot_it=False, period=1000):
        self.synthesise_data()
        self.calc_tracebacks()
        self.perform_free_fit(plot_it=plot_it, period=period)
        self.analyse_free_fit()
        self.perform_fixed_fits(period=period)
        self.analyse_fixed_fits()
        self.gather_results()


class FreeFit():
    def __init__(self, tb_file):
        self.tb_file = tb_file

    def run_fit(self, plot_it=False, burnin_steps=1000, sampling_steps=1000):
        self.best_like_fit, self.chain, lnprob =\
            gf.fit_group(
                self.tb_file, plot_it=plot_it, burnin_steps=burnin_steps,
                sampling_steps=sampling_steps
            )

        self.nwalkers, self.nsteps, self.npars = self.chain.shape
        flat_chain = np.reshape(self.chain, (-1, self.npars))
        self.nsamples = self.nwalkers*self.nsteps

        cov_mats = self.calc_covs(flat_chain)
        self.calc_pos_radii(cov_mats)
        #self.calc_phase_radius(cov_mats)
        self.calc_mean_fit(cov_mats, flat_chain)

    def calc_covs(self, flat_chain):
        cov_mats = np.zeros((self.nsamples, 6, 6))
        for i in range(self.nsamples):
            cov_mats[i] = utils.generate_cov(flat_chain[i])
        return cov_mats

    def calc_pos_radii(self, cov_mats):
        """Calculate the determinants of the positional description

        Used as a proxy for the effective radius of the 3D ellipse since the
        determinant is the product of eigen values, and eigen values are related
        to the axes lengths of ellipses.

        TODO: Write out maths that connects eigenvalues to axes lengths
        """
        self.pos_radii = np.zeros(self.nsamples)
        for i, cov_mat in enumerate(cov_mats):
            # since eigen values are dX^2 maybe need to take the 1/6 root???
            self.pos_radii[i] = (np.linalg.det(cov_mat[:3,:3]) ** (1/6))
        self.mean_radius = np.mean(self.pos_radii)

    def calc_phase_radius(self, cov_mats):
        """Not sure if useful...."""
        self.phase_radii = np.zeros(self.nsamples)
        for i, cov_mat in enumerate(cov_mats):
            self.phase_radii[i] = np.linalg.det(cov_mat) ** (1/6)

    def calc_mean_fit(self, cov_mats, flat_chain):
        self.mean_mu = np.mean(flat_chain[:,:6], axis=0)
        self.mean_sigma = np.mean(cov_mats, axis=0)
        self.mean_age = np.mean(flat_chain[:,-1])

    def analyse(self):
        pass

class FixedFit(FreeFit):
    def __init__(self, tb_file, fixed_age):
        FreeFit.__init__(self, tb_file)
        self.fixed_age = fixed_age

    def run_fit(self, plot_it=False, burnin_steps=1000, sampling_steps=1000):
        self.best_like_fit, chain, self.lnprob =\
            gf.fit_group(
                self.tb_file, fixed_age=self.fixed_age, burnin_steps=1000,
                sampling_steps=1000
            )
        self.nwalkers, self.nsteps, self.npars = chain.shape
        flat_chain = np.reshape(chain, (-1, self.npars))
        self.nsamples = self.nwalkers*self.nsteps
        logging.info("Fixed fit completed for {}".format(self.fixed_age))

        cov_mats = self.calc_covs(flat_chain)
        self.calc_pos_radii(cov_mats)
        self.calc_mean_fit(cov_mats, flat_chain)

    def calc_true_fit(self, perf_tb):
        """
        describe the group's history from the traceback of "perfect" astrometry
        """
        # get xyzuvw for each star at each traceback age
        # take mean of each xyzuvw
        # fit a covariance matrix to xyzuvw points
        pass

    def calc_naive_fit(self):
        """Fit a covariance matrix to the measured gaia stellar xyzuvw
        """
        star_pars = gf.read_stars(self.tb_file)
        _, xyzuvw_interp = gf.interp_cov(
            target_time=self.fixed_age, star_pars=star_pars
        )
        naive_group_mean = np.mean(xyzuvw_interp, axis=0)
        naive_group_cov = np.cov(xyzuvw_interp.T)

        self.naive_spread = np.linalg.det(naive_group_cov[:3,:3]) ** (1/6)

    def analyse(self):
        FreeFit.analyse(self)
        self.calc_naive_fit()
        logging.info(
            "Fixed age analysis done for {}".format(self.fixed_age)
        )






