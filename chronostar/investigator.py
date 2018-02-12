"""Encapsualtes the results (opt. synthesis and) fitting of data into a
single object for ease of analysis and investigation"""

import numpy as np

import chronostar.synthesiser as syn
import chronostar.traceback as tb
import chronostar.groupfitter as gf

class SynthFit():
    ntimes = None
    times = None
    init_group_pars_ex = None
    fixed_age_fits = None
    nfixed_age = None
    free_age_fit = None
    perf_data = None
    perf_tb   = None
    gaia_data = None
    gaia_tb   = None
    save_dir  = None

    perf_tb_file   = None
    gaia_data_file = None
    gaia_tb_file   = None
    save_dir_file  = None

    def __init__(self, init_group_pars, save_dir='', times=None,
                 nfixed_fits=21):
        self.init_group_pars_ex = init_group_pars
        self.save_dir = save_dir
        if times is None:
            self.times = np.linspace(0,20,41)
        else:
            self.times = times

        self.ntimes = len(self.times)
        self.nfixed_fits = nfixed_fits
        self.fixed_age_fits = []

        self.perf_data_file = save_dir + 'perf_data.dat'
        self.gaia_data_file = save_dir + 'gaia_data.dat'
        self.perf_tb_file = save_dir + 'perf_tb.dat'
        self.gaia_tb_file = save_dir + 'gaia_tb.dat'

    def synthesise_data(self, save_dir=''):
        self.perf_data = syn.synthesise_data(
            1, self.init_group_pars, 1e-5, self.perf_data_file
        )
        self.gaia_data = syn.synthesise_data(
            1, self.init_group_pars, 1e-5, self.gaia_data_file
        )

    def calc_tracebacks(self, times=None):
        self.perf_tb = tb.traceback(
            self.perf_data, self.times, savefile=self.perf_tb_file,
            return_tb=True
        )
        self.gaia_tb = tb.traceback(
            self.gaia_data, self.times, savefile=self.gaia_tb_file,
            return_tb=True
        )

    def perform_free_fit(self):
        self.free_age_fit = FreeFit(self.gaia_tb_file, self.gaia_tb)

    def analyse_free_fit(self):
        if self.free_age_fit is None:
            print("Need to first run 'perform_free_fit'...")
        # do something with the free fit internal inspection etc
        pass

    def perform_fixed_fits(self):
        for ix in range(self.nfixed_fits):
            self.fixed_age_fits.append(FxedFit(self.gaia_tb_file, self.gaia_tb))

    def analyse_fixed_fits(self):
        for fixed_fit in self.fixed_age_fits:
            fixed_fit.analyse()

#class SingleFit():

