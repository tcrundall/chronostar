class FittingResult():
    def __init__(self, init_pars=None, init_times=None, lnprobs=None,
                 chains=None, naive_spreads=None, bayes_spreads=None,
                 time_probs=None, free_chain=None, free_lnprob=None,
                 emcee_pars=None):
        """
        Initialising the object

        Parameters
        ----------
        init_pars : [15] array
            [X,Y,Z,U,V,W,dX,dY,dZ,dV,Cxy,Cxz,Cyz,age,nstars]
            These parameters inform the distribution from which the stars are
            initially drawn
        init_times : [n_init_times] array
            The time steps for which traceback has been calculated. Need not
            necessarily be the same set of times upon which the fixed_age fits
            are later performed.
        lnprobs : [ntimes, nwalkers, nsteps] array
            An array of emcee sampler.lnprobability arrays, with one for each
            fixed_time fit
        chains : [ntimes, nwalkers, nsteps, npars]
            An array of emcee sampler.chain arrays, with one for each fixed_time
            fit
        naive_spreads : [ntimes]
            The idealised spherical radius of the gaussian fit to the "exact"
            orbits
        bayes_spreads : [ntimes] array
            The idealised spherical radius of the position component of the
            bayesian fit
        time_probs : [ntimes] array
            scaled average likelihoods of the samples for each fixed time fit
        free_chain : [nwalkers, nsteps, npars] array
            The emcee sampler.chain array for a single emcee fit with
            unrestricted time parameter
        free_lnprob : [nwalkers, nsteps] array
            The emcee sampler.lnprobability for a single emcee fit with
            unrestricted time parameter
        emcee_pars : dict
            nburnin : int
                number of steps to be performed in burnin stage
            nsteps : int
                number of steps to be performed in sampling stage
            nwalkers : int
                number of walkers to be used in fit [must be EVEN and >=2*npar]
        """
        self.init_pars = init_pars
        self.init_times = init_times
        self.lnprobs = lnprobs
        self.chains = chains
        self.naive_spreads = naive_spreads
        self.bayes_spreads = bayes_spreads
        self.time_probs = time_probs
        self.free_chain = free_chain
        self.free_lnprob = free_lnprob
        self.emcee_pars = emcee_pars

    def __str__(self):
        return "Fitting result object"

    def print_details(self):
        print("Data generated from:\n{}\n".format(self.init_pars))
        print("with traceback timesteps:\n{}\n".format(self.init_times))
        print("using emcee parameters:\n{}\n".format(self.emcee_pars))
