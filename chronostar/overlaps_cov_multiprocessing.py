"""
Author: Marusa Zerjal, 2019 - 07 - 18

"""

from multiprocessing import Pool
import contextlib
import time

import scipy
_SCIPY_VERSION= [int(v.split('rc')[0])
                 for v in scipy.__version__.split('.')]
if _SCIPY_VERSION[0] == 0 and _SCIPY_VERSION[1] < 10:
    from scipy.maxentropy import logsumexp
elif ((_SCIPY_VERSION[0] == 1 and _SCIPY_VERSION[1] >= 3) or
    _SCIPY_VERSION[0] > 1):
    from scipy.special import logsumexp
else:
    from scipy.misc import logsumexp

try:
    print('Using C implementation in expectmax')
    from _overlap import get_lnoverlaps
except:
    print("WARNING: Couldn't import C implementation, using slow pythonic overlap instead")
    logging.info("WARNING: Couldn't import C implementation, using slow pythonic overlap instead")
    from chronostar.likelihood import slow_get_lnoverlaps as get_lnoverlaps

class Bg_ols_cov_multiprocessing():
    def __init__(self, background_means, background_covs, nstars, star_covs, star_means):
        self.star_means = star_means
        self.star_covs = star_covs
        self.background_means = background_means
        self.background_covs = background_covs
        self.nstars = nstars


    def func(self, index):
        """
        Author: Marusa Zerjal, 2019 - 07 - 18

        Multiprocessing function should be pickable

        :param index:
        :return:
        """
        star_mean = self.star_means[index]
        star_cov = self.star_covs[index]
        print(star_mean, star_cov)
        try:
            #print('***********', nstars, star_cov, star_mean, background_covs, background_means)
            bg_lnol = get_lnoverlaps(star_cov, star_mean, self.background_covs,
                                     self.background_means, self.nstars)
            #print('intermediate', bg_lnol)
            # bg_lnol = np.log(np.sum(np.exp(bg_lnol))) # sum in linear space
            bg_lnol = logsumexp(bg_lnol) # sum in linear space

        # Do we really want to make exceptions here? If the sum fails then
        # there's something wrong with the data.
        except:
            # TC: Changed sign to negative (surely if it fails, we want it to
            # have a neglible background overlap?
            print('bg ln overlap failed, setting it to -inf')
            bg_lnol = -np.inf
        #bg_lnols.append(bg_lnol)
        #print(bg_lnol)
        #print('')
        return bg_lnol

    def compute_bg_ols(self):
        num_threads = 8
        start = time.time()
        # ~ with contextlib.closing( Pool(num_threads) ) as pool:
        #with Pool(num_threads) as pool:
        indices = range(len(self.star_means))
        print('indices', indices)
        with contextlib.closing(Pool(num_threads)) as pool:
            #results = pool.map(func, zip(star_means, star_covs))
            results = pool.map(self.func, indices)
        end = time.time()
        print(end - start, 'multiprocessing')
        print('results', results)
        return results