from __future__ import print_function, division

"""
For investigating accuracy of field distribution required to provide
realistic BMPG membership probabilities based on current best fit to group
"""

import numpy as np
import matplotlib as pyplot
import sys

from astropy.io import fits

sys.path.insert(0, '..')
import chronostar.synthesiser as syn

rdir = "../results/em_fit/gaia_dr2_bp/"

# final groups represent the mode of the final sampling stage
final_groups_file = rdir + "final/final_groups.npy"
final_chain0_file = rdir + "final/group0/final_chain.npy"
final_chain1_file = rdir + "final/group1/final_chain.npy"

#gaia_a


