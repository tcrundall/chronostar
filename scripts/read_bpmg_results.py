from __future__ import print_function, division

"""
Simple script to load in astropy table of beta Pictoris run with nearby
Gaia stars.
"""

from astropy.table import Table
from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt

# CHANGE FILENAME AS NEEDED
filename = '../data/bpmg_w_nearby_gaia_memberships_magnitudes.fits'

table = Table.read(filename)

# Probabilities for component n are in columns
# 'Comp [n] prob'
# e.g. beta Pictoris is contained in component 0 so
bpmg_rows = table[np.where(table['Comp 0 prob'] > 0.2)]
# extracts all (primary) with 20% probability of belonging to Chronostar's BPMG

# multiple star systmes are denoted by having a '|' at the beginning of their
# name, and also by the boolean column 'Companion'

# Photometry info is in columns 'phot_g_mean_mag' and 'bp_rp'
# Parallax is in column 'parallax'

# e.g. (without correcting for distance)
plt.clf()
plt.plot(bpmg_rows['bp_rp'], bpmg_rows['phot_g_mean_mag'], '.')
plt.xlim(-1,5)
plt.ylim(16,-5)
plt.savefig('temp_plots/bpmg_photometry.pdf')
# compared to...
plt.clf()
plt.plot(table['bp_rp'], table['phot_g_mean_mag'], '.')
plt.xlim(-1,5)
plt.ylim(16,-5)
plt.savefig('temp_plots/all_photometry.pdf')


