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
THRESHOLD = 0.5
FONTSIZE = 12
MARKSIZE = 150
NOT_BG_ALPHA = 0.6
BG_ALPHA = 0.3
filename = '../data/bpmg_w_nearby_gaia_memberships_magnitudes.fits'

table = Table.read(filename)

# Probabilities for component n are in columns
# 'Comp [n] prob'
# e.g. beta Pictoris is contained in component 0 so
bpmg_rows = table[np.where(table['Comp 0 prob'] > THRESHOLD)]
nearby_gaia = table[np.where(table['Comp 0 prob'] < THRESHOLD)]
# extracts all (primary) with 20% probability of belonging to Chronostar's BPMG

# multiple star systmes are denoted by having a '|' at the beginning of their
# name, and also by the boolean column 'Companion'

# Photometry info is in columns 'phot_g_mean_mag' and 'bp_rp'
# Parallax is in column 'parallax'

xs = (1.5, 3)
ys = (7.5,10.)
m = (ys[1] - ys[0]) / (xs[1] - xs[0])
c = ys[0] - m * xs[0]

def line_eq(x):
    return m*x + c

# paramterising Marusa's main sequence fit
fitpar= [
    0.17954163,
    -2.48748376,
    12.9279348,
    -31.35434182,
    38.31330583,
    -12.25864507,
]
poly=np.poly1d(fitpar)
all_xs = np.linspace(1.0,2.5,100)

abs_mag = bpmg_rows['phot_g_mean_mag']\
          + 5*(np.log10(1e-3*bpmg_rows['parallax'])+1)

# find Chronostar membeers which are photometerically inconsistent
main_seq_stars = np.where(line_eq(bpmg_rows['bp_rp']) < abs_mag)
nearby_abs_mag = nearby_gaia['phot_g_mean_mag'] \
                 + 5*(np.log10(1e-3*nearby_gaia['parallax'])+1)
print(bpmg_rows[main_seq_stars]['source_id'])

non_banyan_mask = np.where(nearby_gaia['Moving group'] != 'beta Pictoris')
banyan_mask = np.where(nearby_gaia['Moving group'] == 'beta Pictoris')

# e.g. (without correcting for distance)
fig, ax = plt.subplots()
ax.scatter(nearby_gaia['bp_rp'][non_banyan_mask],
           nearby_abs_mag[non_banyan_mask], c='black', alpha=BG_ALPHA,
           marker='.',
           label='Nearby Gaia', linewidths=0.1, s=0.4*MARKSIZE)
#ax.savefig('temp_plots/all_photometry.pdf')

ax.plot(all_xs, poly(all_xs), label='Main sequence', color='orange', linewidth=3., ls = '--')


ax.scatter(bpmg_rows['bp_rp'][:35], abs_mag[:35], s=MARKSIZE,  marker='.',
           c='blue', label=r'Confirmed BANYAN $\beta$PMG',
           alpha=NOT_BG_ALPHA, linewidths=0.1)

ax.scatter(bpmg_rows['bp_rp'][35:], abs_mag[35:], c='blue',
           marker='^',
           label=r'New $\mathbf{Chronostar}$ $\beta$PMG', alpha=NOT_BG_ALPHA,
           s=MARKSIZE, linewidths=0.1)
ax.scatter(bpmg_rows['bp_rp'][main_seq_stars], abs_mag[main_seq_stars],
           marker='^',
           c='red', label='Photometric outliers', linewidths=0.1, s=MARKSIZE)
ax.scatter(nearby_gaia['bp_rp'][banyan_mask], nearby_abs_mag[banyan_mask],
           c='magenta', alpha=NOT_BG_ALPHA, marker='.', s=MARKSIZE,
           label='Rejected BANYAN', linewidths=0.1,)
#ax.plot(xs, ys, c='red', ls='--')
# ax.xlim(-1,6)

ax.set_xlim(0,4)
ax.set_ylim(12,-0.5)
ax.set_xlabel(r'$G_{bp} - G_{rp}$')
ax.set_ylabel(r'$M_G$')
ax.legend(loc='best')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(FONTSIZE)

plt.savefig('temp_plots/bpmg_photometry.pdf')
# compared to...
#ax.clf()


