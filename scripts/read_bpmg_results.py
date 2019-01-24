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
# filename = '../data/bpmg_w_nearby_gaia_memberships_magnitudes.fits'
filename = '../data/beta_Pictoris_with_gaia_small_everything.fits'
cmd_filename = '../data/MIST_iso_5c47c57b3d6e7.iso.cmd'

# EXTRACT MIST ISOCHRONES
table = Table.read(filename)
cmd = Table.read(cmd_filename, format='ascii.commented_header',
                 header_start=-1)
cmd_young_age = 7.3
cmd_young = cmd[np.where(cmd['log10_isochrone_age_yr'] == cmd_young_age)]
cmd_old_age = 9.
cmd_old = cmd[np.where(cmd['log10_isochrone_age_yr'] == cmd_old_age)]

# CALCULATE ABSOLUTE MAGNITUDES
# parallax is in [mas]
table['abs_g_mag'] = table['phot_g_mean_mag'] \
                 + 5*(np.log10(1e-3*table['parallax'])+1)


# PARTITION DATASET INTO BPMG MEMBERS AND BACKGROUND AS DETERMINED BY CHRONOSTAR
bpmg_rows = table[np.where(table['comp_A'] > THRESHOLD)]
nearby_gaia = table[np.where(table['comp_A'] < THRESHOLD)]

def line_eq(x):
    """
    Line equation for photometric rejects
    """
    xs = (1.5, 3)
    ys = (7.5,10.)
    m = (ys[1] - ys[0]) / (xs[1] - xs[0])
    c = ys[0] - m * xs[0]
    return m*x + c

# # paramterising Marusa's main sequence fit
# fitpar= [
#     0.17954163,
#     -2.48748376,
#     12.9279348,
#     -31.35434182,
#     38.31330583,
#     -12.25864507,
# ]
# poly=np.poly1d(fitpar)
# all_xs = np.linspace(1.0,2.5,100)

# IDENTIFY PHOTOMETRICALLY INCONSISTENT BPMG STARS
main_seq_stars = np.where(line_eq(bpmg_rows['bp_rp']) < bpmg_rows['abs_g_mag'])

non_banyan_mask = np.where(nearby_gaia['Moving group'] != 'beta Pictoris')
banyan_mask = np.where(nearby_gaia['Moving group'] == 'beta Pictoris')

# ---------- PLOTTING ----------
fig, ax = plt.subplots()

# Plotting isochrones
ax.plot(cmd_young['Gaia_BP_DR2Rev'] - cmd_young['Gaia_RP_DR2Rev'],
        cmd_young['Gaia_G_DR2Rev'],
        label='20 Myr', color='yellow', linewidth=3., ls='--')
ax.plot(cmd_old['Gaia_BP_DR2Rev'] - cmd_old['Gaia_RP_DR2Rev'],
        cmd_old['Gaia_G_DR2Rev'],
        label='1 Gyr', color='orange', linewidth=3., ls='--')

# plotting background gaia stars
ax.scatter(nearby_gaia['bp_rp'][non_banyan_mask],
           nearby_gaia['abs_g_mag'][non_banyan_mask],
           c='black',
           alpha=BG_ALPHA,
           marker='.',
           label='Nearby Gaia',
           linewidths=0.1,
           s=0.4*MARKSIZE)


ax.scatter(bpmg_rows['bp_rp'][:35], bpmg_rows['abs_g_mag'][:35], s=MARKSIZE,  marker='.',
           c='blue', label=r'Confirmed BANYAN $\beta$PMG',
           alpha=NOT_BG_ALPHA, linewidths=0.1)

ax.scatter(bpmg_rows['bp_rp'][35:], bpmg_rows['abs_g_mag'][35:], c='blue',
           marker='^',
           label=r'New $\mathbf{Chronostar}$ $\beta$PMG', alpha=NOT_BG_ALPHA,
           s=MARKSIZE, linewidths=0.1)
ax.scatter(bpmg_rows['bp_rp'][main_seq_stars], bpmg_rows['abs_g_mag'][main_seq_stars],
           marker='^',
           c='red', label='Photometric outliers', linewidths=0.1, s=MARKSIZE)
ax.scatter(nearby_gaia['bp_rp'][banyan_mask], nearby_gaia['abs_g_mag'][banyan_mask],
           c='magenta', alpha=NOT_BG_ALPHA, marker='.', s=MARKSIZE,
           label='Rejected BANYAN', linewidths=0.1,)

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


