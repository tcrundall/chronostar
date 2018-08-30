from __future__ import print_function, division

"""
This script calculates the correction factor to estimate stellar density
if Gaia was sensitive to the dimmest star in catalogue.
Depending on line fit, correction factor is about 15.3 (+ 7 - 1.5),
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
all_gaia_file = '../data/all_rvs_w_ok_plx.fits'

hdul = fits.open(all_gaia_file)

g_key = 'phot_g_mean_mag'
g_fluxes = hdul[1].data[g_key]

flux_hist = np.histogram(g_fluxes, bins=40)
ln_counts = np.log(flux_hist[0])
bin_width = flux_hist[1][1] - flux_hist[1][0]


# upon review, the slope is approximately constant
# between mags 4 and 12

# first order, fit a line between the points at mag 4 and mag 12
approx_f1 = 10
approx_f2 = 12
f1_ix = np.digitize(approx_f1, flux_hist[1])
f2_ix = np.digitize(approx_f2, flux_hist[1])

f1 = flux_hist[1][f1_ix]
ln_n1 = ln_counts[f1_ix]
f2 = flux_hist[1][f2_ix]
ln_n2 = ln_counts[f2_ix]

approx_slope = (ln_n2 - ln_n1) / (f2 - f1)
approx_interp = ln_n1 - approx_slope * f1

def lnflux_dens_fit(flux, slope, interp):
    return slope * flux + interp

fs = np.linspace(2,17,100)

plt.clf()
plt.plot(fs, lnflux_dens_fit(fs, approx_slope, approx_interp))
plt.plot(flux_hist[1][:-1], ln_counts)
plt.xlabel("Flux in G band (mag)")
plt.ylabel("Density, log N per {:.4} mag".format(bin_width))
plt.savefig("temp_plots/gaia_flux_histogram.pdf")

def dens_functional_form(flux, slope, interp, bin_width):
    """For a given flux, gives the PDF (number of stars per units
    magnitude)"""
    return np.exp(lnflux_dens_fit(flux, slope, interp)) / bin_width


def sum_stars_from_fit(min_f, max_f, nrecs, bin_width, fit_slope,
                       fit_interp):
    rec_width = (max_f - min_f) / nrecs

    f = min_f
    total_area = 0
    while f < max_f:
        total_area += rec_width *\
                      dens_functional_form(f, fit_slope, fit_interp,
                                           bin_width)
        f += rec_width
    return total_area


max_bp_mag = 16.32 # 2MASS J03350208+2342356

max_mag_ix = np.digitize(max_bp_mag, flux_hist[1])
gaia_star_count = np.sum(flux_hist[0][:max_mag_ix])

approx_cut_off = 12
cut_off_ix = np.digitize(approx_cut_off, flux_hist[1])
exact_cut_off = flux_hist[1][cut_off_ix]
print(flux_hist[1][cut_off_ix])

faint_star_count = sum_stars_from_fit(min_f=exact_cut_off, max_f=max_bp_mag,
                                      nrecs=1000, bin_width=bin_width,
                                      fit_slope=approx_slope,
                                      fit_interp=approx_interp)


corrected_star_count = np.sum(flux_hist[0][:cut_off_ix])\
                       + faint_star_count

correction_factor = corrected_star_count / gaia_star_count

print("Correction factor: {}".format(correction_factor))

