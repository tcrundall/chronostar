from __future__ import print_function, division

"""
Reads in table prepared by Marusa with all of BANYAN $\Sigma$ bona fide
members paired with Gaia astrometry
"""

from astropy.table import Table
import numpy as np

ddir = '../data/'
gagne_file = 'gagne_bonafide_full_kinematics_with_best_radial_velocity.fits'
galah_file = 'galah_banyan_bonafide_and_candidates_kinematics' \
             '_for_chronostar.fits'

gagnet = Table.read(ddir+gagne_file)
galaht = Table.read(ddir+galah_file)
