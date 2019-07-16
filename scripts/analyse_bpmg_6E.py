"""
Temporary script which aids in the loading and
analysis of BPMG 6E fit
"""
import numpy as np

from astropy.table import Table

import sys
sys.path.insert(0, '..')
from chronostar import tabletool
from chronostar.component import SphereComponent

master_table_file = '../data/paper1/beta_Pictoris_corrected_everything.fits'

bpmg_rdir = '../results/beta_Pictoris_with_gaia_small_inv2/'

fz = np.load(bpmg_rdir + 'final_membership.npy')
fcomps = SphereComponent.load_raw_components(bpmg_rdir + 'final_comps.npy')
fmers = np.load(bpmg_rdir + 'final_med_errs.npy')

master_table = Table.read(master_table_file)
star_pars, table_ixs = tabletool.build_data_dict_from_table(
        master_table,
        return_table_ixs=True
)

old_table_file = '../data/paper1/beta_Pictoris_with_gaia_small_everything_final.fits'
old_table = Table.read(old_table_file)
