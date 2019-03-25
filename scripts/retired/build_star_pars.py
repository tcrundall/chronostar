from __future__ import print_function, division

"""
Takes a table with LSR cartesain columns (i.e. `banyan_parser.py`)
and binaries mass-weighted (i.e. `banyan_binary_combiner.py`) and
converts into dictionary format to be used by chronostar groupfitter/expectmax
"""

if __name__=='__main__':
    bt_filename = '../data/gagne_bonafide_full_kinematics_with_lit_and' \
                  'best_radial_velocity_comb_binars.fits'


