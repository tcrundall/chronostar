from __future__ import print_function, division, unicode_literals

import numpy as np
import sys
sys.path.insert(0, '..')

from chronostar import tabletool


def get_region(assoc_name):

    gagne_reference_data =\
        '../data/gagne_bonafide_full_kinematics_with_lit_and_best_radial_velocity' \
        '_comb_binars_with_banyan_radec.fits'

    gagne_table = tabletool.read(gagne_reference_data)

    if assoc_name not in set(gagne_table['Moving group']):
        raise UserWarning,\
            'Association name must be one of:\n{}\nReceived: "{}"'.format(
                    list(set(gagne_table['Moving group'])), assoc_name
            )

    # Extract all stars
    subtable = gagne_table[np.where(gagne_table['Moving group'] == assoc_name)]

    star_means = tabletool.buildDataFromTable(subtable, only_means=True)

    data_upper_bound = np.nanmax(star_means, axis=0)
    data_lower_bound = np.nanmin(star_means, axis=0)

    data_span = data_upper_bound - data_lower_bound
    data_centre = 0.5 * (data_upper_bound + data_lower_bound)

    # Set up boundaries of box that span double the association
    box_lower_bound = data_centre - data_span
    box_upper_bound = data_centre + data_span
    return box_lower_bound, box_upper_bound

if __name__ == '__main__':
    assoc_name = sys.argv[1]
    box_lower_bound, box_upper_bound = get_region(assoc_name)
    print('box lower bound: {}\nbox upper bound: {}'.format(box_lower_bound,
                                                            box_upper_bound,))

