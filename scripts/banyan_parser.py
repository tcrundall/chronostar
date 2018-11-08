from __future__ import division, print_function

"""
TODO: Come up with a neater way to handle missing data. Maybe fits
will permit blanks to be included, should explore this
"""

import numpy as np
from astropy.io import fits

def clean_row(row,i):

    char_array = np.array(list(row))
    # manually extract and convert entries

    #insert entry dividers, then split by them
    div_ix = (
        np.array([6, 34, 48, 51, 54, 60, 64, 67, 72, 80, 86, 94, 100,
                  107, 112, 119, 125, 137, 141, 145, 156]),
    )

    char_array[div_ix] = ','
    new_csv_row = (''.join(char_array)).split(',')
    new_csv_row = np.array([entry.strip() for entry in new_csv_row])

    return new_csv_row

if __name__=='__main__':
    NHEADER_ROWS = 348

    file_name = '../data/banyan_data.txt'

    fp = open(file_name, 'r')
    data_raw = fp.readlines()

    data_cleaned = [clean_row(row, i) for (i, row) in enumerate(data_raw) if
                    i > NHEADER_ROWS]
