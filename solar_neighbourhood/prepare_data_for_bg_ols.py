"""
Author: Marusa Zerjal, 2019 - 08 - 02

Convert big data to cartesian system.

"""

import sys
sys.path.insert(0, '..')
from chronostar import tabletool

datafile = '../data/ScoCen_box_result.fits'

data_table = tabletool.read(datafile)

print('DATA READ', len(data_table))

# Convert to Cartesian
print('Convert to cartesian')
# Performs conversion in place (in memory) on `data_table`
tabletool.convert_table_astro2cart(table=data_table, return_table=True)

data_table.write('/priv/mulga1/marusa/chronostar/data/ScoCen_box_result_cartesian.fits')
print('Cartesian written.', len(data_table), len_original)
