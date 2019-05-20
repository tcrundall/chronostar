import numpy as np
from astropy.table import Table

data = Table.read('../results/marusa_testing_original/same_centroid_synth_measurements.fits')

print data
print data.colnames