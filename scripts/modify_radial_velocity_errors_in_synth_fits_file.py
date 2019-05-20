import numpy as np
from astropy.table import Table

data = Table.read('../results/marusa_testing_original/same_centroid_synth_measurements2.fits')

print data
print data.colnames


data['radial_velocity_error'] = 1e+10 # infinite errors

data.write('../results/marusa_testing_original/same_centroid_synth_measurements2.fits')