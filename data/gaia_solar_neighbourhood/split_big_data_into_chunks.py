"""
Split this big data table into smaller chunks for computation of bg ols.
It is safer to do it in chunks in case something breaks while the process is running.
"""

import numpy as np
from astropy.table import Table

data = Table.read('ScoCen_box_result.fits') # 1.5M stars between 80 and 200 pc.
# TODO: add 0-80 pc sample! and beyond 200pc!

# How many chunks?
N=10

indices_chunks = np.array_split(range(len(data)), N)

for i, ind in enumerate(indices_chunks):
    subdata = data[ind]
    subdata.write('/priv/mulga1/marusa/chronostar/data/ScoCen_box_result_chunk%d.fits'%(i+1))
    print 'ScoCen_box_result_chunk%d.fits'%(i+1), 'written', len(subdata)