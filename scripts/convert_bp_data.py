import numpy as np
import logging
import pickle
import sys

sys.path.insert(0, '..')

import chronostar.retired.groupfitter as rgf
import chronostar.measurer as ms
import chronostar.converter as cv

original_tb_file = "../data/bp_TGAS2_traceback_save.pkl"
data_file = "../data/betaPic.csv"
astro_file = "../data/bp_astro.dat"
xyzuvw_file = "../data/bp_xyzuvw.fits"

with open(original_tb_file, 'r') as fp:
    table, _, _, _ = pickle.load(fp)

with open (data_file, 'r') as fp:
    header = fp.readline().split('\r')[0].split(',')

data_str = np.loadtxt(data_file, skiprows=1, delimiter=',',
                      dtype='S20')

data_astro = data_str[:,7:].astype('float64')
mini_hdr = ['RV','e_RV','RAdeg','DEdeg','Plx','e_Plx','pmRA','e_pmRA','pmDE','e_pmDE']

data_ordered = np.zeros((0,len(mini_hdr)))

data_ordered = data_str[:,0]
data_ordered = np.vstack((data_ordered, data_astro[:,2]))
data_ordered = np.vstack((data_ordered, data_astro[:,3]))
data_ordered = np.vstack((data_ordered, data_astro[:,4]))
data_ordered = np.vstack((data_ordered, data_astro[:,5]))
data_ordered = np.vstack((data_ordered, data_astro[:,6]))
data_ordered = np.vstack((data_ordered, data_astro[:,7]))
data_ordered = np.vstack((data_ordered, data_astro[:,8]))
data_ordered = np.vstack((data_ordered, data_astro[:,9]))
data_ordered = np.vstack((data_ordered, data_astro[:,0]))
data_ordered = np.vstack((data_ordered, data_astro[:,1]))
data_ordered = data_ordered.T

t = ms.convertAstroArrayToTable(data_ordered)
try:
    t.write(astro_file, format='ascii', overwrite=False)
except IOError:
    print("{} already exists...".format(astro_file))

xyzuvw_dict =\
    cv.convertMeasurementsToCartesian(loadfile=astro_file,
                                      savefile=xyzuvw_file)