#! /usr/bin/env python

import chronostar.traceback as tb
import numpy as np
import pickle
import pdb

group_names = [
   'bPic', 'TW Hya', 'Tuc Hor', 'Columba','Carina', 'Argus', 'AB Dor']

# taken from http://www.astro.umontreal.ca/~malo/banyan.php
xyzuvw_group_all = np.array([
    [9.27,-5.96,-13.59,-10.94,-16.25,-9.27],
    [12.49, -42.28, 21.55, -9.95, -17.91, -4.65],
    [11.39, -21.21, -35.40, -9.88, -20.70, -0.90],
    [-27.44, -31.32, -27.97, -12.24, -21.32, -5.58],
    [15.55, -58.53, -22.95, -10.34, -22.31, -5.91],
    [14.80, -24.67, -6.72, -21.78, -12.08, -4.52],
    [-2.37, 1.48, -15.62, -7.12, -27.31, -13.81]
    ])

# ages simply being the middle of range provided in malo
ages_all = np.array([17, 14, 35, 35, 35, 40, 80])

assert(np.size(ages_all) == np.shape(xyzuvw_group_all)[0])
assert(len(group_names) == np.shape(xyzuvw_group_all)[0])

initial_groups = np.zeros(np.shape(xyzuvw_group_all))

for i in range(np.shape(xyzuvw_group_all)[0]):
    initial_groups[i] = tb.traceback_group(
        xyzuvw_group_all[i], ages_all[i] )

save_file = "/short/kc5/init_mgs.pkl"
pickle.dump((group_names, initial_groups, ages_all), 
            open(save_file, 'w'))
