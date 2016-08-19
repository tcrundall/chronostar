"""This is a test script for tracing back beta Pictoris stars"""

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from chronostar.error_ellipse import plot_cov_ellipse
import chronostar.traceback as traceback
plt.ion()

age = 150
times = np.linspace(0,age,(age+1))
xyzuvw = np.array([1,1,1,1,1,1])
x = traceback.traceforward(xyzuvw,age)
print(x)
        
