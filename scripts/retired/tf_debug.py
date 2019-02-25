import numpy as np
import sys
sys.path.insert(0, '..')

from chronostar.retired.tracingback import trace_forward
import chronostar.transform as tf

cov = np.eye(6,6)
cov[:3] *= 4.
cov[3:6] *= 25.

mean = [-80, 80, 50, 10, -20, -5]

age = 10.

new_cov = tf.transformCovMat(cov, trace_forward, mean, dim=6, args=(age,))
