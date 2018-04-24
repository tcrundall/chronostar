
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '..')

import chronostar.errorellipse as ee

points = np.random.multivariate_normal(
    mean=(1, 1), cov=[[0.4, -0.3], [0.2, 0.4]], size=1000
)
# Plot the raw points...
fig, ax = plt.subplots(1, 1)
x, y = points.T
el = ee.plot_point_cov(points, nstd=3)#, ax=ax)
plt.plot(x, y, 'ro')

# Plot a transparent 3 standard deviation covariance ellipse
fig.savefig("temp_plots/error_ellipse_default.png")

# Plot a transparent 3 standard deviation covariance ellipse
# Notice how the plot bounds include entire ellipse
plt.clf()
fig2, ax2 = plt.subplots(1, 1)
el = ee.plot_point_cov(points, nstd=3, ax=ax2, alpha=0.2, color='green')
fig2.savefig("temp_plots/error_ellipse_solo.png")


