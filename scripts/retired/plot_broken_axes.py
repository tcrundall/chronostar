from __future__ import print_function, division

"""
Broken axis example, where the y-axis will have a portion cut out.
"""
import matplotlib.pyplot as plt
import numpy as np

mu1 = 30
sig1 = 1
mu2 = 10
sig2 = 1
pts = np.hstack((np.random.randn(100)*sig1 + mu1, np.random.randn(20)*sig2 + mu2 ))

# # 30 points between [0, 0.2) originally made using np.random.rand(30)*.2
# pts = np.array([
#     0.015, 0.166, 0.133, 0.159, 0.041, 0.024, 0.195, 0.039, 0.161, 0.018,
#     0.143, 0.056, 0.125, 0.096, 0.094, 0.051, 0.043, 0.021, 0.138, 0.075,
#     0.109, 0.195, 0.050, 0.074, 0.079, 0.155, 0.020, 0.010, 0.061, 0.008])
#
# # Now let's make two outlier points which are far away from everything.
# pts[[3, 14]] += .8

# If we were to simply plot pts, we'd lose most of the interesting
# details due to the outliers. So let's 'break' or 'cut-out' the y-axis
# into two portions - use the top (ax) for the outliers, and the bottom
# (ax2) for the details of the majority of our data
f, (ax, ax2) = plt.subplots(1, 2, sharey=True)

bin_width = 1.
max_age = np.ceil(np.max(pts))+ 0.5*bin_width
min_age = np.floor(np.min(pts)) - 0.5*bin_width
nbins = int((max_age - min_age) / bin_width)

# plot the same data on both axes
hist, edges = np.histogram(pts, bins=nbins)
ax.bar(edges[:-1], hist, align='edge', hatch='/', edgecolor='r', color='none')
# BUG!?!! SETTING ALPHA MAKES FACECOLOR GO TO BLACK
# ax.bar(edges[:-1], hist, align='edge', alpha=0.2, hatch='/', edgecolor='r', color='none')
ax.bar(edges[:-1]+0.3, hist, align='edge', hatch='+', edgecolor='b', color='none')
ax2.bar(edges[:-1], hist, align='edge', hatch='/')
# ax.hist(pts, bins=nbins)
# ax2.hist(pts, bins=nbins)

# zoom-in / limit the view to different portions of the data
ax.set_xlim(5, 15)  # outliers only
ax2.set_xlim(25, 35)  # most of the data

# hide the spines between ax and ax2
ax.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
# ax.xaxis.tick_top()
ax.yaxis.tick_left()
ax.tick_params(labelright='off')  # don't put tick labels at the top
ax2.yaxis.tick_right()

# This looks pretty good, and was fairly painless, but you can get that
# cut-out diagonal lines look with just a bit more work. The important
# thing to know here is that in axes coordinates, which are always
# between 0-1, spine endpoints are at these locations (0,0), (0,1),
# (1,0), and (1,1).  Thus, we just need to put the diagonals in the
# appropriate corners of each of our axes, and so long as we use the
# right transform and disable clipping.

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
# ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (-d, +d), **kwargs)  # bottom-left diagonal
ax2.plot((- d, + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

# What's cool about this is that now if we vary the distance between
# ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# the diagonal lines will move accordingly, and stay right at the tips
# of the spines they are 'breaking'

plt.savefig('temp_plots/broken_axes.pdf')
