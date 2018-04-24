import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plotPointCov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).
    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plotCovEllipse(cov, pos, nstd, ax, **kwargs)


def plotCovEllipse(cov, pos, nstd=2, ax=None, with_line=False, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    if 'alpha' not in kwargs.keys():
        ellip.set_alpha(0.3)
    if 'color' not in kwargs.keys():# and 'c' not in kwargs.keys():
        ellip.set_facecolor('red')

    ax.add_patch(ellip)

    # brute forcing axes limits so they contain ellipse patch
    # maybe a cleaner way of doing this, but I couldn't work it out
    max_range = 0.5 * max(width,height)

    lx = pos[0] - max_range
    ux = pos[0] + max_range
    ly = pos[1] - max_range
    uy = pos[1] + max_range

    # THEN just fucking plot an invisible line across the ellipse.
    if with_line:
        ax.plot((lx, ux), (ly, uy), alpha=0.)

    return ellip