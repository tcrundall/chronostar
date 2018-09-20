"""
Provides many functions that aid plotting of stellar data sets and their fits
"""

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import synthesiser as syn
import errorellipse as ee
import traceorbit as torb
import transform as tf

COLORS = ['xkcd:neon purple','xkcd:orange', 'xkcd:cyan',
          'xkcd:sun yellow', 'xkcd:shit', 'xkcd:bright pink']
HATCHES = ['|', '/',  '+', '.', '*'] * 10

def add_arrow(line, position=None, indices=None, direction='right', size=15, color=None):
    """
    Add an arrow along a plotted line.

    Parameters
    ----------
    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.

    -- credit to some forgotten contributor to stackoverflow --
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if indices is None:
        if position is None:
            position = xdata.mean()
        # find closest index
        indices = [np.argmin(np.absolute(xdata - position))]

    for start_ind in indices:
        if direction == 'right':
            end_ind = start_ind + 1
        else:
            end_ind = start_ind - 1

        line.axes.annotate('',
            xytext=(xdata[start_ind], ydata[start_ind]),
            xy=(xdata[end_ind], ydata[end_ind]),
            arrowprops=dict(arrowstyle="->", color=color),
            size=size
        )


def plotPane(dim1=0, dim2=1, ax=None, groups=[], star_pars=None,
             group_then=False, group_now=False, group_orbit=False,
             annotate=False):
    """
    Plots a single pane capturing kinematic info in any desired 2D plane

    Uses global constants COLORS and HATCHES to inform consistent colour
    scheme.
    Can use this to plot different panes of one whole figure

    Parameters
    ----------
    dim1: x-axis, can either be integer 0-5 (inclusive) or a letter form
          'xyzuvw' (either case)
    dim2: y-axis, same conditions as dim1
    ax:   the axes object on which to plot (defaults to pyplots currnet axes)
    groups: a list of (or just one) synthesiser.Group objects, corresponding
            to the fit of the origin(s)
    star_pars:  dict object with keys 'xyzuvw' ([nstars,6] array of current
                star means) and 'xyzuvw_cov' ([nstars,6,6] array of current
                star covariance matrices)
    group_then: (bool) plot the group's origin
    group_now:  (bool) plot the group's current day distribution
    group_orbit: (bool) plot the trajectory of the group's mean
    annotate: (bool) add text describing the figure's contents

    Returns
    -------
    (nothing returned)
    """
    labels = 'XYZUVW'
    units = 3 * ['pc'] + 3 * ['km/s']

    # Tidying up inputs
    if ax is None:
        ax = plt.gca()
    if type(dim1) is not int:
        dim1 = labels.index(dim1.upper())
    if type(dim2) is not int:
        dim2 = labels.index(dim2.upper())
    if type(groups) is not list:
        groups = [groups]


    if star_pars:
        mns = star_pars['xyzuvw']
        covs = star_pars['xyzuvw_cov']
        ax.plot(mns[:,dim1], mns[:,dim2], '.', color='xkcd:red')
        for star_mn, star_cov in zip(mns, covs):
            ee.plotCovEllipse(star_cov[np.ix_([dim1, dim2], [dim1, dim2])],
                              star_mn[np.ix_([dim1, dim2])],
                              ax=ax, alpha=0.1, linewidth='0.1',
                              color='xkcd:red',
                              )

    for i, group in enumerate(groups):
        # TODO: get colors behaving neatly
        assert isinstance(group, syn.Group)
        cov_then = group.generateSphericalCovMatrix()
        mean_then = group.mean

        if group_then:
            ax.plot(mean_then[dim1], mean_then[dim2], marker='x',
                    color=COLORS[i])
            ee.plotCovEllipse(cov_then[np.ix_([dim1,dim2], [dim1,dim2])],
                              mean_then[np.ix_([dim1,dim2])],
                              with_line=True,
                              ax=ax, alpha=0.3, ls='--',
                              color=COLORS[i])
            if annotate:
                ax.annotate(r'$\mathbf{\mu}_0, \mathbf{\Sigma}_0$',
                            (mean_then[dim1],
                             mean_then[dim2]),
                             color=COLORS[i])

        if group_now:
            mean_now = torb.traceOrbitXYZUVW(mean_then, group.age,
                                             single_age=True)
            cov_now = tf.transform_cov(cov_then, torb.traceOrbitXYZUVW,
                                       mean_then, args=[group.age])
            ax.plot(mean_now[dim1], mean_now[dim2], marker='x',
                   color=COLORS[i])
            ee.plotCovEllipse(cov_now[np.ix_([dim1,dim2], [dim1,dim2])],
                              mean_now[np.ix_([dim1,dim2])],
                              with_line=True,
                              ax=ax, alpha=0.3, ls='-.',
                              ec=COLORS[i], fill=False, hatch=HATCHES[i],
                              color=COLORS[i])
            if annotate:
                ax.annotate(r'$\mathbf{\mu}_c, \mathbf{\Sigma}_c$',
                            (mean_now[dim1],mean_now[dim2]),
                            color=COLORS[i])

        if group_orbit:
            ntimes = 50
            orb_alpha = 0.1
            gorb = torb.traceOrbitXYZUVW(mean_then,
                                       times=np.linspace(0,group.age,ntimes),
                                       single_age=False)
            line_obj = ax.plot(gorb[:,dim1], gorb[:,dim2], ls='-',
                               alpha=orb_alpha,
                               color=COLORS[i])
            indices = [int(ntimes/3), int(2*ntimes/3)]
            add_arrow(line_obj[0], indices=indices,
                      color=COLORS[i])
            if annotate:
                ax.annotate("Orbital trajectory",
                            (gorb[int(ntimes/2), dim1],
                             gorb[int(ntimes/2), dim2]),
                             color=COLORS[i])


        ax.set_xlabel("{} [{}]".format(labels[dim1], units[dim1]))
        ax.set_ylabel("{} [{}]".format(labels[dim2], units[dim2]))


