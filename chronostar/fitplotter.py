"""
Provides many functions that aid plotting of stellar data sets and their fits
"""

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import synthesiser as syn
import errorellipse as ee
import traceorbit as torb
import transform as tf
import datatool as dt

COLORS = ['xkcd:neon purple','xkcd:orange', 'xkcd:cyan',
          'xkcd:sun yellow', 'xkcd:shit', 'xkcd:bright pink']
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
HATCHES = ['|', '/',  '+', '.', '*'] * 10

def add_arrow(line, position=None, indices=None, direction='right',
              size=15, color=None):
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
    https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot-with-matplotlib
    thomas - https://stackoverflow.com/users/5543796/thomas
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
        end_ind = start_ind + 1 if direction == 'right' else start_ind - 1

        line.axes.annotate('',
            xytext=(xdata[start_ind], ydata[start_ind]),
            xy=(xdata[end_ind], ydata[end_ind]),
            arrowprops=dict(arrowstyle="->", color=color),
            size=size
        )


def plotOrbit(pos_now, dim1, dim2, ax, end_age, ntimes=50, group_ix=None,
              with_arrow=False, annotate=False):
    """
    For traceback use negative age

    Parameters
    ----------
    pos_now: [6] array, known position of object
    dim1: integer, x-axis dimension
    dim2: integer, y-axis dimension
    ax: axes object, axes on which to plot line
    end_age: non-zero number, time to orbit till.
        Negative value --> traceback
        Positive value --> trace forward
    ntimes: integer {50], number of timesteps to calculate
    group_ix: index of group being plotted (for coloring reasons)
    with_arrow: (bool) {False}, whether to include arrows along orbit
    annotate: (bool) {False}, whether to include text
    """
    if group_ix is None:
        color = 'xkcd:red'
    else:
        color = COLORS[group_ix]



    # orb_alpha = 0.1
    gorb = torb.traceOrbitXYZUVW(pos_now,
                                 times=np.linspace(0, end_age, ntimes),
                                 single_age=False)
    line_obj = ax.plot(gorb[:, dim1], gorb[:, dim2], ls='-',
                       alpha=0.1,
                       color=color)
    indices = [int(ntimes / 3), int(2 * ntimes / 3)]
    if with_arrow:
        # make sure arrow is always pointing forwards through time
        direction = 'right' if end_age > 0 else 'left'
        add_arrow(line_obj[0], indices=indices, direction=direction,
                  color=color)
    if annotate:
        ax.annotate("Orbital trajectory",
                    (gorb[int(ntimes / 2), dim1],
                     gorb[int(ntimes / 2), dim2]),
                    color=color)


def plotPane(dim1=0, dim2=1, ax=None, groups=[], star_pars=None,
             star_orbits=False,
             group_then=False, group_now=False, group_orbit=False,
             annotate=False, membership=None, true_memb=None):
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
    star_orbits: (bool) plot the calculated stellar traceback orbits of
                        central estimate of measurements
    group_then: (bool) plot the group's origin
    group_now:  (bool) plot the group's current day distribution
    group_orbit: (bool) plot the trajectory of the group's mean
    annotate: (bool) add text describing the figure's contents

    Returns
    -------
    (nothing returned)
    TODO: Extend to handle membership probabilities
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
    if star_pars:
        nstars = star_pars['xyzuvw'].shape[0]
        pt_colors = np.array(nstars * ['xkcd:red'])
        markers = np.array(nstars * ['.'])
        if membership:
            best_mship = np.argmax(membership, axis=0)
            pt_colors = np.array(COLORS)[best_mship]
            # give correct and incorrect memberships a point and X respectively
            if true_memb:
                mrk_opts = np.array(['X', '.'])
                markers = mrk_opts[(best_mship == true_memb).astype(np.int)]

    # ensure groups is iterable
    try:
        len(groups)
    except:
        groups = [groups]
    # if type(groups) is not list: #???
    #     groups = [groups]

    # plot stellar data (positions with errors and optionally traceback
    # orbits back to some ill-defined age
    if star_pars:
        mns = star_pars['xyzuvw']
        covs = star_pars['xyzuvw_cov']
        ax.scatter(mns[:,dim1], mns[:,dim2], marker='.', color=pt_colors)
        for star_mn, star_cov, pt_color in zip(mns, covs, pt_colors):
            # plot uncertainties
            ee.plotCovEllipse(star_cov[np.ix_([dim1, dim2], [dim1, dim2])],
                              star_mn[np.ix_([dim1, dim2])],
                              ax=ax, alpha=0.1, linewidth='0.1',
                              color=pt_color,
                              )
            # plot traceback orbits for as long as oldest group (if known)
            # else, 30 Myr
            if star_orbits:
                try:
                    tb_limit = max([g.age for g in groups])
                except:
                    tb_limit = 30
                plotOrbit(star_mn, dim1, dim2, ax, end_age=-tb_limit)

    # plot info for each group (fitted, or true synthetic origin)
    for i, group in enumerate(groups):
        # try:
        #     assert isinstance(group, syn.Group) # for autocomplete when coding
        # except:
        #     import pdb; pdb.set_trace()

        cov_then = group.generateSphericalCovMatrix()
        mean_then = group.mean
        # plot group initial distribution
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

        # plot group current day distribution (should match well with stars)
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

        # plot orbit of mean of group
        if group_orbit:
            plotOrbit(mean_now, dim1, dim2, ax, -group.age, group_ix=i,
                      with_arrow=True, annotate=annotate)

        ax.set_xlabel("{} [{}]".format(labels[dim1], units[dim1]))
        ax.set_ylabel("{} [{}]".format(labels[dim2], units[dim2]))


def plotMultiPane(dim_pairs, star_pars, groups, save_file='dummy.pdf'):
    """
    Flexible function that plots many 2D slices through data and fits

    Takes as input a list of dimension pairs, stellar data and fitted
    groups, and will plot each dimension pair in a different pane.

    TODO: Maybe add functionality to control the data plotted in each pane

    Parameters
    ----------
    dim_pairs: a list of dimension pairs e.g.
        [(0,1), (3,4), (0,3), (1,4), (2,5)]
        ['xz', 'uv', 'zw']
        ['XY', 'UV', 'XU', 'YV', 'ZW']
    star_pars: either
        dicitonary of stellar data with keys 'xyzuvw' and 'xyzuvw_cov'
            or
        string filename to saved data
    groups: either
        a single synthesiser.Group object,
        a list or array of synthesiser.Group objects,
            or
        string filename to data saved as '.npy' file
    save_file: string, name (and path) of saved plot figure

    Returns
    -------
    (nothing)
    """

    # Tidying up inputs
    if type(star_pars) is str:
        star_pars = dt.loadXYZUVW(star_pars)
    if type(groups) is str:
        groups = np.load(groups)
        # handle case where groups is a single stored object
        if len(groups.shape) == 0:
            groups = groups.item()
    # ensure groups is iterable
    try:
        len(groups)
    except:  # groups is a single group instance
        groups = [groups]

    # setting up plot dimensions
    npanes = len(dim_pairs)
    rows = int(np.sqrt(npanes)) #plots are never taller than wide
    cols = (npanes + rows - 1) // rows  # get enough cols
    ax_h = 5
    ax_w = 5
    f, axs = plt.subplots(rows, cols)
    f.set_size_inches(ax_w * cols, ax_h * rows)

    # drawing each axes
    for i, (dim1, dim2) in enumerate(dim_pairs):
        plotPane(dim1, dim2, axs.flatten()[i], groups=groups,
                 star_pars=star_pars, star_orbits=False,
                 group_then=True, group_now=True, group_orbit=True,
                 annotate=False)

    f.savefig(save_file, bbox_inches='tight', format='pdf')


def sampleStellarPDFs(dim, star_pars):
    """
    Sample each provided star the build histogram from samples
    """
    all_samples = np.zeros(0)
    count = 100
    for mn, cov in zip(star_pars['xyzuvw'], star_pars['xyzuvw_cov']):
        samples = np.random.randn(count) * np.sqrt(cov[dim,dim]) + mn[dim]
        all_samples = np.append(all_samples, samples)
    return all_samples


def calcStellarPDFs(x, dim, star_pars):
    """
    For each point in `xs`, sum up the contributions from each star's PDF

    :param xs:
    :param star_pars:
    :return:
    """
    total = 0
    for mn, cov in zip(star_pars['xyzuvw'], star_pars['xyzuvw_cov']):
        total += dt.gauss(x, mn[dim], cov[dim,dim]**.5)
    return total


def evaluatePointInHist(x, hist_vals, hist_bins, normed=True):
    """
    Little utility function to evaluate the density of a histogram at point x

    NOTE! currently can't handle x > np.max(hist_bins)
    If x is below bin range, just extrapolate wings

    Parameters
    ----------
    x: a single value
    hist_vals: [nbin] number array; heights of each bin
    hist_bins: [nbin+1] float array; edges of bins
    """
    # if x < np.min(hist_bins) or x > np.max(hist_bins):
    #     return 0.

    bin_width = hist_bins[1] - hist_bins[0]
    if normed:
        hist_vals = hist_bins / float(np.sum(hist_vals))
    bin_height = hist_vals[np.digitize(x, hist_bins)]
    bin_density = bin_height / bin_width
    return bin_density


def plot1DProjection(dim, star_pars, groups, weights, ax=None, horizontal=False,
                     bg_hists=None):
    """
    Given an axes object, plot the 1D projection of stellar data and fits

    :param dim:
    :param star_pars:
    :param groups:
    :param z:
    :param vertical:
    bg_hists: [6, 2, ~nbins] list
        for each of the six dimensions, has two elements: the bin heights,
        and the bin edges
    :return:
    """
    if horizontal:
        orientation = 'horizontal'
    else:
        orientation = 'vertical'
    weights = np.array(weights).astype(np.float)
    if len(weights.shape) > 1:
        weights = weights.sum(axis=0)

    # Normalise weights to be unity
    norm_factor = weights.sum()
    weights /= norm_factor

    npoints = 1000
    if ax is None:
        ax = plt.gca()
    # ax.plot(xs, calcStellarPDFs(xs, dim, star_pars))
    vals, bins, _ = \
        ax.hist(sampleStellarPDFs(dim, star_pars), normed=True, histtype='step',
                orientation=orientation)

    xs = np.linspace(np.min(bins), np.max(bins), npoints)
    combined_gauss = np.zeros(xs.shape)
    for i, (group, weight) in enumerate(zip(groups, weights)):
        mean_now = torb.traceOrbitXYZUVW(group.mean, group.age, single_age=True)
        cov_now = tf.transform_cov(group.generateCovMatrix(), torb.traceOrbitXYZUVW,
                                   group.mean, args=[group.age])
        group_gauss = weight*dt.gauss(xs, mean_now[dim],
                                      np.sqrt(cov_now[dim,dim]))
        combined_gauss += group_gauss
        if bg_hists is not None:
            hist_contrib = weights[-1]*\
                           evaluatePointInHist(xs, bg_hists[dim][0],
                                               bg_hists[dim][1])
            combined_gauss += hist_contrib
        if horizontal:
            ax.plot(group_gauss, xs, color=COLORS[i])
        else:
            ax.plot(xs, group_gauss, color=COLORS[i])

    if horizontal:
        ax.plot(combined_gauss, xs, color='black', ls='--', alpha=0.4)
    else:
        ax.plot(xs, combined_gauss, color='black', ls='--', alpha=0.4)


def plotPaneWithHists(dim1, dim2, fignum=None, groups=[], weights=None,
                      star_pars=None,
                      star_orbits=False,
                      group_then=False, group_now=False, group_orbit=False,
                      annotate=False, bg_hists=None, membership=None,
                      true_memb=None):
    """
    Plot a 2D projection of data and fit along with flanking 1D projections.

    Uses global constants COLORS and HATCHES to inform consistent colour
    scheme.
    Can use this to plot different panes of one whole figure

    TODO: Incorporate Z
    TODO: incoporate background histogram

    Parameters
    ----------
    dim1: x-axis, can either be integer 0-5 (inclusive) or a letter form
          'xyzuvw' (either case)
    dim2: y-axis, same conditions as dim1
    fignum: figure number in which to create the plot
    groups: a list of (or just one) synthesiser.Group objects, corresponding
            to the fit of the origin(s)
    star_pars:  dict object with keys 'xyzuvw' ([nstars,6] array of current
                star means) and 'xyzuvw_cov' ([nstars,6,6] array of current
                star covariance matrices)
    star_orbits: (bool) plot the calculated stellar traceback orbits of
                        central estimate of measurements
    group_then: (bool) plot the group's origin
    group_now:  (bool) plot the group's current day distribution
    group_orbit: (bool) plot the trajectory of the group's mean
    annotate: (bool) add text describing the figure's contents

    Returns
    -------
    (nothing returned)
    """
    labels = 'XYZUVW'
    if weights is None and len(groups) > 0:
        weights = np.ones(len(groups)) / len(groups)
    if type(dim1) is not int:
        dim1 = labels.index(dim1.upper())
    if type(dim2) is not int:
        dim2 = labels.index(dim2.upper())
    if type(star_pars) is str:
        star_pars = dt.loadXYZUVW(star_pars)
    if type(groups) is str:
        groups = np.load(groups)
        if len(groups.shape) == 0:
            groups = np.array(groups.item())
    if type(bg_hists) is str:
        bg_hists = np.load(bg_hists)

    # Set up plot
    fig = plt.figure(fignum)
    plt.clf()
    gs = gridspec.GridSpec(4, 4)

    # Plot central pane
    axcen = plt.subplot(gs[1:, :-1])
    plotPane(dim1, dim2, ax=axcen, groups=groups, star_pars=star_pars,
             star_orbits=star_orbits, group_then=group_then,
             group_now=group_now, group_orbit=group_orbit, annotate=annotate,
             membership=membership, true_memb=true_memb)

    # Plot flanking 1D projections
    xlim = axcen.get_xlim()
    axtop = plt.subplot(gs[0, :-1])
    axtop.set_xlim(xlim)
    axtop.set_xticklabels([])
    plot1DProjection(dim1, star_pars, groups, weights, ax=axtop,
                     bg_hists=bg_hists)

    ylim = axcen.get_ylim()
    axright = plt.subplot(gs[1:, -1])
    axright.set_ylim(ylim)
    axright.set_yticklabels([])
    plot1DProjection(dim2, star_pars, groups, weights, ax=axright,
                     bg_hists=bg_hists, horizontal=True)



