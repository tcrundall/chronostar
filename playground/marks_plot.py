"""
This is a little utility function that makes 2d PDF plots with
flanking histograms.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic_2d

def pdfplot(x, y, fignum=None, xlim=None, ylim=None,
            nxbin=50, nybin=50, thresh=None, threshfrac=0.99,
            log=True, zmax=None, zscaling='count',
            xhistlim=None, yhistlim=None, colorbar=True,
            nticks=None, xlabel=None, ylabel=None,
            scat_alpha=1.0, aspect='auto'):
    """
    Makes a 2D PDF plot with flanking histograms.

    Parameters:
       x : array, shape (N,)
          x coordinates of points
       y : array, shape (N,)
          y coordinates of points
       fignum : int
          figure number in which to create the plot
       xlim : arraylike, shape (2,)
          plotting limits in x direction
       ylim : arraylike, shape (2,)
          plotting limits in y direction
       nxbin : int or arraylike, shape (N,)
          number of bins in x direction
       nybin : int or arraylike, shape (N,)
          number of bins in y direction
       thresh : float
          threshhold below which to hide the density histogram and
          show individual points; default is to use threshfrac instead
       threshfrac : float
          minimum fraction of the points that should be within the
          density threshhold; points in lower density regions are shown
          individually in a scatter plot
       log : bool
          use linear or logarithmic scale for PDFs
       zmax : float
          maximum value for 2D density PDF plots; default if this is
          not set depends on zscaling
       zscaling : 'max' | 'count' | 'frac' | 'density' | 'normed'
          method used to scale the PDFs; 'max' means that all
          histograms / PDFs are normalised to have a maximum of unity,
          'count' means that histograms / PDFs show absolute number
          counts in each bin, 'frac' means that histograms / PDFs show
          the fraction of the points in each bin, 'density' means
          that histograms / PDFs show the density of points in each
          bin, and 'normed' means that histograms / PDFs show the
          probability density in each bin
       xhistlim : arraylike, shape (2,)
          limits on the histogram in the x direction
       yhistlim : arraylike, shape (2,)
          limits on the histogram in the x direction
       colorbar : bool
          include a color bar for the density 2D PDFs or not
       nticks : int
          number of tick marks on the color bar
       xlabel : string
          labels for x dimension
       ylabel : string
          labels for y dimension
       scat_alpha : float
          alpha value for scatter plot points
       truths : arraylike, shape (M,)
          true values of parameters, which should be highlighted

    Returns
       Nothing
    """

    # Define the plotting grid
    if xlim is None:
        xlim = [np.amin(x), np.amax(x)]
    if ylim is None:
        ylim = [np.amin(y), np.amax(y)]
    xgrd = np.linspace(xlim[0], xlim[1], nxbin+1)
    ygrd = np.linspace(ylim[0], ylim[1], nybin+1)
    xgrd_h = 0.5*(xgrd[1:]+xgrd[:-1])
    ygrd_h = 0.5*(ygrd[1:]+ygrd[:-1])
    xx, yy = np.meshgrid(xgrd_h, ygrd_h)

    # Get 2D histogram; note that we have to handle the case of
    # inverted axis limits with care, because binned_statistic_2d
    # doesn't natively support them
    xlim1 = np.sort(xlim)
    ylim1 = np.sort(ylim)
    count, xe, ye, binidx \
        = binned_statistic_2d(x, y,
                              np.ones(x.shape),
                              statistic='sum',
                              bins=[nxbin, nybin],
                              range = [[float(xlim1[0]), float(xlim1[1])],
                                       [float(ylim1[0]), float(ylim1[1])]],
                              expand_binnumbers=True)
    if xlim[0] > xlim[1]:
        count = count[::-1, :]
        xe = xe[::-1]
        binidx[0,:] = nxbin+1 - binidx[0,:]
    if ylim[0] > ylim[1]:
        count = count[:, ::-1]
        ye = ye[::-1]
        binidx[1,:] = nybin+1 - binidx[1,:]        

    # Set z
    if zscaling == 'max':
        z = count / np.amax(count)
    elif zscaling == 'count':
        z = count
    elif zscaling == 'frac':
        z = count / len(x)
    elif zscaling == 'density':
        z = count / np.abs((xe[1]-xe[0])*(ye[1]-ye[0]))
    elif zscaling == 'normed':
        z = count / np.abs(len(x)*(xe[1]-xe[0])*(ye[1]-ye[0]))

    # Set minima and maxima for 2D plot
    if zmax is None:
        if zscaling == 'max':
            zmax = 1.0
        else:
            zmax = np.amax(z)
    if thresh is not None:
        zmin = thresh
    else:
        zsort = np.sort(z, axis=None)
        csum = np.cumsum(zsort)
        csum = csum/csum[-1]
        zmin = zsort[np.argmax(csum > 1.0-threshfrac)]
    if log:
        zmin = np.log10(zmin)
        zmax = np.log10(zmax)

    # Take log if requested
    if log:
        if np.amax(z) == 0.0:
            raise ValueError("cannot use log scale: no positive z values")
        z[z == 0] = 1.0e-6*np.amin(z[z > 0])
        z[z == 0] = np.amin(z[z > 0])
        z = np.log10(z)

    # Get indices of individual points to show
    flag = np.logical_and.reduce((binidx[0,:] > 0,
                                  binidx[1,:] > 0,
                                  binidx[0,:] <= count.shape[0],
                                  binidx[1,:] <= count.shape[1]))
    scatteridx = np.zeros(len(x), dtype=bool)
    scatteridx[flag] \
        = z[binidx[0,flag]-1, binidx[1,flag]-1] < zmin
        
    # Set up plot
    fig = plt.figure(fignum)
    plt.clf()
    gs = gridspec.GridSpec(4, 4)
    axcen = plt.subplot(gs[1:, :-1])

    # Plot contour at threshhold
    axcen.contour(xx, yy, np.transpose(z), levels=[zmin],
                  colors='k',
                  linestyles='-')
    
    # Plot scatter points outside contour
    axcen.scatter(x[scatteridx],
                  y[scatteridx],
                  color='k', s=5, alpha=scat_alpha,
                  edgecolor='none')

    # Plot density map
    img = axcen.imshow(np.transpose(z),
                       origin='lower', aspect=aspect,
                       vmin=zmin, vmax=zmax, cmap=cm.afmhot_r,
                       extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
    
    # Set plot range
    axcen.set_xlim(xlim)
    axcen.set_ylim(ylim)

    # Add labels
    if xlabel is not None:
        axcen.set_xlabel(xlabel)
    if ylabel is not None:
        axcen.set_ylabel(ylabel)

    # Get 1D histograms
    histx, xe \
        = np.histogram(x, bins=nxbin, range=xlim1)
    histy, ye \
        = np.histogram(y, bins=nybin, range=ylim1)
    if xlim[0] > xlim[1]:
        histx = histx[::-1]
        xe = xe[::-1]
    if ylim[0] > ylim[1]:
        histy = histy[::-1]
        ye = ye[::-1]
    if zscaling == 'max':
        histx = histx / float(np.amax(histx))
        histy = histy / float(np.amax(histy))
    elif zscaling == 'count':
        pass
    elif zscaling == 'frac':
        histx = histx / float(len(x))
        histy = histy / float(len(y))
    elif zscaling == 'density':
        histx = histx / np.abs(xe[1]-xe[0])
        histy = histy / np.abs(ye[1]-ye[0])
    elif zscaling == 'normed':
        histx = histx / np.abs(len(x)*(xe[1]-xe[0]))
        histy = histy / np.abs(len(y)*(ye[1]-ye[0]))
    else:
        raise ValueError("bad value of zscaling")

    # Add flanking histograms
    if zscaling == 'max':
        label = 'Scaled PDF'
    elif zscaling == 'count':
        label = r'N'
    elif zscaling == 'frac':
        label = 'Fraction'
    elif zscaling == 'density':
        label = 'Density'
    elif zscaling == 'normed':
        label = 'PDF'
    axtop = plt.subplot(gs[0, :-1])
    axtop.bar(xgrd[:-1], histx, xgrd[1]-xgrd[0],
              align='edge',
              facecolor='C0', edgecolor='black')
    axtop.set_xlim(xlim)
    if xhistlim is not None:
        axtop.set_ylim(xhistlim)
    if log:
        axtop.set_yscale('log')
    axtop.set_xticklabels([])
    axtop.set_ylabel(label)
    axright = plt.subplot(gs[1:, -1])
    axright.barh(ygrd[:-1], histy, ygrd[1]-ygrd[0], 
                 align='edge',
                 facecolor='C0',
                 edgecolor='black')
    axright.set_ylim(ylim)
    if yhistlim is not None:
        axright.set_xlim(yhistlim)
    if log:
        axright.set_xscale('log')
    axright.set_yticklabels([])
    axright.set_xlabel(label)

    # Add colorbar
    if colorbar:
        if zscaling == 'max':
            if log:
                label = 'log Scaled PDF'
            else:
                label = 'Scaled PDF'
        elif zscaling == 'count':
            if log:
                label = 'log N'
            else:
                label = 'N'
        elif zscaling == 'frac':
            if log:
                label = 'log fraction'
            else:
                label = 'fraction'
        elif zscaling == 'density':
            if log:
                label = 'log density'
            else:
                label = 'density'
        elif zscaling == 'normed':
            if log:
                label = 'log PDF'
            else:
                label = 'PDF'
        axcbar = plt.subplot(gs[0,-1])
        cbar = plt.colorbar(img, ax=axcbar, orientation='vertical',
                            fraction=0.8,
                            aspect=10,
                            label=label)
        if nticks is not None:
            cbar.set_ticks(np.linspace(zmin, zmax, nticks))
        axcbar.remove()

    # Return handles to axes
    return (axcen, axtop, axright)
