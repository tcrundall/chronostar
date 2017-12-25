#From:
#https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py
#Copyright (c) 2012 Free Software Foundation
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of
#this software and associated documentation files (the "Software"), to deal in
#the Software without restriction, including without limitation the rights to
#use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
#of the Software, and to permit persons to whom the Software is furnished to do
#so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import numpy as np
import pickle
import pdb

import chronostar.groupfitter as gf
from chronostar import utils
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_point_cov(points, nstd=2, ax=None, **kwargs):
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
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
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
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    
    ellip.set_alpha(0.3)
    ellip.set_facecolor("red")

    ax.add_artist(ellip)
    return ellip

def calc_spread(xandys):
    """Rough calculation of occupied volume
    """
    approx_cov = np.cov(xandys)
    eig_val1, eig_val2 = np.sqrt(np.linalg.eigvalsh(approx_cov))
    return (eig_val1 * eig_val2)**0.5

def plot_something(dims, infile, fit_bayes=True):
    """
    Plot something.

    Input
    ______
    dims : (int, int)
        index of the two space-velocity dimensions in which one wishes to plot

    infile : .pkl traceback file
        file with traceback info
    """

    max_plot_error = 10000

    dim1, dim2 = dims

    cov_ix1 = [[dim1,dim2],[dim1,dim2]]
    cov_ix2 = [[dim1,dim1],[dim2,dim2]]
    axis_titles=['X (pc)','Y (pc)','Z (pc)','U (km/s)','V (km/s)','W (km/s)']

    stars, times, xyzuvw, xyzuvw_cov = pickle.load(open(infile, 'r'))
    nstars = len(xyzuvw)

    # calculate naiive volume
    spreads = np.zeros(len(times))
    for i in range(len(times)):
        joined_data = np.vstack((xyzuvw[:,i,dim1], xyzuvw[:,i,dim2]))
        spreads[i] = calc_spread(joined_data)

    bayes_spread = np.zeros(len(times)-1)
    if fit_bayes:
        # calculate Bayes fit
        for i in range(len(times)-1):
            best_fit, chain = gf.fit_group(
                infile, fixed_age=times[i], plot_it=True
            )
            #bayes_cov = utils.generate_cov(best_fit)
            #bayes_spread[i] = utils.approx_spread(bayes_cov)
            bayes_spread[i] = utils.approx_spread_from_chain(chain)

    for j in range(len(times)-1):
    #for j in range(5):
        plt.clf()
        f, (ax1, ax2) = plt.subplots(1,2)
        f.set_size_inches(10,5)
        for i in range(nstars):
            cov_end = xyzuvw_cov[i,j,cov_ix1,cov_ix2]
        #if (np.sqrt(cov_end.trace()) < max_plot_error):
            # if plot_text:
            #     if i in text_ix:
            #         plt.text(xyzuvw[i,0,dim1]*1.1
            #                 + xoffset[i],xyzuvw[i,0,dim2]*1.1
            #                 + yoffset[i],star['Name'],fontsize=11)
            ax1.plot(xyzuvw[i,:j+1,dim1],xyzuvw[i,:j+1,dim2],'b-')
            plot_cov_ellipse(
                xyzuvw_cov[i,0,cov_ix1,cov_ix2],
                [xyzuvw[i,0,dim1],xyzuvw[i,0,dim2]],color='g',alpha=1,
                ax=ax1
                )
            plot_cov_ellipse(
                cov_end, [xyzuvw[i,j,dim1],xyzuvw[i,j,dim2]],
                color='r',alpha=0.2, ax=ax1)
        ax1.set(aspect='equal')
        ax1.set_xlabel(axis_titles[dim1])
        ax1.set_ylabel(axis_titles[dim2])
        #plt.axis(axis_range)
        POS_RANGE = 300
        ax1.set_ylim(-POS_RANGE, POS_RANGE)
        ax1.set_xlim( POS_RANGE,-POS_RANGE)
        #plt.axes().set_aspect('equal', 'datalim')

        #ax2.set(aspect='equal')
        ax2.set_xlim(times[0], times[-1])
        ax2.set_xlabel("Traceback Time [Myr]")
        ax2.set_ylabel("Spread in XY plane [pc]")

        if j < len(times) - 1:
            ax2.plot(times[0:j+1],spreads[0:j+1], label="Naive standard dev")
            ax2.set_ylim(
                bottom=0.0, top=max(np.max(spreads),np.max(bayes_spread))
            )

            if fit_bayes:
                ax2.plot(times[0:j+1], bayes_spread[0:j+1], label="Bayes fit")
            ax2.axvline(
                7.0, ax2.get_ylim()[0], ax2.get_ylim()[1], color='r',
                ls = '--'
            )

        ax2.legend(loc=1)

        f.suptitle("{:.2f} Myr".format(times[j]))
        f.tight_layout(pad=2.0)
        f.savefig("temp_plots/{}plot{}{}.png".format(
            j, axis_titles[dim1][0], axis_titles[dim2][0]))

if __name__ == '__main__':
    #-- Example usage -----------------------
    # Generate some random, correlated data
    points = np.random.multivariate_normal(
            mean=(1,1), cov=[[0.4, 0.3],[0.2, 0.4]], size=1000
            )
    # Plot the raw points...
    x, y = points.T
    plt.plot(x, y, 'ro')

    # Plot a transparent 3 standard deviation covariance ellipse
    plot_point_cov(points, nstd=3, alpha=0.3, color='red')

    plt.show()
    plt.savefig("error_ellipse.png")
