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
import matplotlib.pyplot as plt
import pdb
from matplotlib.patches import Ellipse
import sys

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

def plot_something(dims, infile):
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

    dim1=dims[0]
    dim2=dims[1]
    cov_ix1 = [[dim1,dim2],[dim1,dim2]]
    cov_ix2 = [[dim1,dim1],[dim2,dim2]]
    axis_titles=['X (pc)','Y (pc)','Z (pc)','U (km/s)','V (km/s)','W (km/s)']
    
    stars, times, xyzuvw, xyzuvw_cov = pickle.load(open(infile, 'r'))
    nstars = len(xyzuvw)

    for j in range(len(times)-1):
    #for j in range(5):
        plt.clf()
        for i in range(nstars):
            cov_end = xyzuvw_cov[i,j,cov_ix1,cov_ix2]
        #if (np.sqrt(cov_end.trace()) < max_plot_error):
            # if plot_text:
            #     if i in text_ix:
            #         plt.text(xyzuvw[i,0,dim1]*1.1
            #                 + xoffset[i],xyzuvw[i,0,dim2]*1.1 
            #                 + yoffset[i],star['Name'],fontsize=11)
            plt.plot(xyzuvw[i,:j+1,dim1],xyzuvw[i,:j+1,dim2],'b-')
            plot_cov_ellipse(
                    xyzuvw_cov[i,0,cov_ix1,cov_ix2],
                    [xyzuvw[i,0,dim1],xyzuvw[i,0,dim2]],color='g',alpha=1)
            plot_cov_ellipse(
                    cov_end, [xyzuvw[i,j,dim1],xyzuvw[i,j,dim2]],
                    color='r',alpha=0.2)
    
        plt.xlabel(axis_titles[dim1])
        plt.ylabel(axis_titles[dim2])
        plt.title(times[j])
        #plt.axis(axis_range)
        plt.ylim(-200,200)
        plt.xlim(-200,200)
        #plt.axes().set_aspect('equal', 'datalim')
        plt.savefig("temp_plots/{}plot{}{}.png".format(
            j, axis_titles[dim1][0], axis_titles[dim2][0]))

if __name__ == "__main__":
    filename = sys.argv[1]
    plot_something([0,1], filename)
    plot_something([0,2], filename)
    plot_something([1,2], filename)
