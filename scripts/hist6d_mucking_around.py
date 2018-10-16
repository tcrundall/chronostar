# coding: utf-8
get_ipython().magic(u'run ipython_primer.py')
import chronostar.synthesiser as syn
import chronostar.datatool as dt
fgs = dt.loadGroups('../results/em_fit/synth_bpmg/final_groups.npy')
fgs
fgs[0]
fgs[0].pars
fgs[0].dx
fgs[0].dv
fgs[0].mean
fgs[0].getInternalSphericalPars()
for g in fgs:
    print(g.getInternalSphericalPars())
    
get_ipython().magic(u'run perform_incremental_em_bg_synth_fit.py')
get_ipython().magic(u'run perform_incremental_em_bg_synth_fit.py')
get_ipython().magic(u'run perform_incremental_em_bg_synth_fit.py')
get_ipython().magic(u'run extract_nearby_gaia.py')
get_ipython().magic(u'run extract_nearby_gaia.py')
lnlike_star
mystar
n_nearby
hists[0]
hists[0][0]
get_ipython().magic(u'run extract_nearby_gaia.py')
man_mask[0].shape
man_nearby_gaia.shape
hist6d = np.histogramdd(man_nearby_gaia)
hist6d.shape
len(hist6d)
hist6d[0].shape
hist6d[0]
hist6d[-].shape
hist6d[0].shape
hist6d[0,0].shape
hist6d[0][0]
hist6d[0][0].shape
hist6d[1].shape
hist6d[1]
np.digitize(hist6d[1], man_nearby_gaia[0])
hist6d[0].shape
hist6d[0].sum(axis=0)
hist6d[0].sum(axis=0).shape
hist6d[0][5,5,5,5,5]
hist6d[0][5,5,5,5]
hist6d[0].sum()
import tensorflow as tf
import tensorflow as tf
import tensorflow as tf
import tensorflow as tf
import tensorflow as tf
get_ipython().magic(u'run extract_nearby_gaia.py')
hist6d
np.sum(hist6d)
np.sum(hist6d[0])
hist6d[0].shape
3**6
manual_span = manual_ub - manual_lb
manual_span
np.sum(man_mask)
man_mask.shape
man_mask[0].shape
man_mask[1].shape
man_mask[0]
len(man_mask)
len(man_mask[0])
manual_span
np.prod(manual_span)
len(man_mask[0]) /np.prod(manual_span)
span
upper_boundary - lower_boundary
np.prod(upper_boundary - lower_boundary)
len(mask) / np.prod(upper_boundary - lower_boundary)
all_hist6d = np.histogramdd(gaia_xyzuvw, bins=10)
all_hist6d[1][0]
all_hist6d[1][1]
all_hist6d[1][2]
all_hist6d[1][3]
all_hist6d[1][4]
np.sum(all_hist6d[0])
10**6
np.median(all_hist6d[0])
np.mean(all_hist6d[0])
np.max(all_hist6d[0])
np.argmax(all_hist6d[0])
np.unravel(np.argmax(all_hist6d[0]), all_hist6d[0].shape)
np.unravel_index(np.argmax(all_hist6d[0]), all_hist6d[0].shape)
ixs = np.unravel_index(np.argmax(all_hist6d[0]), all_hist6d[0].shape)
al_hist6d[1][ixs]
all_hist6d[1][ixs]
all_hist6d[1].shape
len(all_hist6d[1])
all_hist6d[1][0]
len(all_hist6d[0] > 0.)
len(all_hist6d[0] < 0.)
(all_hist6d[0] > 0.).shape
np.sum(all_hist6d[0] > 0.)
np.sum(all_hist6d[0] < 0.)
np.sum(all_hist6d[0] == 0.)
np.argmax(all_hist6d[0])
np.max(all_hist6d[0])
bin_widths = [bins[1] - bins[0] for bins in all_hist6d[1]]
bin_widths
np.prod(bin_widths)
np.max(all_hist6d[0]) / np.prod(bin_widths)
bpmean = np.array([0,0,0,0,-4,-2])
all_hist6d[1]
myvals = [10,20,30,20,10]
mybins = np.array([0.,1.,2.,3.,4.,5.])
np.digitize(0.5,mybins)
myvals[np.digitize(0.5,mybins)]
myvals[np.digitize(4.5,mybins)]
myvals[np.digitize(4.5,mybins) -1] 
mydata 
mydata = np.random.randn(100)
mydata.shape
vals, bins = np.hist(mydata)
vals, bins = np.histogram(mydata)
vals
bins
mydata[0]
np.digitize(mydata[0], bins)
vals[np.digitize(mydata[0], bins)]
all_hist6d[1]
bpmean
all_hist6d[0][0]
all_hist6d[1][0]
bpmean[0]
np.digitize(bpmean[0], all_hist6d[1][0])
bp_ix = [np.digitize(bpmean[dim], all_hist6d[1][dim]) - 1 for dim in range(6)]
bp_ix
all_hist6d[0][bp_ix]
bp_ix
myarr = np.arange(12).reshape(-1,2,3)
myarr
myix = [0,0,0]
myarr[myix]
myarr[(myix)]
myarr[np.array(myix)]
myarr[np.unravel_index(np.argmax(myarr), myarr.shape)]
np.unravel_index(np.argmax(myarr), myarr.shape)
myix = tuple([0,0,0])
myix
myarr(myix)
myarr[myix]
all_hist6d[tuple(bp_ix)]
all_hist6d[0][tuple(bp_ix)]
allarea = np.prod(upper_boundary - lower_boundary)
all_bin_area = np.prod([bins[1] - bins[0] for bins in all_hist6d[1]])
all_bin_area
all_hist6d[0][tuple(bp_ix)] / all_bin_area
bpmg_dens = all_hist6d[0][tuple(bp_ix)] / all_bin_area
bpmg_dens
get_ipython().magic(u'run extract_nearby_gaia.py')
get_ipython().magic(u'run extract_nearby_gaia.py')
get_ipython().magic(u'run extract_nearby_gaia.py')
get_ipython().magic(u'run extract_nearby_gaia.py')
nbins
all_hist6d[0].shape
bin_widths =  [bins[1] - bins[0] for bins in all_hist6d[1]]
bin_widths
nearby_gaia[0].shape
nearby_gaia.shape
get_ipython().magic(u'run extract_nearby_gaia.py')
get_ipython().magic(u'run extract_nearby_gaia.py')
get_ipython().magic(u'run extract_nearby_gaia.py')
bp_ix
near_hist6d[0].shape
nbins
bin_area
bin_widths
near_hist6d[0][bp_ix]
bp_mean
bpmean
bin_widths
np.max(nearby_gaia, axis=0)
np.min(nearby_gaia, axis=0)
np.hist(nearby_gaia[:5])
np.histogram(nearby_gaia[:5])
np.histogram(nearby_gaia[:5].T)
nearby_gaia[:5]
np.histogram(nearby_gaia[:5])
import chronostar.datatool
dt.calcHistogramDensity(bpmean, near_hist6d[0], near_hist6d[1])
near_hist6d[1].shape
near_hist6d[0].shape
dt.calcHistogramDensity(bpmean, near_hist6d[0], near_hist6d[1])
get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
dt.calcHistogramDensity(bpmean, near_hist6d[0], near_hist6d[1])
import chronostar.datatool as dt
dt.calcHistogramDensity(bpmean, near_hist6d[0], near_hist6d[1])
dt.calcHistogramDensity(bpmean, near_hist6d[0], near_hist6d[1])
myarr
myarr[:,np.newaxis]
myarr1 = np.arange(12)
myarr1
myarr1 = np.arange(4)
myarr2 = np.arange(12).reshape(3,4)
myarr1
myarr2
myarr1 * myarr2
myarr1 * myarr2.T
myarr1[:, np.newaxis] * myarr2.T
myarr1[:,np.newaxis]
myarr
myarr.shape
myarr[0]
myarr[1]
myarr[2]
myarr[0]
myarr[0,1]
myarr[0,1,1]
np.einsum('ijk->j', myarr)
myarr
np.einsum('ij->j', myarr)
myarr2
np.einsum('ij->j', myarr2)
np.einsum('ijk->j', myarr)
myarr
myarr.shape
myarr[0]
myarr[1]
myarr[:,0]
myarr[:,1]
np.sum(myarr[:,1])
np.sum(myarr[:,0])
ord('A')
ord('i')
[chr(105+i) for i in range(6)]
[chr(105+i) for i in range(6)]
str([chr(105+i) for i in range(6)])
[chr(105+i) for i in range(6)]
[chr(105+i) for i in range(6)].join()
str.join([chr(105+i) for i in range(6)])
''.join([chr(105+i) for i in range(6)])
myarr2
dt.collapseHistogram(near_hist6d,0)
dt.collapseHistogram(near_hist6d[0],0)
nbins
dt.collapseHistogram(near_hist6d[0],0).sum()
np.max(nearby_gaia)
np.max(nearby_gaia, axis=0)
np.min(nearby_gaia, axis=0)
get_ipython().magic(u'run extract_nearby_gaia.py')
get_ipython().magic(u'run extract_nearby_gaia.py')
dt
dt.integrateHistogram(near_hist6d[0], near_hist6d[0], near_hist6d[1], manual_lb, manual_ub)
manual_lb
manual_lb[0]
manual_ub[0]
npoints
npoints = 10
np.linspace(-100, 100, 10)
dt.integrateHistogram(near_hist6d[0], near_hist6d[0], near_hist6d[1], manual_lb, manual_ub)
dt.integrateHistogram(near_hist6d[0], near_hist6d[1], manual_lb, manual_ub)
dt.integrateHistogram(near_hist6d[0], near_hist6d[1], manual_lb, manual_ub, 0)
dt.integrateHistogram(near_hist6d[0], near_hist6d[1], manual_lb, manual_ub, 0)
manual_lb[0]
manual_ub[1]
x_step_size = 150./10.
x_step_size
x_vals, x_heights = dt.integrateHistogram(near_hist6d[0], near_hist6d[1], manual_lb, manual_ub, 0)
x_vals
x_heights
np.sum(x_heights) * (x_val[1] - x_val[0])
np.sum(x_heights) * (x_vals[1] - x_vals[0])
(man_nearby_gaia.shape)
x_vals
x_vals[1] - x_vals[0]
200./11
200./9
x_vals, x_heights = dt.integrateHistogram(near_hist6d[0], near_hist6d[1], manual_lb, manual_ub, 0)
x_vals
x_heights
x_step_size
x_step_size = x_vals[1] - x_vals[0]
x_step_size
x_step_size * x_heights
x_heights
x_heights.sum() * x_step_size
np.sum(x_heights) * x_step_size
np.mgrid[-1:1:5j]
np.mgrid[-1:1:5]
np.mgrid[-1:1]
np.mgrid[-1:1:5j]
np.mgrid[0:10, 0:10, 0:10]
np.mgrid[0:10, 0:10, 0:10:5j]
np.mgrid[0:10, 0:10, 0:10:5j].shape
np.mgrid[0:10, 0:10, 0:10:5j][0]
np.mgrid[0:10, 0:10, 0:10:5j][1]
np.mgrid[0:10, 0:10, 0:10:5j][0]
np.mgrid[0:10, 0:10, 0:10:5j][2]
np.mgrid[0:10, 0:10, 0:10].shape
np.mgrid[0:10]
np.mgrid[0:10].shape
np.mgrid[0:5, 0:5].shape
np.mgrid[0:5, 0:5]
import itertools
itertools.combinations([0,1,2], [5,6,7])
itertools.combinations([[0,1,2], [5,6,7]], 2)
list(itertools.combinations([[0,1,2], [5,6,7]], 2))
np.arange(10)
list(itertools.combinations(np.arange(3), 2))
list(itertools.combinations(np.arange(3), 3))
list(itertools.combinations_with_replacement(np.arange(3), 2))
pts = zip(np.mgrid(np.arange(10), np.arange(10)))
pts = zip(np.mgrid[np.arange(10), np.arange(10]))
pts = zip(np.mgrid[np.arange(10), np.arange(10)])
x = np.arange(-5, 5, 1)
x
np.meshgrid(x,x)
np.meshgrid(x,x).shape
np.meshgrid(x,x)[0].shape
np.meshgrid(x,x)[1].shape
np.meshgrid(x,x)[2].shape
np.meshgrid(x,x)
np.meshgrid(x,x)[0]
xs, ys = np.meshgrid(x,x)
xs.shape
ys.shape
pts = zip(xs,ys)
pts
np.mgrid[-1:1:5j]
np.mgrid[-1.5:1:5j]
np.mgrid[-1.5:1.8:5j]
np.mgrid[-1.5:1.8:5j, -10.:-20:5j]
np.mgrid[-1.5:1.8:5j, -10.:-20:5j].shape
xs, ys, zs, us, vs, ws =    np.meshgrid(np.arange(lower_bound[0], upper_bound[0], npoints),
                np.arange(lower_bound[1], upper_bound[1], npoints),
                np.arange(lower_bound[2], upper_bound[2], npoints),
                np.arange(lower_bound[3], upper_bound[3], npoints),
                np.arange(lower_bound[4], upper_bound[4], npoints),
                np.arange(lower_bound[5], upper_bound[5], npoints),
                )
lower_bound = manual_lb
upper_bound = manual_up
upper_bound = manual_ub
npoints
xs, ys, zs, us, vs, ws =    np.meshgrid(np.arange(lower_bound[0], upper_bound[0], npoints),
                np.arange(lower_bound[1], upper_bound[1], npoints),
                np.arange(lower_bound[2], upper_bound[2], npoints),
                np.arange(lower_bound[3], upper_bound[3], npoints),
                np.arange(lower_bound[4], upper_bound[4], npoints),
                np.arange(lower_bound[5], upper_bound[5], npoints),
                )
xs
xs.shape
ys.shape
np.meshgrid(-10:10:5j)
np.mgrid(-10:10:5j)
np.mgrid[-1.5:1.8:5j]
np.mgrid[-10:10:5j]
np.meshgrid[lower_bound[0]:upper_bound[0]:npoints j,
            lower_bound[1]:upper_bound[1]:npoints j,
            lower_bound[2]:upper_bound[2]:npoints j,
            lower_bound[3]:upper_bound[3]:npoints j,
            lower_bound[4]:upper_bound[4]:npoints j,
            lower_bound[5]:upper_bound[5]:npoints j]
np.meshgrid[lower_bound[0]:upper_bound[0]:10j,
            lower_bound[1]:upper_bound[1]:10j,
            lower_bound[2]:upper_bound[2]:10j,
            lower_bound[3]:upper_bound[3]:10j,
            lower_bound[4]:upper_bound[4]:10j,
            lower_bound[5]:upper_bound[5]:10j]
np.mgrid[lower_bound[0]:upper_bound[0]:10j,
           lower_bound[1]:upper_bound[1]:10j,
           lower_bound[2]:upper_bound[2]:10j,
           lower_bound[3]:upper_bound[3]:10j,
           lower_bound[4]:upper_bound[4]:10j,
           lower_bound[5]:upper_bound[5]:10j]
xs, ys, zs, us, vs, ws = np.mgrid[lower_bound[0]:upper_bound[0]:10j,
            lower_bound[1]:upper_bound[1]:10j,
            lower_bound[2]:upper_bound[2]:10j,
            lower_bound[3]:upper_bound[3]:10j,
            lower_bound[4]:upper_bound[4]:10j,
            lower_bound[5]:upper_bound[5]:10j]
xs.shape
ys.shape
zs.shape
us.shape
xs[0]
hist6d[0].shape
man_hists[0].shape
man_hists[0]
man_hist[0][0].shape
man_hists[0][0].shape
near_hist6d = np.histogramdd(man_nearby_gaia, bins=10)
new_hs, new_es = dt.integrateHistogram2(near_hist6d[0], near_hist6d[1], manual_lb, manual_ub, 0)
np.linspace(0,10,10,endpoint=False)
new_hs, new_es = dt.integrateHistogram2(near_hist6d[0], near_hist6d[1], manual_lb, manual_ub, 0)
new_hs, new_es = dt.integrateHistogram2(near_hist6d[0], near_hist6d[1], manual_lb, manual_ub, 0)
new_hs.shape
new_hs[0]
new_hs[1]
new_hs[2]
new_es.shape
new_es
import matplotlib.pyplot as plt
plt.show(new_es[:-1], new_hs)
plt.plot(new_es[:-1], new_hs)
plt.show()
plt.clf()
plt.plot(new_es[:-1], new_hs)
plt.show()
plt.clf()
plt.plot(new_es[:-1], new_hs)
get_ipython().magic(u'pwd ')
plt.savefig("temp_plots/1dhist.pdf")
get_ipython().system(u'open temp_plots/1dhist.pdf')
manual_ub[0]
plt.box(new_es, new_hs)
plt.bar(new_es, new_hs)
plt.bar(new_es[:-1], new_hs)
plt.show()
plt.clf()
plt.bar(new_es[:-1], new_hs)
plt.savefig("temp_plots/1dhist.pdf")
plt.bar(new_es[:-1], new_hs31po4ih;[-0i -01423

)_

)
plt.clf()
plt.bar(new_es[:-1], new_hs, width=20, left='left')
plt.bar(new_es[:-1], new_hs, width=20, left=True)
plt.bar(new_es[:-1], new_hs, width=20)
plt.savefig("temp_plots/1dhist.pdf")
plt.clf()
plt.step(new_es[:-1], new_hs, width=20)
plt.step(new_es[:-1], new_hs)
plt.step(new_es[:-1], new_hs)
plt.savefig("temp_plots/1dhist.pdf")
new_hs, new_es = dt.integrateHistogram2(near_hist6d[0], near_hist6d[1], manual_lb, manual_ub, 0)
plt.clf()
plt.step(new_es, new_hs)
plt.savefig("temp_plots/1dhist.pdf")
get_ipython().system(u'open temp_plots/1dhist.pdf')
help(plt.step)
plt.clf()
help(plt.step)
plt.step(new_es, new_hs, where='post')
plt.savefig("temp_plots/1dhist.pdf")
help(plt.step)
plt.step(new_es, new_hs, where='mid')
plt.clf()
plt.step(new_es, new_hs, where='mid')
plt.savefig("temp_plots/1dhist.pdf")
plt.clf()
plt.step(new_es, new_hs, where='mid')
plt.xlim()
plt.xlem
plt.slim
plt.xlim
plt.ylim
plt.ylim()
plt.xlim()
plt.plot(0,0)
plt.savefig("temp_plots/1dhist.pdf")
get_ipython().system(u'open temp_plots/1dhist.pdf')
