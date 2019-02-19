# coding: utf-8
get_ipython().magic(u'run ../scripts/ipython_primer.py')
get_ipython().magic(u'run analyse_bg_dens.py')
all_bpmg_dens
plt.yscale('log')
plt.savefig("log-dens-vs-bincount.pdf")
get_ipython().system(u'open log-dens-vs-bincount.pdf')
all_bpmg_dens
all_bpmg_dens[:20]
all_bpmg_dens[:19]
np.mean(all_bpmg_dens[9:17])
all_bpmg_dens[9:17]
all_bpmg_dens[9:18]
all_bpmg_dens[9:17]
np.mean(all_bpmg_dens[9:17])
offset
offset * 2 / 15.
offset * 2 / 10.
offset * 2
offset * 2 / 17.
get_ipython().magic(u'ls ')
get_ipython().magic(u'run analyse_bg_dens.py')
get_ipython().magic(u'ls ')
open sc-log-dens-vs-bincount.pdf
`open sc-log-dens-vs-bincount.pdf`
get_ipython().system(u'`open sc-log-dens-vs-bincount.pdf`')
all_bpmg_dens
all_bpmg_dens[:14]
all_bpmg_dens[:15]
all_bpmg_dens[:14]
all_bpmg_dens[10:14]
all_bpmg_dens[8:14]
np.mean(all_bpmg_dens[8:14])
near_g6d_hist[0].shape
my_g6d_hist = np.histogramdd(near_gaia, bins=15)
my_g6d_hist[0].shape
my_g6d_hist[0][7,7,7,7,7,7]
np.max(near_gaia, axis=0)
np.min(near_gaia, axis=0)
ref_mean
my_g6d_hist = np.histogramdd(near_gaia, bins=13)
get_ipython().magic(u'ls -rtal')
my_g6d_hist = np.histogramdd(near_gaia, bins=13)
my_g6d_hist[0][6,6,6,6,6,6]
my_g6d_hist[1][:,0]
my_g6d_hist[1][0]
my_g6d_hist[1]
my_g6d_hist[1].shape
my_widths = [my_g6d_hist[1][i][1] - my_g6d_hist[1][i][0] for i in range(6)]
my_widths
my_g6d_hist[0][6,6,6,6,6,6]
my_g6d_hist[0][6,6,6,6,6,7]
my_g6d_hist[0][6,6,6,6,6,6]
my_g6d_hist[0][6,6,7,6,6,6]
my_g6d_hist[0][6,6,6,6,6,6]
np.sum(my_g6d_hist[0])
my_g6d_hist[0][6,5,6,6,6,6]
my_g6d_hist[0][5,5,6,6,6,6]
my_g6d_hist[0][6,6,6,6,6,6]
my_g6d_hist[0][6,6,6,6,6,5]
my_g6d_hist[0][6,6,6,6,6,4]
my_g6d_hist[0][6,6,6,6,6,5]
np.std(gaia_xyzuvw, axis=0)
gaia_stds = np.std(gaia_xyzuvw, axis=0)
gaia_stds / 20.
np.where(gaia_xyzuvw < -10)
np.where(gaia_xyzuvw < -10)[0].shape
np.where(gaia_xyzuvw < -10)[1].shape
np.where(gaia_xyzuvw < np.zeros(6))[0].shape
np.where( (gaia_xyzuvw > np.zeros(6)) & (gaia_xyzuvw < np.ones(6)))[0].shape
len(np.where( (gaia_xyzuvw > np.zeros(6)) & (gaia_xyzuvw < np.ones(6)))[0])
dt.getDensity
dt.getDensity(ref_mean, gaia_xyzuvw)
dt.getDensity(ref_mean, gaia_xyzuvw)
dt.getDensity(ref_mean, gaia_xyzuvw)
ref_mean
ref_mean[:-1] -= 4.
ref_mean
ref_mean
dt.getDensity(ref_mean, gaia_xyzuvw)
data = gaia_xyzuvw
offset = 0.5 * np.std(data, axis=0) / 20.
mystars = np.where((data > point-offset) & (data < point+offset))
point = np.array(ref_mean)
mystars = np.where((data > point-offset) & (data < point+offset))
mystars[0]
data[mystars][:5]
data.shape
data[mystars].shape
mystars.shape
len(mystars)
mystars[0].shape
mystars[1].shape
mystars[1][:5]
mystars[0][:5]
mystars = np.where((data > point-offset) and (data < point+offset))
mystars = np.where((data > point-offset) && (data < point+offset))
myarr = np.arange(10)
myarr
np.where(myarr > 2 & myarr < 8)
np.where((myarr > 2) & (myarr < 8))
myarr[np.where((myarr > 2) & (myarr < 8))]
mystars = np.where((data > point-offset).all() and (data < point+offset).all())
mystars[0].shape
mystars
offset
offset = 0.5 * np.std(data, axis=0) / 20.
offset
offset = 0.5 * np.std(data, axis=0) / 20.
nstars = len(
        np.where((data > point-offset).all() & (data < point+offset).all())[0]
    )
nstars
offset = 0.5 * np.std(data, axis=0) / 15.
nstars = len(
        np.where((data > point-offset).all() & (data < point+offset).all())[0]
    )
nstars
offset
point - offset
point + offset
offset = 0.5 * np.std(data, axis=0) / 10.
nstars = len(
        np.where((data > point-offset).all() & (data < point+offset).all())[0]
    )
nstars
offset = 0.5 * np.std(data, axis=0)
nstars = len(
        np.where((data > point-offset).all() & (data < point+offset).all())[0]
    )
nstars
dt.getDensity(ref_mean, gaia_xyzuvw)
dt.getDensity(ref_mean, gaia_xyzuvw)
dt.getDensity(ref_mean, gaia_xyzuvw, 5.)
for i in range(3,15):
    print(getDensity(ref_mean, gaia_xyzuvw, i))
    
for i in range(3,15):
    print(dt.getDensity(ref_mean, gaia_xyzuvw, i))
    
bins_per_std = []
gen_dens = []
for i in range(3,20):
    bins_per_std.append(i)
    gen_dens.append(dt.getDensity(ref_mean, gaia_xyzuvw, i))
    
gen_dens
plt.plot(bins_per_std, gen_dens)
plt.show()
plt.clf()
plt.plot(bins_per_std, gen_dens)
plt.savefig("general_dens_at_bp_mean.pdf")
get_ipython().system(u'`open general_dens_at_bp_mean.pdf`')
plt.yscale('log')
plt.savefig("general_dens_at_bp_mean.pdf")
get_ipython().system(u'`open general_dens_at_bp_mean.pdf`')
w_step = []
x_step = []
bins_per_std = []
gen_dens 
gen_dens = []
gaia_std
gaia_std = np.std(gaia_xyzuvw, axis=0)
gaia_std.shape
gaia_std
for i in range(3,20):
    bins_per_std.append(i)
    w_step.append(gaia_std[-1]/float(i))
    x_step.append(gaia_std[0]/i)
    gen_dens.append(dt.getDensity(ref_mean, gaia_xyzuvw, i))
    
w_step
x_step
w_step
plt.clf()
plt.plot(w_step, gen_dens)
plt.xlabel("step size in w [km/s]")
plt.ylabel(r"Density ([pc km/s]$^{-3}$")
plt.title("General density at bpmg mean with differing step size")
plt.savefig("gen_dens_vs_w_step_size.pdf")
get_ipython().system(u'`open gen_dens_vs_w_step_size.pdf`')
plt.yscale('log')
plt.savefig("gen_dens_vs_w_step_size.pdf")
get_ipython().system(u'`open gen_dens_vs_w_step_size.pdf`')
plt.clf()
plt.plot(x_step, gen_dens)
plt.xlabel("step size in x [pc]")
plt.ylabel(r"Density ([pc km/s]$^{-3}$")
plt.yscale('log')
plt.title("General density at bpmg mean with differing step size")
plt.savefig("gen_dens_vs_x_step_size.pdf")
get_ipython().system(u'`open gen_dens_vs_x_step_size.pdf`')
gen_dens = []
ref_mean
plt.title("General density at twin bpmg mean with differing step size")
plt.savefig("gen_dens_vs_x_step_size.pdf")
x_step
np.argmin(abs(np.array(x_step) - 160))
x_step[np.argmin(abs(np.array(x_step) - 160))]
bins_per_std[5]
bins_per_std[7]
x_step[7]
gen_dens[7]
get_ipython().magic(u'pwd ')
