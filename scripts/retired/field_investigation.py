# coding: utf-8
import chronostar.component

get_ipython().magic(u'cd field_blind/')
get_ipython().magic(u'run ../debug.py')
get_ipython().magic(u'ls final/')
final_groups = np.load("final_groups.npy")
get_ipython().magic(u'ls ')
star_pars.keys()
help(gf.get_lnoverlaps)
help(gf.lnprobFunc)
help(gf.lnprobFunc)
gf.lnprobFunc(
)
final_groups[0].age
final_groups[1].age
final_groups[0].getInternalSphericalPars()
gf.lnlike(final_groups[0].getInternalSphericalPars(), star_pars)
z = np.ones((nstars,1))
z.shape
gf.lnlike(final_groups[0].getInternalSphericalPars(), star_pars, z)
final_lnols = gf.lnlike(final_groups[0].getInternalSphericalPars(), star_pars, z, return_lnols=True)
final_lnols[:50]
final_lnols[40:60]
np.set_printoptions(suppress=True)
final_lnols[40:60]
np.max(final_lnols[50:])
final_groups[0].generateCov()
final_groups[0].generateCovarianceMat()
final_groups[0].generateCovMat()
final_groups[0].generateCovMatrix()
get_ipython().magic(u'ls ')
get_ipython().magic(u'ls final')
tf.transform_cov(final_groups[0].generateCovMatrix(), torb.traceOrbitXYZUVW, final_groups[0].mean, (final_groups[0].age,))
tf.transform_cov(final_groups[0].generateCovMatrix(), torb.traceOrbitXYZUVW, final_groups[0].mean, args=(final_groups[0].age,))
ass_cov_now = tf.transform_cov(final_groups[0].generateCovMatrix(), torb.traceOrbitXYZUVW, final_groups[0].mean, args=(final_groups[0].age,))
ass_cov_now
ass_mn_now = torb.traceOrbitXYZUVW(final_groups[0].mean, final_groups[0].age)
ass_mn_now
ass_cov_now
np.eye(6,6)
ass_cov_now[np.eye(6,6)]
ass_cov_now[np.where(np.eye(6,6) == 1.0)]
ass_cov_now[np.where(np.eye(6,6) != 1.0)]
red_ass_cov_now = np.copy(ass_cov_now)
red_ass_cov_now[np.where(np.eye(6,6) != 1.0)] = 0.0
red_ass_cov_now
dx_now = np.mean(ass_cov_now[np.where(np.eye(3,3) == 1.0)])
dx_now
dv_now = np.mean(ass_cov_now[np.where(3+np.eye(3,3) == 1.0)])
dv_now
np.eye(3,3)
dv_now = np.mean(ass_cov_now[3+np.where(np.eye(3,3) == 1.0)])
dv_now = np.mean((ass_cov_now[3,3], ass_cov_now[4,4], ass_cov_now[5,5])
)
dv_now
dx_now = np.mean(ass_cov_now[np.where(np.eye(3,3) == 1.0)]**0.5)
dx_now
dv_now = np.mean((ass_cov_now[3,3]**0.5, ass_cov_now[4,4]**0.5, ass_cov_now[5,5]**0.5))
dv_now
dx_now
approx_group_now_pars = np.hstack((ass_mn_now, dx_now, dv_now, 1e-5))
approx_group_now_pars
approx_group_now = syn.Group(approx_group_now_pars)
import chronostar.synthdata as syn
approx_group_now = chronostar.component.Component(approx_group_now_pars)
approx_group_now.generateCovMatrix()
ass_cov_now
gf.lnlike(approx_group_now.generateInternalSphericalPars(),
        star_pars, z)
gf.lnlike(approx_group_now.getInternalSphericalPars(),
        star_pars, z)
approx_lnols = gf.lnlike(approx_group_now.getInternalSphericalPars(),
        star_pars, z, return_lnols=True)
approx_lnols
approx_lnols.shape
np.min(approx_lnols[:50])
np.where(approx_lnols < np.min(approx_lnols[:50]))
np.mean(approx_lnols[:50])
np.where(approx_lnols < np.mean(approx_lnols[:50]))
np.where(approx_lnols < np.mean(approx_lnols[:50])).shape
np.where(approx_lnols < np.mean(approx_lnols[:50]))[0].shape
get_ipython().magic(u'save field_investigation.py 1-74')
red_ass_cov_now
ass_cov_now
same_lnols = gf.get_lnoverlaps(ass_cov_now, ass_mn_now, star_pars['xyzuvw_cov'], star_pars['xyzuvw_cov'])
same_lnols = gf.get_lnoverlaps(ass_cov_now, ass_mn_now, star_pars['xyzuvw_cov'], star_pars['xyzuvw_cov'], nstars)
same_lnols = gf.get_lnoverlaps(ass_cov_now, ass_mn_now, star_pars['xyzuvw_cov'], star_pars['xyzuvw'], nstars)
same_lnols
same_lnols[:50]
same_lnols[40:90]
get_ipython().magic(u'save field_investigation.py 1-74')
get_ipython().magic(u'save field_investigation.py 1-83')
ass_cov_now
baynan_cov_now = np.copy(ass_cov_now)
banyan_cov_now.shape
banyan_cov_now = np.copy(ass_cov_now)
baynan_cov_now = None
baynan_cov_now
p
banyan_cov_now[3:6,:3] = 0
banyan_cov_now
banyan_cov_now[:3,3:6] = 0
banyan_cov_now
banyan_lnols = gf.get_lnoverlaps(banyan_cov_now, ass_mn_now, star_pars['xyzuvw_cov'], star_pars['xyzuvw'], nstars)
banyan_lnols.shape
banyan_lnols[40:60]
np.min(banyan_lnols[:50])
np.max(banyan_lnols[50:])
np.where(banyan_lnols[:50] < np.max(banyan_lnols[50:]))
np.where(banyan_lnols[:50] > np.min(banyan_lnols[:50]))
np.percentile(banyan_lnols[:50], 10)
np.percentile(banyan_lnols[:50], 50)
np.percentile(banyan_lnols[:50], 90)
np.percentile(banyan_lnols[:50], 10)
np.percentile(banyan_lnols[:50], 5)
np.percentile(banyan_lnols[:50], 0)
percent = 50
np.where(banyan_lnols[:50] > np.percentile(banyan_lnols[:50], percent))
np.where(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], percent))
np.where(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 0))
np.where(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 20))
np.where(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 40))
np.where(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 30))
np.sum(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 0))
np.sum(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 0))/1000.
np.sum(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 0))/1000. * 100.
np.sum(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 10))/1000. * 100.
np.sum(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 20))/1000. * 100.
np.sum(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 20))/50. * 100.
np.sum(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 0))/50. * 100.
np.sum(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 90))/50. * 100.
np.sum(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 10))/50. * 100.
np.sum(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 20))/50. * 100.
for perc in [0, 10, 20, 30, 40,50,60,70,80,90]:
    print(np.sum(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], perc))/50. * 100.)
    
banyan_cov_now
membership_recovery = [100, 90, 80, 70, 60, 50, 40, 30, 20, 0]
contamination = []
for recov in membership_recovery:
    contamination.append(np.sum(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 100-recov))/50. * 100.)
    
    
contamination
plot
import matplotlib.pyplot as plt
plt.plot(membership_recovery, contamination)
plt.xlabel('Member recovery')
plt.ylabel('Contamination rate')
plt.savefig("contamination-rate.png")
plt.show()
ass_cov_now
basic_cov_now = np.copy(ass_cov_now)
basic_cov_now = np.copy(banyan_cov_now)
basic_cov_now
indexer = np.eye(6)
indexer[:3,:3] = 1
inexer
indexer
basic_cov_now[np.where(indexer != 1.0)]
basic_cov_now[np.where(indexer != 1.0)] = 0.0
basic_cov_now
basic_lnols = gf.get_lnoverlaps(basic_cov_now, ass_mn_now, star_pars['xyzuvw_cov'], star_pars['xyzuvw'], nstars)
basic_lnols
basic_con = []
membership_recovery
membership_recovery = [100, 95, 90, 85, 80, 75, 70, 60, 50, 40, 30, 2010]
for recov in membership_recovery:
    basic_con.append(np.sum(basic_lnols[50:] > np.percentile(banyan_lnols[:50], 100-recov))/50. * 100.)
    
    
    
for recov in membership_recovery:
    basic_con.append(np.sum(basic_lnols[50:] > np.percentile(basic_lnols[:50], 100-recov))/50. * 100.)
    
membership_recovery
membership_recovery[-1] = 20
for recov in membership_recovery:
    basic_con.append(np.sum(basic_lnols[50:] > np.percentile(basic_lnols[:50], 100-recov))/50. * 100.)
    
basic_con
basic_con = []
for recov in membership_recovery:
    basic_con.append(np.sum(basic_lnols[50:] > np.percentile(basic_lnols[:50], 100-recov))/50. * 100.)
    
basic_con
membership_recovery
banyan_con = []
for recov in membership_recovery:
    banyan_con.append(np.sum(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 100-recov))/50. * 100.)
    
banyan_con
banyan_lnols
basic_lnols
get_ipython().magic(u'ls ')
get_ipython().magic(u'pwd ')
banyan_con = []
for recov in membership_recovery:
    banyan_con.append(np.sum(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 100-recov) + 50)/50. * 100.)
    
banyan_con
for recov in membership_recovery:
    banyan_con.append(np.sum(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 100-recov))/50. * 100.)
    
banyan_con
for recov in membership_recovery:
    banyan_con.append(np.sum(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 100-recov) +50.)/50. * 100.)
    
banyan_con
for recov in membership_recovery:
    banyan_con.append(1/(50./np.sum(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 100-recov)) +1)* 100.)
    
    
banyan_con
banyan_con = []
for recov in membership_recovery[:6]:
    banyan_con.append(1/(50./np.sum(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 100-recov)) +1)* 100.)
    
    
banyan_con = []
for recov in membership_recovery[:6]:
    banyan_con.append(1/(50./np.sum(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 100-recov)) +1)* 100.)
    
banyan_con
for recov in membership_recovery[:7]:
    banyan_con.append(1/(50./np.sum(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 100-recov)) +1)* 100.)
   
baynan_con = []
baynan_con 
for recov in membership_recovery[:7]:
    banyan_con.append(1/(50./np.sum(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 100-recov)) +1)* 100.)
   
banyan_con
for recov in membership_recovery[:7]:
   print( banyan_con.append(1/(50./np.sum(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 100-recov)) +1)* 100.) )
   
banyan_con]
banyan_con
banyan_con = []
banyan_con
for recov in membership_recovery[:7]:
   print( banyan_con.append(1/(50./np.sum(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 100-recov)) +1)* 100.) )
   
banyan_con
basic_con = []
plt.plot(membership_recovery[:7], banyan_con)
plt.xlabel('Member recovery')
plt.ylabel('Contamination rate')
plt.savefig("contamination-rate.png")
approx_mlist = np.where(banyan_lnols > np.percentile(banyan_lnols[:50], 5))
len(approx_mlist[0])
approx_mlist[0]
approx_mlist[0] < 50
np.sum(approx_mlist[0] < 50)
banyan_con[1]
np.sum(approx_mlist[0] > 50)
mns.shape
true_mlist = np.where(banyan_lnols[:50] > np.percentile(banyan_lnols[:50], 5))
true_mlist
len(true_mlist[0])
false_mlist = np.where(banyan_lnols[50:] > np.percentile(banyan_lnols[:50], 5))
false_mlist
plt.clf()
(mns[:50][true_mlist,0])
plt.plot(mns[:50][true_mlist,0], mns[:50][true_mlist,1], '.')
plt.plot(mns[50:][false_mlist,0], mns[50:][false_mlist,1], '.')
plt.xlabel("X [pc]")
plt.ylabel("Y [pc]")
plt.savefig("top95.png")
plt.show()
plt.clf()
plt.plot(mns[50:][false_mlist,0], mns[50:][false_mlist,1], '.', color='orange')
plt.plot(mns[:50][true_mlist,0], mns[:50][true_mlist,1], '.', color='blue')
plt.xlabel("X [pc]")
plt.ylabel("Y [pc]")
plt.savefig("top95.png")
plt.show()
plt.plot(mns[:50][true_mlist,3], mns[:50][true_mlist,4], '.', color='blue')
plt.plot(mns[50:][false_mlist,3], mns[50:][false_mlist,4], '.', color='orange')
plt.xlabel("U [km/s]")
plt.ylabel("V [km/s]")
plt.savefig("top95UV.png")
plt.show()
