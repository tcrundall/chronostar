from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

def points_within():
    print("Getting fraction of points within 1 sigma for 3 and 6 dims")
    for exponent in [4,5,6]:
        for dim in [3,6]:
            npoints = 10**exponent
            points = np.random.randn(npoints, dim)
            rs = r_from_xyz(points)
            frac = 100.0*np.sum(rs < 1) / npoints
            print("{}D - within 1sigma: {}".format(dim,frac))
            frac = 100.0*np.sum(rs < 2) / npoints
            print("{}D - within 2sigma: {}".format(dim,frac))

def calc_KE():
    print("Stating numerically that T = 0.5 N M_star v_disp^2")
    nstars = 10000000
    for v_disp in [1,5]:
        vs = np.random.randn(nstars) * v_disp
        m_star = 0.7
        KE_num = np.sum(0.5 * m_star * vs**2)
        KE_anl = 0.5 * nstars * m_star * v_disp**2
        print("KE numerically: {}\nKE analytically: {}".format(KE_num, KE_anl))

def gauss3D(r, mu=0, sig=1):
    coeff = 4 * np.pi / np.sqrt( (2*np.pi)**3 * sig**6)
    expon = -(r-mu)**2 / (2*sig**2)
    return coeff * np.exp(expon) * r**2

def r_from_xyz(xyzs):
    xyzs = np.copy(xyzs)
    if len(xyzs.shape) == 1:
        xyzs = np.array([xyzs])
    return np.sqrt(np.sum(xyzs**2, axis=1))

    
def failure():
    r = 10.
    npoints = 10000000
    xyz = np.random.rand(npoints, 3) * 2.*r - r
    xyz_within = xyz[np.where(r_from_xyz(xyz) < r)]
    rs_within = r_from_xyz(xyz_within)
    npoints_within = xyz_within.shape[0]
    heights = np.random.rand(npoints_within)

    is_below = np.where(heights < gauss3D(rs_within))

    sphere_vol = 4 * np.pi / 3 * r**3
    hyper_sphere_vol = 1.0 * sphere_vol
    vol_under_curve = hyper_sphere_vol * len(is_below[0]) / len(heights)
    print("Volume under curve: {}".format(vol_under_curve))

def calc_integral(R, sig=1):
    npoints = 10000
    dr = R / npoints
    rs = np.linspace(0, R, npoints, endpoint=False) + 0.5*dr
    vals = gauss3D(rs, sig=sig)
    integral = np.sum(vals*dr)
    return integral

def calc_pe(s1, s2):
    r_diff =np.sqrt(np.sum((s1-s2)**2))
    return G * m_s**2 / r_diff**2 

m_s = 1.0
G = 1.0
nstars = 2000
SIG = 1.0
xyz = np.random.randn(nstars,3) * SIG
def calc_tot_pe_num(xyz):
    nstars = xyz.shape[0]
    tot_pe = 0
    for i in range(nstars):
        for j in range(i+1,nstars):
            tot_pe += calc_pe(xyz[i], xyz[j])
    return tot_pe

def calc_m_int(R, nstars=1000, sig=1.):
    """Checks out, have compared with monte carlo"""
    m_int = nstars * m_s * calc_integral(R, sig=sig) 
    return m_int

def calc_tot_pe_anl(xyz):
    nstars = xyz.shape[0]
    tot_pe = 0
    for i in range(nstars):
        r = r_from_xyz(xyz[i])[0]
        tot_pe += G * m_s * calc_m_int(r,nstars,SIG) / r**2
    return tot_pe

expons = np.array([2,2.5,3,3.25,3.5, 3.75, 4.0])
pe_nums = np.zeros(expons.shape[0])
pe_anls = np.zeros(expons.shape[0])
for i, expon in enumerate(expons):
    sig = 1.0
    nstars = int(10**expon)
    print(nstars)
    xyz = np.random.randn(nstars,3) * sig
    pe_num = calc_tot_pe_num(xyz)
    pe_nums[i] = pe_num
    print(pe_num)
    pe_anl = calc_tot_pe_anl(xyz)
    pe_anls[i] = pe_anl
    print(pe_anl)
    print(pe_num / pe_anl)
plt.plot(10**expons, pe_nums, c='b')
plt.plot(10**expons, pe_anls, c='r')
plt.show()
