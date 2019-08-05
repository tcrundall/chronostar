"""
Author: Marusa Zerjal, 2019 - 07 - 15

Take Sco-Cen components fitted to 6D data and make overlaps
(using covariance matrix) with stars missing radial velocities
in order to find more Sco-Cen candidates.

"""

import numpy as np
from astropy.table import Table, vstack, join
import matplotlib.pyplot as plt

# READ DATA
d = Table.read('data_table_cartesian_including_tims_stars_with_bg_ols_and_component_overlaps.fits')
Gmag = d['phot_g_mean_mag'] - 5 * np.log10(1.0 / (d['parallax'] * 1e-3) / 10)  # tab['parallax'] in micro arcsec
d['Gmag'] = Gmag
dist=1/d['parallax']*1000.0
d['dist']=dist

# Main sequence parametrization
# fitpar for pmag, rpmag
fitpar = [0.17954163, -2.48748376, 12.9279348, -31.35434182, 38.31330583, -12.25864507]
poly = np.poly1d(fitpar)
x = np.linspace(1, 4, 100)
y = poly(x)
m = y > 4
yms = y[m]
xms = x[m]

def tims_components():
    """
    Check if there are any duplicate components
    :return:
    """
    c = np.load('all_nonbg_scocen_comps.npy')
    print('components', c.shape)

    print('Are there duplicate components?')
    for i, x in enumerate(c[:-1]):
        for j, y in enumerate(c):
            if j<1 or i==j:
                continue
            #print x, y
            f=np.allclose(x, y, atol=1)

            if f:
                print i, j, f


def tims_members():
    # USco
    data_usco = Table.read('usco_res/usco_run_subset.fit')
    memb_usco = np.load('usco_res/final_membership.npy')
    for i in range(1, memb_usco.shape[1]):
        data_usco['prob_c%d'%i]=memb_usco[:,i-1]
    data_usco['prob_bg']=memb_usco[:,-1]

    # UCL
    data_ucl = Table.read('ucl_res/ucl_run_subset.fit')
    memb_ucl = np.load('ucl_res/final_membership.npy')
    for i in range(1, memb_ucl.shape[1]):
        data_ucl['prob_c%d'%i]=memb_ucl[:,i-1]
    data_ucl['prob_bg']=memb_ucl[:,-1]

    # LCC
    data_lcc = Table.read('lcc_res/lcc_run_subset.fit')
    memb_lcc = np.load('lcc_res/final_membership.npy')
    for i in range(1, memb_lcc.shape[1]):
        data_lcc['prob_c%d'%i]=memb_lcc[:,i-1]
    data_lcc['prob_bg']=memb_lcc[:,-1]

    # Print number of components
    print memb_usco.shape[1]-1, memb_ucl.shape[1]-1, memb_lcc.shape[1]-1

    return data_usco, data_ucl, data_lcc

def are_tims_members_in_my_data_table(d):
    data_usco, data_ucl, data_lcc = tims_members()
    data = [data_usco, data_ucl, data_lcc]

    for t in data:
        mask = np.in1d(d['source_id'], t['source_id'])
        dt=d[mask]
        print len(d), len(t), len(dt)

def cmd_for_each_component(d, show=True):
    """
    Plot CMD for members of each component
    :param d:
    :return:
    """

    # Minimal probability required for membership
    p=0.5

    fig=plt.figure()
    for i in range(1, 15+1):
        ax = fig.add_subplot(4, 4, i)
        mask=d['comp_overlap_%d'%i]>p
        t=d[mask]

        if len(t)>100:
            alpha=0.5
        else:
            alpha=1

        ax.scatter(t['bp_rp'], t['Gmag'], s=1, c='k', alpha=alpha)

        ax.plot(xms, yms, c='brown', label='Median main sequence', linewidth=1)
        ax.plot(xms, yms - 1, c='brown', label='1 mag above the median', linewidth=1, linestyle='--')
        ax.plot(xms, yms - 1.5, c='brown', label='1.5 mag above the median', linewidth=1, linestyle='--')

        ax.axvline(x=0.369, linewidth=0.5, color='k')  # F
        ax.axvline(x=0.767, linewidth=0.5, color='k')  # G
        ax.axvline(x=0.979, linewidth=0.5, color='k')  # K
        ax.axvline(x=1.848, linewidth=0.5, color='k')  # M

        ax.set_xlim(-1, 6)
        ax.set_ylim(16, -2)

        # Sum probability
        sump=np.sum(d['comp_overlap_%d'%i])
        ax.set_title('%d (%d)'%(len(t), sump))


    # Plot background component
    ax = fig.add_subplot(4, 4, 16)
    mask=d['comp_overlap_bg']>p
    t=d[mask]

    ax.scatter(t['bp_rp'], t['Gmag'], s=1, c='k', alpha=0.5)

    ax.plot(xms, yms, c='brown', label='Median main sequence', linewidth=1)
    ax.plot(xms, yms - 1, c='brown', label='1 mag above the median', linewidth=1, linestyle='--')
    ax.plot(xms, yms - 1.5, c='brown', label='1.5 mag above the median', linewidth=1, linestyle='--')

    ax.axvline(x=0.369, linewidth=0.5, color='k')  # F
    ax.axvline(x=0.767, linewidth=0.5, color='k')  # G
    ax.axvline(x=0.979, linewidth=0.5, color='k')  # K
    ax.axvline(x=1.848, linewidth=0.5, color='k')  # M

    ax.set_xlim(-1, 6)
    ax.set_ylim(16, -2)

    # Sum probability
    sump=np.sum(d['comp_overlap_bg'])
    ax.set_title('%d (%d)'%(len(t), sump))

    ax.set_xlabel('BP-RP')
    ax.set_ylabel('Gaia G mag')

    if show:
        plt.show()

def cmd_for_each_component_for_Tims_members(show=True):
    """
    Plot CMD for members of each component
    :param d:
    :return:
    """

    data_usco, data_ucl, data_lcc = tims_members()
    data = [data_usco, data_ucl, data_lcc]

    # Minimal probability required for membership
    p=0.5

    for d in data:
        fig=plt.figure()
        for i in range(1, 15+1):
            ax = fig.add_subplot(4, 4, i)
            mask=d['prob_c%d'%i]>p
            t=d[mask]

            if len(t)>100:
                alpha=0.5
            else:
                alpha=1

            ax.scatter(t['bp_rp'], t['Gmag'], s=1, c='k', alpha=alpha)

            ax.plot(xms, yms, c='brown', label='Median main sequence', linewidth=1)
            ax.plot(xms, yms - 1, c='brown', label='1 mag above the median', linewidth=1, linestyle='--')
            ax.plot(xms, yms - 1.5, c='brown', label='1.5 mag above the median', linewidth=1, linestyle='--')

            ax.axvline(x=0.369, linewidth=0.5, color='k')  # F
            ax.axvline(x=0.767, linewidth=0.5, color='k')  # G
            ax.axvline(x=0.979, linewidth=0.5, color='k')  # K
            ax.axvline(x=1.848, linewidth=0.5, color='k')  # M

            ax.set_xlim(-1, 6)
            ax.set_ylim(16, -2)

            # Sum probability
            sump=np.sum(d['prob_c%d'%i])
            ax.set_title('%d (%d)'%(len(t), sump))


        # Plot background component
        ax = fig.add_subplot(4, 4, 16)
        mask=d['prob_bg']>p
        t=d[mask]

        ax.scatter(t['bp_rp'], t['Gmag'], s=1, c='k', alpha=0.5)

        ax.plot(xms, yms, c='brown', label='Median main sequence', linewidth=1)
        ax.plot(xms, yms - 1, c='brown', label='1 mag above the median', linewidth=1, linestyle='--')
        ax.plot(xms, yms - 1.5, c='brown', label='1.5 mag above the median', linewidth=1, linestyle='--')

        ax.axvline(x=0.369, linewidth=0.5, color='k')  # F
        ax.axvline(x=0.767, linewidth=0.5, color='k')  # G
        ax.axvline(x=0.979, linewidth=0.5, color='k')  # K
        ax.axvline(x=1.848, linewidth=0.5, color='k')  # M

        ax.set_xlim(-1, 6)
        ax.set_ylim(16, -2)

        # Sum probability
        sump=np.sum(d['prob_bg'])
        ax.set_title('%d (%d)'%(len(t), sump))

        ax.set_xlabel('BP-RP')
        ax.set_ylabel('Gaia G mag')

    if show:
        plt.show()

def histogram_for_each_component(d, show=True):
    """
    Plot histogram of membership probability for members of each component
    :param d:
    :return:
    """

    bins=np.linspace(0, 1, 21)

    fig=plt.figure()
    for i in range(1, 15+1):
        ax = fig.add_subplot(4, 4, i)

        ax.hist(d['comp_overlap_%d'%i], bins=bins)

        ax.set_ylim(0, 1000)

        # Sum probability
        sump=np.sum(d['comp_overlap_%d'%i])
        ax.set_title('N(sum(p)): %d, N(p$>$0.5): %d'%(sump, len(d[d['comp_overlap_%d'%i]>0.5])))


    # Plot background component
    ax = fig.add_subplot(4, 4, 16)
    ax.hist(d['comp_overlap_bg'], bins=np.linspace(0, 1, 100))

    ax.set_ylim(0, 2000)

    # Sum probability
    sump=np.sum(d['comp_overlap_bg'])
    ax.set_title('N(sum(p)): %d, N(p$>$0.5): %d'%(sump, len(d[d['comp_overlap_%d'%i]>0.5])))

    if show:
        plt.show()

def distance_for_each_component(d, show=True):
    """
    Plot distance histogram of membership probability for members of each component
    :param d:
    :return:
    """

    # Minimal probability required for membership
    p=0.5

    bins=np.linspace(0, 250, 50)

    fig=plt.figure()
    for i in range(1, 15+1):
        ax = fig.add_subplot(4, 4, i)

        mask=d['comp_overlap_%d'%i]>p
        t=d[mask]

        ax.hist(t['dist'], bins=bins)

#        ax.set_ylim(0, 1000)



    # Plot background component
    ax = fig.add_subplot(4, 4, 16)
    ax.hist(d['dist'], bins=bins)

    ax.set_ylim(0, 2000)

    # Sum probability
    sump=np.sum(d['comp_overlap_bg'])
    ax.set_title('N(sum(p)): %d, N(p$>$0.5): %d'%(sump, len(d[d['comp_overlap_%d'%i]>0.5])))

    if show:
        plt.show()



def cmd_for_candidate_members(d):
    """
    I'm a but lost in here. Split this code into smaller parts.

    :param d:
    :return:
    """

    mask13=d['comp_overlap_13']>0.5
    d13=d[mask13]

    mask = d['comp_overlap_bg'] > 0.5
    tbg = d[mask]

    mask=d['comp_overlap_1']<-10 # everything False
    for i in range(1, 15+1):
        #if i==13:
        #    continue
        mask = np.logical_or(mask, d['comp_overlap_%d'%i]>0.5)
    d=d[mask]

    #mask=d['comp_overlap_bg']<0.1
    #mask=(d['comp_overlap_bg']<0.1) & (d['comp_overlap_13']<0.1) # component 13 is weird: mostly main sequence stars
    #d=d[mask]

    # fitpar for pmag, rpmag
    fitpar = [  0.17954163,  -2.48748376,  12.9279348,  -31.35434182,  38.31330583, -12.25864507]
    poly = np.poly1d(fitpar)

    # Count stars 1 mag or more above the main sequence
    mask = (d['Gmag']<poly(d['bp_rp'])-1) & (d['bp_rp']>1)
    print len(d[mask]), float(len(d[mask]))/float(len(d))

    fig=plt.figure()
    ax=fig.add_subplot(111)
    mask = d['comp_overlap_13'] > 0.5
    ax.scatter(d[~mask]['bp_rp'], d[~mask]['Gmag'], s=1)
    #ax.scatter(d13['bp_rp'], d13['Gmag'], s=1, c='r')

    # Plot main sequence parametrization
    x = np.linspace(1, 4, 100)
    y = poly(x)
    m = y > 4
    y = y[m]
    x = x[m]
    ax.plot(x, y, c='brown', label='Median main sequence', linewidth=1)
    ax.plot(x, y - 1, c='brown', label='1 mag above the median', linewidth=1, linestyle='--')
    ax.plot(x, y - 1.5, c='brown', label='1.5 mag above the median', linewidth=1, linestyle='--')


    figh=plt.figure()

    fig2=plt.figure()
    clrs=['k', 'b', 'g', 'orange', 'r']
    for i in range(1, 15+1):
        ax2 = fig2.add_subplot(4, 4, i)
        axh = figh.add_subplot(4, 4, i)
        for j, p in enumerate([0.5, 0.6, 0.7, 0.8, 0.9]):
            mask=d['comp_overlap_%d'%i]>p
            t=d[mask]
            if len(t)<5:
                continue

            ax2.scatter(t['bp_rp'], t['Gmag'], s=1, c=clrs[j])


            #mask=t['comp_overlap_%d'%i]>0.9
            #ax2.scatter(t[mask]['bp_rp'], t[mask]['Gmag'], s=1, c='r')

        axh.hist(t['comp_overlap_%d' % i], bins=20)

        trv=d[d['comp_overlap_%d'%i]>0.5]
        mask = trv['radial_velocity_error'] < 500
        ax2.set_title('%d (%d)'%(len(trv[mask]), len(trv[~mask])))

        ax2.set_xlim(-1, 6)
        ax2.set_ylim(14, -2)
        ax2.plot(x, y, c='brown', label='Median main sequence', linewidth=1)
        ax2.plot(x, y - 1, c='brown', label='1 mag above the median', linewidth=1, linestyle='--')
        ax2.plot(x, y - 1.5, c='brown', label='1.5 mag above the median', linewidth=1, linestyle='--')

        if i==13:
            continue
        #ax.scatter(t['bp_rp'], t['Gmag'], s=1)

    ax2 = fig2.add_subplot(4, 4, 16)
    ax2.scatter(tbg['bp_rp'], tbg['Gmag'], s=1)
    ax2.set_xlim(-1, 6)
    ax2.set_ylim(14, -2)
    ax2.plot(x, y, c='brown', label='Median main sequence', linewidth=1)
    ax2.plot(x, y - 1, c='brown', label='1 mag above the median', linewidth=1, linestyle='--')
    ax2.plot(x, y - 1.5, c='brown', label='1.5 mag above the median', linewidth=1, linestyle='--')


    ax.set_ylim(14, -2)
    ax.set_xlabel('BP-RP')
    ax.set_ylabel('Gaia G mag')

    ax.axvline(x=0.369, linewidth=0.5, color='k') # F
    ax.axvline(x=0.767, linewidth=0.5, color='k') # G
    ax.axvline(x=0.979, linewidth=0.5, color='k') # K
    ax.axvline(x=1.848, linewidth=0.5, color='k') # M

    plt.show()

def plot_Z_W_for_each_component(d, show=True):
    """
    Plot Z vs W for each component.

    :param d:
    :param show:
    :return:
    """

    # Minimal probability required for membership
    p = 0.5

    fig = plt.figure()

    for i in range(1, 15 + 1):
        ax = fig.add_subplot(4, 4, i)

        mask = d['comp_overlap_%d' % i] > p
        t = d[mask]

        ax.scatter(t['Z'], t['W'], c='k', s=1)

        # Plot stars with known RV
        mask = t['radial_velocity_error']<50.0
        t=t[mask]
        ax.scatter(t['Z'], t['W'], c='r', s=1)

    #        ax.set_ylim(0, 1000)


    #ax.set_ylim(0, 2000)

    if show:
        plt.show()

if __name__ == "__main__":
    #cmd_for_candidate_members(d)

    #are_tims_members_in_my_data_table(d)
    #cmd_for_each_component_for_Tims_members(show=True)

    #plot_Z_W_for_each_component(d, show=True)
    #exit(0)

    #distance_for_each_component(d, show=False)

    #exit(0)

    cmd_for_each_component(d, show=False)
    histogram_for_each_component(d, show=False)

    plt.show()

