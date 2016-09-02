import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt

def get_bad():
    bad_stars = np.where( (crvad2['e_RV'] > 10) &
        ((crvad2['e_Plx']/crvad2['Plx'] >=  0.2) |
         (crvad2['e_Plx']/crvad2['Plx'] <= -0.2)))[0]
    return bad_stars

def get_good():
    good_stars = (crvad2['e_RV'] < 5) & \
                   (crvad2['Plx']/crvad2['e_Plx'] >  5)
    return good_stars

def get_close():
    boundary = 80
    close_stars = np.where( (crvad2['Plx'] < -boundary) |
                            (crvad2['Plx'] >  boundary))[0]
    return close

def get_bright():
    threshold = 6
    bright_stars = np.where( (crvad2['Bmag']) < threshold)[0]
    return bright_stars

def get_pl_loc():
    width = 0.1
    height = 0.8
    threshold = 6
    pl_loc_stars = np.where( (crvad2['Bmag'] < threshold)      &
                             (crvad2['RAhour'] < 3.79 + width) & 
                             (crvad2['RAhour'] > 3.79 - width) &
                             (crvad2['DEdeg'] < 24.123 + height) &
                             (crvad2['DEdeg'] > 24.123 - height) )[0]
    return pl_loc_stars

def get_pl_loc2():
    width = 1 
    height = 15 
    threshold = 6
    pl_loc_stars = np.where( (crvad2['Bmag'] < threshold)      &
                             (crvad2['RAhour'] < 3.79 + width) & 
                             (crvad2['RAhour'] > 3.79 - width) &
                             (crvad2['DEdeg'] < 24.123 + height) &
                             (crvad2['DEdeg'] > 24.123 - height) &
                             (crvad2['Plx']/crvad2['e_Plx'] >  5) &
                             (crvad2['e_RV'] < 5) )[0]
    return pl_loc_stars

if __name__ == "main":
   
    crvad2 = Table.read('crvad2.dat',
                 readme='crvad2.ReadMe',
                 format='ascii.cds')


    array1 = get_bright()
    plt.plot(crvad2['RAhour'][array1],crvad2['DEdeg'][array1],'.')
    array2 = get_pl_loc2()
    plt.plot(crvad2['RAhour'][array2],crvad2['DEdeg'][array2],'.')

    print array2

    for i in range(len(array2)):
        print array2[i]

    sum_pmRA = 0
    sum_pmDE = 0
    if (False):
        for star in array2:
            sum_pmRA += crvad2['pmRA'][star]
            sum_pmDE += crvad2['pmDE'][star]
            print str(crvad2['Plx'][star]) + " +- " + str(crvad2['e_Plx'][star])
            print "Proper motion RA: " + str(crvad2['pmRA'][star]) + " +- " + str(crvad2['e_pmRA'][star])
            print "Proper motion DE: " + str(crvad2['pmDE'][star]) + " +- " + str(crvad2['e_pmDE'][star])

        print "Average pmRA: " + str(sum_pmRA/len(array2))
        print "Average pmDE: " + str(sum_pmDE/len(array2))

    #Here is Mike's approach
    #import pdb; pdb.set_trace()
    med_pmRA = np.median(crvad2['pmRA'][array2])
    med_pmDE = np.median(crvad2['pmDE'][array2])
    med_RV = np.median(crvad2['RV'][array2])
    print "Median pmRA,pmDE,RV: {0:5.2f} {1:5.2f} {2:5.2f}"\
           .format(med_pmRA,med_pmDE,med_RV)
    
    #plt.show()
