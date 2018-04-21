#! /usr/bin/env python
"""This will test coordinates along the lines of subsample_density.py

This is important because of co-ordinate system reversals between systems for
X in particular.

To Do:
- confirm with Mike how similar we expect these values to actually be
- once confirmed, implement test with appropriate thresholds
- once ^^ is done, move back into root tests folder
"""
from __future__ import division, print_function

import numpy as np
import sys
import unittest

sys.path.insert(0,'../..') #hacky way to get access to module

from astropy import units as u 
from astropy.coordinates import SkyCoord
import chronostar.traceback as traceback

class TestCoords(unittest.TestCase):
    REVERSE_MALO_SIGN = False

    def setUp(self):
        #Use Beta Pic, TW Hya as tests. From Malo et al (2013) or
        #http://www.astro.umontreal.ca/~malo/banyan.php. We'll take a 
        #canonical "group center" star in each case.
        self.group_names = ['bPic', 'TW Hya', 'AB Dor', 'Carina']
        self.star_names = ['bPic', 'TW Hya', 'AB Dor', 'HIP 32235']
        self.xyzuvw_group_all = np.array([
            [9.27,-5.96,-13.59,-10.94,-16.25,-9.27],
            [12.49, -42.28, 21.55, -9.95, -17.91, -4.65],
            [-2.37, 1.48, -15.62, -7.12, -27.31, -13.81],
            [15.55, -58.53, -22.95, -10.34, -22.31, -5.91]
        ])
        self.star_radecpipmrv_all = [
            [86.82, -51.067, 51.44, 4.65, 83.1, 20], 
            [165.466, -34.705, 18.62, -66.19, -13.9, 13.4], 
            [82.187, -65.45, 65.93, 33.16, 150.83, 32.4], 
            [100.94, -71.977, 17.17, 6.17, 61.15, 20.7]
        ]

    def test_correctness(self):

        for group_name, star_name, xyzuvw_group, star_radecpipmrv in zip(
                self.group_names, self.star_names,
                self.xyzuvw_group_all, self.star_radecpipmrv_all,
            ):
            if self.REVERSE_MALO_SIGN:
                xyzuvw_group[0] *= -1

            # What does astropy think that the XYZ means? Nothing. We
            # unfortunately have to do the trigonometry ourselves, as
            # version 1.2.1 only supports cartesian coordinates for
            # ra, dec, degs.
            dist = np.sqrt(np.sum(xyzuvw_group[:3]**2))
            l = np.degrees(np.arctan2(xyzuvw_group[1], xyzuvw_group[0]))
            b = np.degrees(
                np.arctan2(
                    xyzuvw_group[2], np.sqrt(np.sum(xyzuvw_group[:2]**2))
                )
            )
            c = SkyCoord(l, b, dist, unit=['deg','deg','pc'], frame='galactic')
            print("\n*** Star: " + star_name + " in Group: " + group_name + " ***")
            print("*** Manual computation ***")
            print("Group center  RA: {:8.3f}".format(c.fk5.ra))
            print("Star          RA: {:8.3f}".format(star_radecpipmrv[0]))
            print("Group center Dec: {:8.3f}".format(c.fk5.dec))
            print("Star         Dec: {:8.3f}".format(star_radecpipmrv[1]))
            print("Group center Plx: {:6.1f}".format(1000./c.distance.pc))
            print("Star         Plx: {:6.1f}".format(star_radecpipmrv[2]))
            group_radecpipmrv=traceback.xyzuvw_to_skycoord(xyzuvw_group, None, False)
            print("*** traceback.py computation ***")
            print("Group center  RA: {:8.3f}".format(group_radecpipmrv[0]))
            print("Star          RA: {:8.3f}".format(star_radecpipmrv[0]))
            print("Group center Dec: {:8.3f}".format(group_radecpipmrv[1]))
            print("Star         Dec: {:8.3f}".format(star_radecpipmrv[1]))
            print("Group center Plx: {:6.1f}".format(group_radecpipmrv[2]))
            print("Star         Plx: {:6.1f}".format(star_radecpipmrv[2]))
            print("Group center pmRA: {:8.3f}".format(group_radecpipmrv[3]))
            print("Star         pmRA: {:8.3f}".format(star_radecpipmrv[3]))
            print("Group center pmDE: {:8.3f}".format(group_radecpipmrv[4]))
            print("Star         pmDE: {:8.3f}".format(star_radecpipmrv[4]))
            print("Group center   RV: {:6.1f}".format(group_radecpipmrv[5]))
            print("Star           RV: {:6.1f}".format(star_radecpipmrv[5]))

if __name__ == '__main__':
    unittest.main()

sys.path.insert(0,'.') #replace home directroy into path
