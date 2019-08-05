"""
Select Sco-Cen candidates in the XYZUVW space from the entire Gaia catalog

XYZUVW boundaries are from scripts/get_association_region.py

"""

from astropy.table import Table

gaia = Table.read('../data/ScoCen_box_result_with_kinematics.fits')



'Upper Scorpius'
box lower bound = [ 56.87367643, -69.18555335,  22.25421284, -10.01563496, -13.54545694, -11.41068768]
box upper bound = [218.26147375,  25.87253732, 130.83646679 , 23.02256602, 9.30307811, 11.23932607]

'Upper Centaurus Lupus'
box lower bound = [  32.79498384, -157.99533042,   -4.08698572,  -15.01930024,  -29.85332018, -8.95795674]
box upper bound = [285.44134956 ,  1.1258381,  139.3493295,   25.61949509,   6.69554063, 11.3932266 ]

'Lower Centaurus-Crux'
box lower bound = [  -8.28344555, -299.72023071,  -22.29055599,  -31.20408951,  -35.59503406, -20.9189266 ]
box upper bound = [222.71232615, -29.81057382,  97.87765013,  16.18878038, 11.13683546, 12.17198221]