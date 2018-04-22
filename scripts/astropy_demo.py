from astropy.coordinates import SkyCoord



from astropy import units as u


c = SkyCoord(x=0, y=0, z=10, v_x=0, v_y=0, v_z=-1.0, unit='kpc', representation='cartesian')
d = SkyCoord(u=0, v=0, w=5, frame='galactic', representation='cartesian')