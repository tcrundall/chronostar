'''
Get Gaia astrometry for the 1400 Gagne moving group members. Combine with GALAH radial velocities where possible, as they are more precise.
'''
import numpy as np
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia


# Search with 2MASS IDs
# tmass is a list of 2mass ids (with no J letter, e.g. '18453485-3750195').
#~ tmass=['18453485-3750195', '18453485-3750195']
#~ query='''
#~ SELECT gaia.*, tmass.*
#~ FROM gaiadr2.gaia_source AS gaia
#~ INNER JOIN gaiadr2.tmass_best_neighbour AS tmass
    #~ ON gaia.source_id = tmass.source_id
#~ WHERE tmass.original_ext_source_id IN %s
#~ '''%str(tuple(tmass))

# Search with Gaia IDs
gaiaid=[5283961585534643712, 5283965296387249920]
query='''
SELECT gaia.*
FROM gaiadr2.gaia_source AS gaia
WHERE gaia.source_id IN %s
'''%str(tuple(gaiaid))

job = Gaia.launch_job_async(query, dump_to_file=True)

# Your astropy table with results
r = job.get_results()

keys=['source_id', 'phot_bp_mean_flux','ra_pmdec_corr','ra_error','ra','pmra_error','ecl_lon','designation','l','phot_rp_mean_mag','parallax_pmdec_corr','ra_parallax_corr','pmdec_error','phot_g_mean_mag','pmra','parallax','radial_velocity','radial_velocity_error','ra_dec_corr','parallax_error','dec_pmdec_corr','dec_error','pmdec','parallax_over_error','b','ref_epoch','ra_pmra_corr','dec_parallax_corr','phot_bp_mean_mag','dec','dec_pmra_corr','pmra_pmdec_corr','parallax_pmra_corr','bp_rp','ecl_lat']

r2=r[keys]
d=dict(zip(r2['source_id'], r2)) # for easier crossmatch with source_id

print r2

for source_id in gaiaid:
    try:
        g=d[source_id]
        print source_id, g['parallax']
    except:
        print source_id, 'No entry found in the Gaia catalogs.'
