
import logging
import sys

sys.path.insert(0, '..')

import chronostar.retired2.converter as cv
import chronostar.compfitter as gf

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    temp_dir = 'temp_data/'
    astro_table_file = temp_dir + 'astro_table.txt'
    temp_xyzuvw_save_file = temp_dir + 'xyzuvw_now.fits'

    xyzuvw_dict_orig = cv.convertMeasurementsToCartesian(
        loadfile=astro_table_file, savefile=temp_xyzuvw_save_file
    )

    xyzuvw_dict_load = gf.loadXYZUVW(temp_xyzuvw_save_file)