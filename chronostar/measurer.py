"""
measurer module

Replicates the measurement of synthetic stars with precisely known XYZUVW values
by converting into RA, DEC, parallax, proper motion, radial velocity with
appropriate errors.
"""

from __future__ import print_function, division

import numpy as np

