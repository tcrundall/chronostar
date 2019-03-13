# coding: utf-8
get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
import numpy as np
np.set_printoptions(suppress=True)
import os
import sys
# sys.path.insert(0, '..')
sys.path.insert(0, os.path.abspath('..'))
try:
    sys.path.insert(0, '~/code/chronostar')
except:
    print("Only inserted relative path to chronostar from /scripts")
from chronostar.synthdata import SynthData
from chronostar import tabletool
from chronostar.component import SphereComponent
