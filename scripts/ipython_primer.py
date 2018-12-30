# coding: utf-8
get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')
import numpy as np
import sys
sys.path.insert(0, '..')
try:
    sys.path.insert(0, '~/code/chronostar')
except:
    print("Only inserted relative path to chronostar from /scripts")
np.set_printoptions(suppress=True)
