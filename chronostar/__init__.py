
# coding: utf-8

"""  """

__author__ = "Michael Ireland <michael.ireland@anu.edu.au>"
__version__ = "0.1"


#!!! Note that from traceback import * doesn't work!
import error_ellipse
import expectmax
import traceback 
import fit_group
import groupfitter
import analyser
import synthesiser
try:
    import _overlap
except:
    print("overlap not imported, SWIG not possible. Need to make in"
            " directory...")

#Tim's code to play around with miscelaneous data sets neatly
# import play
