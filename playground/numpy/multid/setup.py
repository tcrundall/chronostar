#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig

#Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory. This logic works across numpy versions
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# multid extension module
_multid = Extension("_multid",
                    ["multid.i", "multid.c"],
                    include_dirs = [numpy_include],
                    )

# NumpyTypemapTests setup
setup(  name        = "multid function",
        description = "multid takes a double array and doubles each of its elements in-place",
        author      = "Egor Zindy",
        version     = "1.0",
        ext_modules = [_multid]
        )

