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

# flatten extension module
_flatten = Extension("_flatten",
                    ["flatten.i", "flatten.c"],
                    include_dirs = [numpy_include],
                    )

# flatten setup
setup(  name        = "flatten function",
        description = "Function that peforms a flatten",
        author      = "Egor Zindy",
        version     = "1.0",
        ext_modules = [_flatten]
        )

