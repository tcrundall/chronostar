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

# array3D extension module
_array3D = Extension("_array3D",
                    ["array3D.i", "array3D.c"],
                    include_dirs = [numpy_include],
                    )

# NumpyTypemapTests setup
setup(  name        = "array3D function",
        description = "array3D takes a double array and doubles each of its elements in-place",
        author      = "Egor Zindy",
        version     = "1.0",
        ext_modules = [_array3D]
        )

