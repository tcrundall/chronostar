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

# overlap extension module
_just_checking = Extension("_just_checking",
                    ["overlap.i", "overlap.c"],
                    include_dirs = [numpy_include],
                    libraries = ['gsl', 'gslcblas'],
                    extra_compile_args = [
                        "-mmacosx-version-min=10.5"
                        ]
                    )

# NumpyTypemapTests setup
setup(  name        = "overlap function",
        description = "overlap calculates the overlap integral of (a) star(s) and a group",
        author      = "Tim Crundall",
        version     = "1.0",
        ext_modules = [_just_checking]
        )

