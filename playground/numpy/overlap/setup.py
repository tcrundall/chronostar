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
_overlap = Extension("_overlap",
                    ["overlap.i", "overlap.c"],
                    include_dirs = [numpy_include],
                    )

# overlap setup
setup(  name        = "overlap function",
        description = "Function that peforms a overlap",
        author      = "Egor Zindy",
        version     = "1.0",
        ext_modules = [_overlap]
        )

