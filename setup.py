# coding: utf-8

""" A stellar orbit traceback code """

import os
import re
import sys

try:
    from setuptools import setup

except ImportError:
    from distutils.core import setup

try:
    from setuptools import Extension

except ImportError:
    from distutils.core import Extension


major, minor1, minor2, release, serial =  sys.version_info

readfile_kwargs = {"encoding": "utf-8"} if major >= 3 else {}

def readfile(filename):
    with open(filename, **readfile_kwargs) as fp:
        contents = fp.read()
    return contents

version_regex = re.compile("__version__ = \"(.*?)\"")
contents = readfile(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "chronostar",
    "__init__.py"))

version = version_regex.findall(contents)[0]

#Third-party modules - we depend on numpy for everything
import numpy

# Obtain the numpy include directory. This logic works across numpy versions
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

_overlap = Extension("chronostar/_overlap",
                    ["chronostar/overlap.i", "chronostar/overlap.c"],
                    include_dirs = [numpy_include],
                    libraries = ['gsl', 'gslcblas'],
                    )

setup(name="chronostar",
      version=version,
      author="Michael J. Ireland",
      author_email="michael.ireland@anu.edu.au",
      packages=["chronostar"],
      url="http://www.github.com/mikeireland/chronostar/",
      license="MIT",
      description="A stellar orbit traceback code.",
      long_description=readfile(os.path.join(os.path.dirname(__file__), "README.md")),
      install_requires=[
        "requests",
        "requests_futures"
      ],
      ext_modules = [_overlap]
     )
