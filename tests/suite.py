#! /usr/bin/env python -W ignore

import unittest
import maths_test
import synthesiser_test
import groupfitter_test
import analyser_test
import utils_test

maths_suite       = unittest.TestLoader().loadTestsFromModule(maths_test)
synthesiser_suite = unittest.TestLoader().loadTestsFromModule(synthesiser_test)
groupfitter_suite = unittest.TestLoader().loadTestsFromModule(groupfitter_test)
analyser_suite    = unittest.TestLoader().loadTestsFromModule(analyser_test)
utils_suite       = unittest.TestLoader().loadTestsFromModule(utils_test)
quadplotter_suite = unittest.TestLoader().loadTestsFromModule(quadplotter_test)

alltests = unittest.TestSuite([
    maths_suite, synthesiser_suite, groupfitter_suite,
    analyser_suite, utils_suite, quadplotter_suite
    ])

unittest.TextTestRunner(verbosity=2).run(alltests)
