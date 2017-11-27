#! /usr/bin/env python -W ignore

import unittest
import maths_test
import synthesiser_test
import groupfitter_test

maths_suite       = unittest.TestLoader().loadTestsFromModule(maths_test)
synthesiser_suite = unittest.TestLoader().loadTestsFromModule(synthesiser_test)
groupfitter_suite = unittest.TestLoader().loadTestsFromModule(groupfitter_test)

alltests = unittest.TestSuite([
    maths_suite, synthesiser_suite, groupfitter_suite
    #maths_suite, groupfitter_suite
    ])

unittest.TextTestRunner(verbosity=2).run(alltests)
