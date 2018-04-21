# coding: utf-8
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
bp_xyzuvw_now
bp_xyzuvw_now
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
bp_then_inv = bp_xyzuvw_then.copy()
bp_then_inv
bp_then_inv[3:6] = - bp_then_inv[3:6]
bp_then_inv
bp_xyzuvw_then
tb.integrate(bp_then_inv, np.array([0,age]))
tb.integrate_xyzuvw(bp_then_inv, np.array([0,age]))
tb.integrate_xyzuvw(tb.xyzuvw_to_skycoord(bp_then_inv), np.array([0,age]))
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
bp_xyzuvw_many
bp_xyzuvw_many
get_ipython().magic(u'run traceback_checker.py')
bp_xyzuvw_many_larger
tstep
larger_ntimes
get_ipython().magic(u'run traceback_checker.py')
tstep
larger_ntimes
get_ipython().magic(u'run traceback_checker.py')
bp_xyzuvw_many_larger.shape
larger_ntimes
larger_ntimes
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
bp_xyzuvws
bp_xyzuvws - bp_xyzuvw_many[::-1]
np.max(bp_xyzuvws - bp_xyzuvw_many[::-1])
get_ipython().magic(u'run traceback_checker.py')
bp_xyzuvw_many[::-1]
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
bp_xyzuvw_then_cheating
bp_xyzuvws[-1]
bp_xyzuvws[-1]
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
bp_xyzuvw_then-cheating
bp_xyzuvw_then_cheating
bp_xyzuvw_then_cheating_sky
tb.integrate(bp_xyzuvw_then_cheating_sky, np.array(0, 1e-3))[0]
tb.integrate_xyzuvw(bp_xyzuvw_then_cheating_sky, np.array(0, 1e-3))[0]
tb.integrate_xyzuvw(bp_xyzuvw_then_cheating_sky, np.array([0, 1e-3]))[0]
bp_xyzuvw_then_cheating
tb.xyzuvw_to_skycoord(bp_xyzuvw_then_cheating, solarmotion='schoenrich', reverse_x_sign=True)
bp_xyzuvw_then_cheating_sky
bp_xyzuvw_then_cheating_sky = tb.xyzuvw_to_skycoord(bp_xyzuvw_then_cheating, solarmotion='schoenrich', reverse_x_sign=True)
tb.integrate_xyzuvw(bp_xyzuvw_then_cheating_sky, np.array([0, 1e-3]))[0]
bp_xyzuvw_then_cheating
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
bp_xyzuvw_then_cheating
stars_cheating = tb.stars_table(bp_xyzuvw_then_cheating_sky)
bp_xyzuvw_cheating = tb.traceback(stars_cheating, times)[0]
bp_xyzuvw_cheating
bp_xyzuvw_cheating[::-1] - bp_xyzuvw
bp_xyzuvw_cheating[::-1] - bp_xyzuvws
get_ipython().magic(u'run traceback_checker.py')
bp_xyzuvws_cheating
bp_xyzuvws
bp_xyzuvws_cheating
get_ipython().magic(u'run traceback_checker.py')
bp_xyzuvws_cheating
bp_xyzuvws
bp_xyzuvws_cheating
bp_xyzuvws
bp_xyzuvws_cheating
bp_xyzuvws
get_ipython().magic(u'run traceback_checker.py')
bp_xyzuvws_cheating
bp_xyzuvws
bp_xyzuvws_cheating
bp_xyzuvws_cheating[::-1]
get_ipython().magic(u'run traceback_checker.py')
bp_xyzuvws_cheating
bp_xyzuvws
bp_xyzuvws_cheating
bp_xyzuvws
bp_xyzuvws_cheating
bp_xyzuvws
bp_xyzuvws_cheating
bp_xyzuvws
bp_xyzuvws_cheating
bp_xyzuvws
bp_xyzuvws - bp_xyzuvws_cheating
np.max(bp_xyzuvws - bp_xyzuvws_cheating)
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
bp_xyzuvws_cheating
bp_xyzuvws
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
get_ipython().magic(u'run traceback_checker.py')
bp_xyzuvw_now
tb.xyzuvw_to_skycoord(bp_xyzuvw_now + np.array([0,0,0,3,0,0])
)
tb.xyzuvw_to_skycoord(bp_xyzuvw_now + np.array([0,0,0,3,0,0]), 'schoenrich', True)
get_ipython().magic(u'run traceback_checker.py')
