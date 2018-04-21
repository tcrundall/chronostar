import numpy as np
import sys

sys.path.insert(0, '..')

import chronostar.retired.tracingback as tb

bp_astr = [86.82, -51.067, 51.44, 4.65, 83.1, 20]

# modified so U is stronger
#bp_astr = [86.82, -51.06, 51.43, 4.85, 51.04, 19.47]

age = 2
times = np.array([0., age])

bp_xyzuvws = tb.integrate_xyzuvw(bp_astr, times)
bp_xyzuvw_now = bp_xyzuvws[0]
bp_xyzuvw_then = bp_xyzuvws[1]

print(" ------ xyzuvw_to_skycoord --------- ")
bp_astr_same = tb.xyzuvw_to_skycoord(bp_xyzuvw_now, 'schoenrich',
                                     True)  # , solarmotion='schoenrich')
print("Original astrometry is: {}".format(bp_astr))
print("Rederived astrometry is: {}".format(bp_astr_same))

# pdb.set_trace()
rtol = 1e-3

both = (np.allclose(
    tb.xyzuvw_to_skycoord(bp_xyzuvw_now, 'schoenrich', True), bp_astr,
    rtol=rtol)
)
# print(plain, x_sign, schoen, both)


print(" ------ tracing forward then back --------- ")
bp_xyzuvw_now_same = tb.trace_forward(bp_xyzuvw_then, age, solarmotion=None)
print(bp_xyzuvw_now)
print(bp_xyzuvw_now_same)

print(" ------- tracing forward via a mid point --------- ")
half_age = 0.5 * age
bp_xyzuvw_mid = tb.trace_forward(bp_xyzuvw_then, half_age, solarmotion=None)
bp_xyzuvw_now_same_same = tb.trace_forward(bp_xyzuvw_mid, half_age,
                                           solarmotion=None)

print(bp_xyzuvw_now)
print(bp_xyzuvw_now_same_same)

print(" ------- tracing forward via arbitrarily small steps ------ ")
ntimes = 101
tstep = float(age) / (ntimes - 1)

bp_xyzuvw_many = np.zeros((ntimes, 6))
bp_xyzuvw_many[0] = bp_xyzuvw_then

for i in range(1, ntimes):
    bp_xyzuvw_many[i] = tb.trace_forward(bp_xyzuvw_many[i - 1], tstep,
                                         solarmotion=None)

print("For {} steps of size {:.4f} Myr".format(ntimes, tstep))
print(bp_xyzuvw_now)
print(bp_xyzuvw_many[-1])
print(
    ' ... which works fine. But if we increase the age, but leave the '
    'timestep fixed')

if True:
    larger_age = 10 * age
    larger_ntimes = 10 * (ntimes - 1) + 1
    larger_tstep = float(larger_age) / (larger_ntimes - 1)

    bp_xyzuvw_many_larger = np.zeros((larger_ntimes, 6))
    bp_xyzuvw_then_larger = \
        tb.integrate_xyzuvw(bp_astr, np.array([0., larger_age]))[1]
    bp_xyzuvw_many_larger[0] = bp_xyzuvw_then_larger

    for i in range(1, larger_ntimes):
        bp_xyzuvw_many_larger[i] = tb.trace_forward(
            bp_xyzuvw_many_larger[i - 1], larger_tstep, solarmotion=None)

    print("For {} steps of size {:.4f} Myr".format(larger_ntimes, larger_tstep))
    print(bp_xyzuvw_now)
    print(bp_xyzuvw_many_larger[-1])

print(
    ' ------- Comparing multiple tracebacks with multiple traceforwards '
    '------- ')
times = np.linspace(0, age, ntimes)
stars = tb.stars_table(bp_astr)
bp_xyzuvws = tb.traceback(stars, times)[0]

print(np.allclose(bp_xyzuvw_many[::-1], bp_xyzuvws, rtol=1e-1))


print(
    ' ------- Comparing traceforward with traceback of negafied star '
    '------- '
)
bp_xyzuvw_then_cheating = bp_xyzuvws[-1].copy()
bp_xyzuvw_then_cheating[3:6] = -bp_xyzuvw_then_cheating[3:6]

bp_xyzuvw_then_cheating_sky = tb.xyzuvw_to_skycoord(
    bp_xyzuvw_then_cheating, solarmotion='schoenrich', reverse_x_sign=True
)
assert np.allclose(
    bp_xyzuvw_then_cheating,
    tb.integrate_xyzuvw(bp_xyzuvw_then_cheating_sky, np.array([0, 1e-3]))[0]
)

stars_cheating = tb.stars_table(bp_xyzuvw_then_cheating_sky)
bp_xyzuvws_cheating = tb.traceback(stars_cheating, times)[0,::-1]
#bp_xyzyvws_cheating = bp_xyzuvws_cheating[::-1].copy()
bp_xyzuvws_cheating[:,3:6] = -bp_xyzuvws_cheating[:,3:6]

assert np.allclose(
    bp_xyzuvws_cheating,
    bp_xyzuvws,
    #rtol=1e-2
    atol=1e-3
)

print(np.max(bp_xyzuvws_cheating - bp_xyzuvws))

#bp_xyzuvws_cheating = tb.traceback()


# ntimes = 3
# times = np.linspace(0, age, ntimes)
# tstep = float(age) / (ntimes - 1)
# print(times)
# bp_xyzuvws_from_past = np.zeros(6*ntimes)
# pdb.set_trace()
# bp_xyzuvws_from_past[0] = bp_xyzuvw_then
