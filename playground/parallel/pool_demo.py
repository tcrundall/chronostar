import timeit
from multiprocessing import Pool
import numpy as np
import time

def f(x):
    return x*x

def factorial(x):
    if x == 0 or x == 1:
        return 1
    else:
        return x * factorial(x-1)

def serial(xs):
    map(factorial, xs)

def parallel(xs):
    p = Pool(4)
    p.map(factorial, xs)

def super_parallel(xs):
    p = Pool(2)
    p.map(factorial, xs)

if __name__ == '__main__':
    xs = [400] * 30000
    timer1 = timeit.Timer('serial(xs)','from __main__ import serial, xs')
    timer2 = timeit.Timer('parallel(xs)','from __main__ import parallel, xs')
    timer3 = timeit.Timer('super_parallel(xs)','from __main__ import super_parallel, xs')

    print(timer1.timeit(1))
    print(timer2.timeit(1))
    print(timer3.timeit(1))

    """
    start = time.clock()
    result = p.map(factorial, x)
    print(len(result))
    end = time.clock()
    print(end - start)

    p = Pool(4)
    start = time.clock()
    result = p.map(factorial, x)
    print(len(result))
    end   = time.clock()
    print(end - start)
    """
