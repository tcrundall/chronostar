"""
Just a dummy profiler to debug cProfile usage
"""

import cProfile
import pstats
import numpy as np


def time_consumer():
    for i in range(100):
        for j in range(i):
            i * j

def demoFunction():
    total = 1
    for i in range(100):
        for j in range(100):
            total += i**j
            time_consumer()
    return

if __name__ == '__main__':
    stat_file = 'dummy.stat'
    cProfile.run('demoFunction()', stat_file)
    stat_obj = pstats.Stats(stat_file)
    stat_obj.print_stats()
