from emcee.utils import MPIPool
import time
import sys

using_mpi = True
try:
    pool = MPIPool()
    print("Successfully initialised mpi pool")
except ImportError:
    print("MPI not installed. Do this with (e.g.) pip install mpi4py,"
          "or unset flag in config file")
    raise UserWarning

print("This message should be printed once per thread")

if using_mpi:
    if not pool.is_master():
        print("One thread is going to sleep")
        pool.wait()
        sys.exit(0)

time.sleep(1)
print("\n\nonly one thread is master")

if using_mpi:
    pool.close()