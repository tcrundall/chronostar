from __future__ import division, print_function
from threading import Thread, Lock
import numpy as np
import scipy.linalg as la

class aa(object):
    def __init__(self, i):
        self.i=i

    def f(self, l, myarr):
        myarr[self.i]=self.i
        myarr[self.i+1]=self.i
        l.acquire()
        print('hello world {0:d}'.format(self.i))
        l.release()

nbig = 1000
niter = 20
global_vect = np.random.random(nbig)

def do_lots(big_matrix):
    for i in range(niter):
        #aa = la.inv(big_matrix)
        aa = np.linalg.inv(big_matrix)
        retval = np.dot(np.dot(global_vect, aa),global_vect)
    print('my retval {}'.format(retval))

if __name__ == '__main__':
    myarr = [0,0,0,0,0,0,0,0,0,0,0]
    lock = Lock()
    ps = []
    objs = []

    for num in range(10):
        objs.append(aa(num))
        ps.append(Thread(target=objs[-1].f, args=(lock, myarr)))
        ps[-1].start()
        
    for num in range(10):
        ps[num].join()
        
    print(myarr)
    
    #Matrix stuff
    ps = []
    nthreads=15
    random_mat = np.random.random( (nthreads,nbig,nbig) )
    for num in range(nthreads):
        ps.append(Thread(target=do_lots, args=(random_mat[num],) ))
        ps[-1].start()
    
    for num in range(nthreads):
        ps[num].join()