from __future__ import print_function, division

from timeit import Timer
import argparse

t_gsl = Timer('result = overlap.get_overlap(A.flatten().tolist(),a.flatten().tolist(),A_det,B.flatten().tolist(),b.flatten().tolist(),B_det)','import overlap\nimport numpy as np\nA = np.random.random( (6,6) )\na = np.random.random(6)\nA_det = abs(np.linalg.det(A))\nB = np.random.random( (6,6) )\nb = np.random.random(6)\nB_det = abs(np.linalg.det(B))')

t_np = Timer('overlap = fit_group_reduced.compute_overlap(A,a,A_det,B,b,B_det)','import numpy as np\nimport fit_group_reduced\nA = np.random.random( (6,6) )\na = np.random.random(6)\nA_det = abs(np.linalg.det(A))\nB = np.random.random( (6,6) )\nb = np.random.random(6)\nB_det = abs(np.linalg.det(B))')

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--iterations', dest='iter', help='number of iterations')

args = parser.parse_args()
iter = int(args.iter)

print("GSL " + str(iter) + " times: {0:f}".format(t_gsl.timeit(iter)))
print("NumPy " + str(iter) + " times: {0:f}".format(t_np.timeit(iter)))
