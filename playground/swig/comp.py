import overlap
import numpy as np
from timeit import Timer
A = np.random.random( (6,6) )

t_gsl = Timer('det = overlap.get_det(A.flatten().tolist())','import overlap\nimport numpy as np\nA = np.random.random( (6,6) )')
t_np = Timer('det = np.linalg.det(A)','import numpy as np\nA=np.random.random( (6,6) )')

print("GSL 100,000 times: {0:f}".format(t_gsl.timeit(100000)))
print("NumPy 100,000 times: {0:f}".format(t_np.timeit(100000)))

det = overlap.get_det(A.flatten().tolist())

print "(In Python) det is: " + str(det)
