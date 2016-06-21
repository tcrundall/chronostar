import numpy as np
import array3D 

def makeArray(instances, rows, cols):
    x = np.arange(instances*rows*cols)
    return x.reshape(instances,rows,cols)

print array3D.sum(makeArray(3,6,6))
