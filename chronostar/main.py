import numpy as np
import overlap

def makeArray(instances, rows, cols):
    x = np.arange(instances*rows*cols)
    return x.reshape(instances,rows,cols)

print overlap.sum(makeArray(3,6,6))
