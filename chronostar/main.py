import numpy as np
import overlap

def makeArray(instances, rows, cols):
    x = np.arange(instances*rows*cols)
    return x.reshape(instances,rows,cols)

print overlap.sum(makeArray(3,6,6))

myArray = np.arange(72, dtype=np.float).reshape(2,6,6)

print overlap.get_overlaps(myArray, 6)
