import overlap
import numpy as np

x = np.array([1,2,3])
y = np.array([2,3,4])

print overlap.flatten(x,y)
print overlap.flatten2(x,x)

print overlap.range(10)

print overlap.flatten3(x,x,5)

myArray = np.arange(72, dtype=np.float)
myArray = myArray.reshape(2, 6, 6)

print overlap.sum(myArray)
print overlap.sum2(myArray,5)
