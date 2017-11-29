#! /usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import pdb

xs = np.linspace(0,10,41)
ys = xs**2

# plt.plot(xs,2*ys)
# plt.savefig("demo1.png")
# plt.show()
# pdb.set_trace()
# plt.clf()

fig, ax = plt.subplots()
ax.plot(xs,ys)
fig.savefig("demo2.png")
fig.show()


