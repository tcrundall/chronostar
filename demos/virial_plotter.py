from __future__ import division, print_function
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

MU = 1.05
SIG = np.sqrt(0.105)

def lognormal(x, mu=MU, sig=SIG):
    coeff = 1. / (x * sig * np.sqrt(2*np.pi))
    expon = - (np.log(x)-mu)**2 / (2*sig**2)
    return coeff * np.exp(expon)

xs = np.linspace(1e-2, 10, 1000)
ys = lognormal(xs)/np.max(lognormal(xs))
plt.clf()
exponents = [1., 0.5, 0.1]
if __name__ == '__main__':
    for exp in exponents:
        plt.plot(xs, ys**exp, label=str(exp))
    plt.legend(loc='best')
    plt.savefig("virial-plot.png")

