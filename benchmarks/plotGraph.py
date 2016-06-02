import matplotlib.pyplot as plt
import numpy as np

eig = [0.730823, 0.72486, 0.727342, 0.726392, 0.727197]

gsl = [0.007299, 0.007733, 0.007961, 0.007401, 0.008027]

eig_std = np.std(eig)
gsl_std = np.std(gsl)

eig_mean = np.mean(eig)
gsl_mean = np.mean(gsl)

plt.bar([0, 1],[eig_mean, gsl_mean], yerr=[eig_std, gsl_std])
plt.xlabel("Eigen                                       GSL")
plt.ylabel("Time (s)")
plt.title("Finding Determinant of 10,000 6x6 Matrices")

plt.show()
