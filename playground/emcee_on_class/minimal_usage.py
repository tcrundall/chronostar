import numpy as np
import modelfit as mf

if __name__ == '__main__':
    # Choose the "true" parameters.
    m_true = -0.9594
    b_true = 4.294

    # Generate some synthetic data from the model.
    N = 50
    x = np.sort(10 * np.random.rand(N))
    yerr = 0.1 + 0.5 * np.random.rand(N)
    y = m_true * x + b_true
    y += yerr * np.random.randn(N)

    test=mf.ModelFit(x=x, y=y, yerr=yerr)
    test.perform_run(1000)

    median_results = test.get_median()
