import numpy as np

def calc_new_mean(mu1, mu2):
    return 0.5*(mu1 + mu2)

def calc_new_sig(mu1, sig1, mu2, sig2):
    new_mu = calc_new_mean(mu1, mu2)
    return np.sqrt(0.25*(sig1**2 + sig2**2 + (mu1+mu2)**2) - new_mu**2)

def calc_new_sig2(mu1, sig1, mu2, sig2):
    new_mu = calc_new_mean(mu1, mu2)
    return np.sqrt(0.5*(sig1**2 + sig2**2 + mu1**2 + mu2**2) - new_mu**2)

def calc_new_sig3(sig1, sig2):
    return 0.5 * np.sqrt(sig1**2 + sig2**2)
