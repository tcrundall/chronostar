import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb

files = [
    'lnprob_0_0_114471_1_0.pkl',
    'lnprob_1_0_114262_1_0.pkl',
    'lnprob_2_0_114304_1_0.pkl',
    'lnprob_3_0_114261_1_0.pkl',
    'lnprob_4_0_114261_1_0.pkl',
    'lnprob_5_0_114260_1_0.pkl',
    'lnprob_6_0_114258_1_0.pkl',
    'lnprob_7_0_114261_1_0.pkl',
    'lnprob_8_0_114261_1_0.pkl',
    'lnprob_9_0_115034_1_0.pkl',
    'lnprob_10_0_114302_1_0.pkl',
    'lnprob_11_0_114303_1_0.pkl',
    'lnprob_12_0_114303_1_0.pkl',
    ]
ntimes = 13
max_age=12
times = np.linspace(0,max_age,ntimes)
mean_lnprob = np.zeros(ntimes)
max_lnprob = np.zeros(ntimes)

for i in range(ntimes):
    lnprob_pars = pickle.load(open('results/'+files[i],'r'))
    mean_lnprob[i] = np.mean(lnprob_pars[0])
    max_lnprob[i]  = np.max(lnprob_pars[0])

plt.plot(times, mean_lnprob, '--', label='Mean')
plt.plot(times, max_lnprob,  label='Max')
plt.legend(loc='best')
plt.xlabel("Time [Myr]")
plt.ylabel("lnprob of fit")
plt.savefig("plots/twa_core_lnprobs.eps")
plt.show()
