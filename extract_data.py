#! /usr/bin/env python
import re
import numpy as np
import matplotlib.pyplot as plt

infiles =  [
    'logs/0_0_141564_1_0_2000.txt', 
    'logs/1_0_141736_1_0_2000.txt', 
    'logs/2_0_141918_1_0_2000.txt', 
    'logs/3_0_142092_1_0_2000.txt', 
    'logs/4_0_142268_1_0_2000.txt', 
    'logs/5_0_142446_1_0_2000.txt', 
    'logs/6_0_142622_1_0_2000.txt', 
    'logs/7_0_142802_1_0_2000.txt', 
    'logs/8_0_142983_1_0_2000.txt', 
    'logs/9_0_143175_1_0_2000.txt', 
    'logs/10_0_143362_1_0_2000.txt', 
    'logs/11_0_143555_1_0_2000.txt', 
    'logs/12_0_143729_1_0_2000.txt', 
    'logs/13_0_143903_1_0_2000.txt', 
    'logs/14_0_144074_1_0_2000.txt', 
    'logs/15_0_144253_1_0_2000.txt', 
    'logs/16_0_144425_1_0_2000.txt', 
    'logs/17_0_144598_1_0_2000.txt', 
    'logs/18_0_144768_1_0_2000.txt', 
    'logs/19_0_144940_1_0_2000.txt', 
    'logs/20_0_145113_1_0_2000.txt', 
    'logs/21_0_145284_1_0_2000.txt', 
    'logs/22_0_145455_1_0_2000.txt', 
    'logs/23_0_145627_1_0_2000.txt', 
    'logs/24_0_145800_1_0_2000.txt', 
    'logs/25_0_145973_1_0_2000.txt', 
    'logs/26_0_146144_1_0_2000.txt', 
    'logs/27_0_146312_1_0_2000.txt', 
    'logs/28_0_146484_1_0_2000.txt', 
    ]

def get_data(infile, pattern):
    """
    Returns a tuple (median, pos_error, neg_error)
    """
    with open(infile, 'r') as fp:
        for l in fp:
            match = re.search(pattern, l)
            if match:
                return match.group(1,2,3)

# parameters = ['dX0', 'dY0', 'dZ0', 'dVel0', 'width']
parameters = ['dX0', 'dY0', 'dZ0', 'width']

# this complicated regex pattern extracts the median and errors
# from a log file

age_pat = 'logs/([0-9]*_[0-9]*)'
ages = np.zeros(len(infiles))

data = np.zeros((len(infiles), len(parameters),3))
for i, infile in enumerate(infiles):
    age_match = re.search(age_pat, infile)
    ages[i] = age_match.group(1).replace('_','.')
    for j, par in enumerate(parameters):
        pattern =\
            par +\
            '\s*:\s*(-*[0-9]*.[0-9]*)\s*\+\s*([0-9]*.[0-9]*)'+\
            '\s*-\s([0-9]*.[0-9]*)'

        data[i,j] = np.array(get_data(infile, pattern))

for i, par in enumerate(parameters):
    plt.errorbar(
        ages, data[:,i,0], yerr=[data[:,i,2], data[:,i,1]],
        label=par.replace('0',''), capsize=3, elinewidth=1)

plt.ylim(ymin=0)
plt.title("Standard Deviation in each axis")
plt.legend(loc='best')
plt.xlabel("Age [Myr]")
plt.ylabel("Standard Deviation [pc]")
plt.savefig('plots/BPMG_discrete_age_fit.jpg')
