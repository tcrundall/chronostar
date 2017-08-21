#! /usr/bin/env python
"""
Simple script demoing the use of expectmax module
"""
import chronostar.expectmax as em
import chronostar.analyser as an

print("Running fit")
samples, lnprob = em.run_fit('data/tb_synth_data_1groups_50stars.pkl', 200, 500)

print("Converting samples")
conv_samples = an.convert_samples(samples.reshape(-1,14), 1, 0, 14)
best_fit = an.calc_best_fit(conv_samples)

print("Job done")
