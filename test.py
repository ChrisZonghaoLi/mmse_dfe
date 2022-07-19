import numpy as np
from eq import *

with open('channel_pulse_response_test.csv') as file_name:
    sampled_pulse_response = np.loadtxt(file_name, delimiter=",")

SNR = 35
n_taps_dfe=2
n_taps_pre = 1
n_taps_post = 3
n_taps_ffe = n_taps_pre + n_taps_post + 1
samples_per_symbol = 8

# invoke FFE and MMSE
ffe = FFE(sampled_pulse_response, n_taps_pre, n_taps_post, n_taps_dfe, samples_per_symbol)
tap_weights_ffe = ffe.mmse(SNR=SNR, optimize_delay=True)
t = np.convolve(tap_weights_ffe, sampled_pulse_response)

# DFE, assume it is noiseless
dfe = DFE(t, n_taps_dfe, samples_per_symbol)
tap_weights_dfe = dfe.coefficients()
pulse_response_ffe_dfe = dfe.eqaulization(tap_weights_dfe, pulse_response_ffe)
