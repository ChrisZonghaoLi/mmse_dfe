import numpy as np
from scipy.linalg import toeplitz
from numpy import linalg

# This is just an alternative, based on Prof.Cioffi's lecture notes from EE379A (page 222 on https://studylib.net/doc/18685925/ch03)

def mmse_ffe_dfe(sampled_pulse_response, n_taps_ffe, n_taps_dfe, signal_power, noise_var, oversampling=1, delay=-1):

    noise_var = np.append(np.array([noise_var]), np.zeros(n_taps_ffe-1))

    size = len(sampled_pulse_response)
    nu = int(np.ceil(size/oversampling) - 1) # channel memory, if it is limited it should be FIR response
    sampled_pulse_response = np.append(sampled_pulse_response, np.zeros((nu+1)*oversampling-size))
    
    # error check
    if n_taps_ffe <= 0:
        print(f'{n_taps_ffe} should be >0')
    if n_taps_dfe <0:
        print(f'{n_taps_dfe} should be >=0')
    if delay > n_taps_ffe+nu -1:
        print(f'delay must be <= {n_taps_ffe}+{len(sampled_pulse_response)}-2')
    if delay < -1:
        print('delay must be >= -1')
    if delay == -1:
        print('optimal delay will be searched')
        delay = [i for i in range(n_taps_ffe+nu)]

    if type(delay) == int and delay !=1:
        delay = [delay]
    
    # do some oversampling
    sampled_pulse_response_temp = np.zeros((oversampling, nu+1))
    sampled_pulse_response_temp[:oversampling,0] = np.append(sampled_pulse_response[0], np.zeros(oversampling-1))
    for i in range(nu):
        sampled_pulse_response_temp[:oversampling, i+1] = np.conj(np.flip(sampled_pulse_response[i*oversampling+1:i*oversampling+2]).T)
        
    dfseSNR = -100
    
    for d in delay:
        n_taps_dfe_used = min(n_taps_ffe+nu-1-d, n_taps_dfe)
        
        # generate channel pulse matrix
        P = np.zeros((n_taps_ffe*oversampling+n_taps_dfe_used, n_taps_ffe+nu))
        for i in range(n_taps_ffe):
            P[i*oversampling:i*oversampling+1, i:(i+nu+1)] = sampled_pulse_response_temp
        P[n_taps_ffe*oversampling:n_taps_ffe*oversampling+n_taps_dfe_used, d+1:d+n_taps_dfe_used+1] = np.eye(n_taps_dfe_used)
        
        # compute Rn, noise autocorrelation matrix
        Rn = np.zeros((n_taps_ffe*oversampling+n_taps_dfe_used, n_taps_ffe*oversampling+n_taps_dfe_used))
        Rn[:n_taps_ffe*oversampling, :n_taps_ffe*oversampling] = toeplitz(noise_var)
        
        # desire output
        c = np.zeros((n_taps_ffe+nu, 1))
        c[d,:] = 1
        
        # MMSE
        Ry = P @ P.T * signal_power + Rn
        Rxy = P @ c * signal_power
        w_t_new = np.linalg.inv(Ry) @ Rxy
    
        # new SNR
        sigma_dfse = np.squeeze(signal_power - np.real(w_t_new.T @ Rxy))
        dfseSNR_new = 10*np.log10(signal_power/sigma_dfse-1)
        # print(f'dfseSNR_new: {dfseSNR_new}')
        
        if dfseSNR_new >= dfseSNR:
            w_t = w_t_new
            dfseSNR = dfseSNR_new
            delay_opt = d
            n_taps_dfe_final = n_taps_dfe_used
            
    if n_taps_dfe_final < n_taps_dfe:
        print(f'For optimal DFE filter n_taps_dfe_final={n_taps_dfe_final} taps are used insteald of n_taps_dfe={n_taps_dfe} ')
    
    w_t = np.squeeze(w_t)
    
    return w_t, delay_opt, dfseSNR


if __name__ == '__main__':
    sampled_pulse_response = np.array([0.807, 0.215, 0.207, 0.314, 0.111])
    signal_power = 0.901#np.mean(abs(sampled_pulse_response**2))
    noise_var = 0.0226
    n_taps_dfe=5
    n_taps_ffe=5
    delay=-1
    w_t, delay_opt, dfseSNR = mmse_ffe_dfe(sampled_pulse_response, n_taps_ffe, n_taps_dfe, signal_power, noise_var, oversampling=1, delay=-1)
    tap_weights_ffe = w_t[:n_taps_ffe]/sum(abs(w_t[:n_taps_ffe]))
    tap_weights_dfe = w_t[n_taps_ffe:n_taps_ffe+n_taps_dfe]
    
