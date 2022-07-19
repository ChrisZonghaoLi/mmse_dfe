import numpy as np
from scipy.linalg import toeplitz
from numpy import linalg

class DFE():
    '''
        This is the implementation of DFE
    
    '''
    
    def __init__(self, sampled_pulse_response, n_taps_dfe, samples_per_symbol):
        self.sampled_pulse_response = sampled_pulse_response
        self.n_taps_dfe = n_taps_dfe
        self.main_cursor = np.max(abs(self.sampled_pulse_response))    
        self.samples_per_symbol = samples_per_symbol
    
    def coefficients(self):
        '''

        Parameters
        ----------
        normalize_factor : float
            This is usually the absolute summation of FFE tap weights before its normalization

        Returns
        -------
        dfe_tap_weights : TYPE
            DESCRIPTION.

        '''
        dfe_tap_weights = self.sampled_pulse_response[1:]
        for i in range(len(dfe_tap_weights)):
            if dfe_tap_weights[i] > 0.5:
                dfe_tap_weights[i] = 0.5
            elif dfe_tap_weights[i] < -0.5:
                dfe_tap_weights[i] = -0.5
                
        return dfe_tap_weights
                
    def eqaulization(self, dfe_tap_weights, pulse_response):
        
        pulse_in = pulse_response

        pulse_out = np.copy(pulse_in)
        max_idx = np.argmax(abs(pulse_out)) 
        for i in range(1,self.n_taps_dfe+1):
            pulse_out[int(max_idx+self.samples_per_symbol*i-self.samples_per_symbol/2):int(max_idx+self.samples_per_symbol*i+self.samples_per_symbol/2)] -= dfe_tap_weights[i-1] 

        return pulse_out
                

class FFE():
    '''
        This is the implementation of FFE, including MMSE  to find optimal tap weights
    
    '''
    
    def __init__(self, sampled_pulse_response, n_taps_pre, n_taps_post, n_taps_dfe, samples_per_symbol):
        self.sampled_pulse_response = sampled_pulse_response
        self.n_taps_pre = n_taps_pre
        self.n_taps_post= n_taps_post
        self.n_taps_ffe = n_taps_pre + n_taps_post + 1
        self.n_taps_dfe = n_taps_dfe
        self.samples_per_symbol = samples_per_symbol
        self.channel_precursor = np.argmax(abs(sampled_pulse_response))
        self.channel_coefficients_len = len(sampled_pulse_response)
        self.idx_max = self.channel_precursor
        self.delay = self.n_taps_pre+self.channel_precursor
        
        self.c = np.zeros((self.channel_coefficients_len+self.n_taps_ffe-1,1))
        self.c[self.delay] = 1
        
        tmp = np.ones(self.channel_coefficients_len+self.n_taps_ffe-1)
        tmp[self.delay+1:self.delay+1+self.n_taps_dfe] = 0
        self.W = np.diag(tmp)
            
        # A: channel convolution matrix
        self.A = np.zeros((self.channel_coefficients_len+self.n_taps_ffe-1, self.n_taps_ffe))
        H = np.append(self.sampled_pulse_response, np.zeros(self.n_taps_ffe-1))
        for i in range(self.n_taps_ffe):
            self.A[:, i] = np.roll(H, i)
            
    def mmse(self, SNR, optimize_delay=False):
        # Autocorrelation matrix: A.T @ A
        # cross-correlation matrix: A.T @ c
        
        if optimize_delay==True:
            # sweep all the possible number of ffe precurosrs
            self.ummse = np.inf 
            for i in range(self.n_taps_ffe):
                delay = i+self.channel_precursor
                c = np.zeros((self.channel_coefficients_len+self.n_taps_ffe-1,1))
                c[delay] = 1
                
                tmp = np.ones(self.channel_coefficients_len+self.n_taps_ffe-1)
                tmp[delay+1:delay+1+self.n_taps_dfe] = 0
                W = np.diag(tmp)
        
                b = np.linalg.inv(self.A.T @ W @ self.A + np.eye(self.n_taps_ffe) * 10**(-(SNR/10))) @ (self.A.T @ c)
                
                # find unbiased MMSE
                t = self.A @ b
                b_unbiased = b/max(abs(t))
                t = np.squeeze(self.A @ b_unbiased)
                t[delay+1:delay+1+self.n_taps_dfe] = 0 # due to DFE post cursors are removed 
                # print(t)
                # print(c)
                ummse_new = linalg.norm(t - np.squeeze(c))**2 + 10**(-(SNR/10))*linalg.norm(np.squeeze(b_unbiased))**2
                if ummse_new < self.ummse:
                    self.ummse = ummse_new
                    self.delay = delay
                    self.c = c
                    self.W = W
                    self.n_taps_pre = i
                    self.n_taps_post = self.n_taps_ffe - i -1
                    
                    #normalize tap weights
                    b = b/np.sum(abs(b))
                    ffe_tap_weights = np.squeeze(b)
                    
        else:
            b = np.linalg.inv(self.A.T @ self.W @ self.A + np.eye(self.n_taps_ffe) * 10**(-(SNR/10))) @ (self.A.T @ self.c)
                    
            #normalize tap weights
            b = b/np.sum(abs(b))
    
            ffe_tap_weights = np.squeeze(b)
                
        return ffe_tap_weights
            
    def convolution(self, tap_weights, h):                
        tap_filter = np.zeros((self.n_taps_ffe-1)*self.samples_per_symbol+1)
        
        for i in range(self.n_taps_ffe):
            tap_filter[i*self.samples_per_symbol] = tap_weights[i]
            
        length = h.size
        h_out = np.convolve(h, tap_filter)
        h_out = h_out[self.n_taps_pre*self.samples_per_symbol:self.n_taps_pre*self.samples_per_symbol+length]
        
        return h_out
