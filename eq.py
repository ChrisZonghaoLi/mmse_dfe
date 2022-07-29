import numpy as np
from scipy.linalg import toeplitz
from numpy import linalg

# Matlab code (not the same but the skeletons are similar): http://bard.ece.cornell.edu/downloads/tutorials/fsedfe/fsedfe.html

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
            
    def mmse(self, SNR, signal_power, optimize_delay=True, zf=False):
        # Autocorrelation matrix: A.T @ A
        # cross-correlation matrix: A.T @ c
        
        if optimize_delay==True:
            # sweep all the possible number of ffe precurosrs
            self.unbiased_SNR = -np.inf 
            for i in range(self.n_taps_ffe):
                delay = i+self.channel_precursor
                c = np.zeros((self.channel_coefficients_len+self.n_taps_ffe-1,1))
                c[delay] = 1
                
                tmp = np.ones(self.channel_coefficients_len+self.n_taps_ffe-1)
                tmp[delay+1:delay+1+self.n_taps_dfe] = 0
                W = np.diag(tmp)
                
                if zf == False:
                    b = np.linalg.inv(self.A.T @ W @ self.A + np.eye(self.n_taps_ffe) * 10**(-(SNR/10))) @ (self.A.T @ c)
                else:
                    b = np.linalg.inv(self.A.T @ W @ self.A) @ (self.A.T @ c)

                # find  MMSE
                # cross-correlation
                Rxy = signal_power * (self.A.T @ c)
                # auto-correlation
                # Ryy = signal_power * (self.A.T @ self.A + np.eye(self.n_taps_ffe) * 10**(-(SNR/10)))
                
                if zf == False:
                    mmse = signal_power - b.T @ Rxy
                    unbiased_SNR = 10*np.log10(signal_power/mmse - 1)
                else:
                    mmse = (signal_power - b.T @ Rxy) + (signal_power * 10**(-(SNR/10)) * linalg.norm(np.squeeze(b))**2)
                    unbiased_SNR = 10*np.log10(signal_power/mmse)
                    
                # print(f'mmse: {mmse} | signal_power: {signal_power} |  b.T @ Rxy: {b.T @ Rxy} ')
                # print(f'unbiased_SNR: {unbiased_SNR}')
                
                if unbiased_SNR > self.unbiased_SNR:
                    self.unbiased_SNR = unbiased_SNR
                    self.delay = delay
                    self.c = c
                    self.W = W
                    self.n_taps_pre = i
                    self.n_taps_post = self.n_taps_ffe - i -1
                    
                    #normalize tap weights
                    b = b/np.sum(abs(b))
                    ffe_tap_weights = np.squeeze(b)       
        else:
            Rxy = signal_power * (self.A.T @ self.c)
            if zf == False:
                Ryy = signal_power * (self.A.T @ self.A) + np.eye(self.n_taps_ffe) * 10**(-(SNR/10))
                b = np.linalg.inv(self.A.T @ self.W @ self.A + np.eye(self.n_taps_ffe) * 10**(-(SNR/10))) @ (self.A.T @ self.c)
                mmse = signal_power - b.T @ Rxy
                self.unbiased_SNR = 10*np.log10(signal_power/mmse - 1)
            else:
                Ryy = signal_power * (self.A.T @ self.A)
                b = np.linalg.inv(self.A.T @ self.W @ self.A) @ (self.A.T @ self.c)        
                mmse = (signal_power - b.T @ Rxy) + (signal_power * 10**(-(SNR/10)) * linalg.norm(np.squeeze(b))**2)
                self.unbiased_SNR = 10*np.log10(signal_power/mmse)
                # print(self.unbiased_SNR )

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


def mmse_ffe_dfe(sampled_pulse_response, n_taps_ffe, n_taps_dfe, signal_power, noise_var, oversampling=1, delay=-1, zf=False):

    noise_var_scaler = np.copy(noise_var)
    noise_auto = np.append(np.array([noise_var]), np.zeros(n_taps_ffe-1))

    size = len(sampled_pulse_response)
    nu = int(np.ceil(size/oversampling) - 1) # channel memory so that it is FIR
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

    if type(delay) == int:
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
        if zf == False:
            Rn[:n_taps_ffe*oversampling, :n_taps_ffe*oversampling] = toeplitz(noise_auto)
        
        # desire output
        c = np.zeros((n_taps_ffe+nu, 1))
        c[d,:] = 1
        
        # MMSE
        Ry = P @ P.T * signal_power + Rn
        Rxy = P @ c * signal_power
        w_t_new = np.linalg.inv(Ry) @ Rxy
        
        # new SNR
        if zf == False:
            sigma_dfse = np.squeeze(signal_power - np.real(w_t_new.T @ Rxy))
            dfseSNR_new = 10*np.log10(signal_power/sigma_dfse-1)
        else:
            sigma_dfse = np.squeeze(signal_power - np.real(w_t_new.T @ Rxy)) + np.squeeze(noise_var_scaler * linalg.norm(np.squeeze(w_t_new)[:n_taps_ffe])**2)
            dfseSNR_new = 10*np.log10(signal_power/sigma_dfse)
        
        if dfseSNR_new >= dfseSNR:
            w_t = w_t_new
            dfseSNR = dfseSNR_new
            delay_opt = d
            n_taps_dfe_final = n_taps_dfe_used
            
    if n_taps_dfe_final < n_taps_dfe:
        print(f'For optimal DFE filter n_taps_dfe_final={n_taps_dfe_final} taps are used insteald of n_taps_dfe={n_taps_dfe} ')
    
    w_t = np.squeeze(w_t)
    
    return w_t, delay_opt, dfseSNR, n_taps_dfe_final


if __name__ == '__main__':
    # Example 3.7.2 from Prof. Cioffi's textbook, use this as a sanity check, two approaches should have the same tap weights and SNR
    
    zf = True
    
    sampled_pulse_response = np.array([0.9, 1])
    signal_power = 1#np.mean(abs(sampled_pulse_response**2))
    noise_var = 0.181
    n_taps_dfe=1
    n_taps_ffe=2
    SNR = 10*np.log10(5.524)
    delay = 1

    ffe = FFE(sampled_pulse_response, n_taps_pre=0, n_taps_post=1, n_taps_dfe=1, samples_per_symbol=1)
    tap_weights_ffe = ffe.mmse(SNR=SNR, signal_power=signal_power, optimize_delay=False, zf=zf)
    delay_opt = ffe.n_taps_pre
    unbiased_SNR = ffe.unbiased_SNR
    
    w_t, delay_opt_Cioffi, dfseSNR = mmse_ffe_dfe(sampled_pulse_response, n_taps_ffe, n_taps_dfe, signal_power, noise_var, oversampling=1, delay=delay, zf=zf)
    tap_weights_ffe_Cioffi = w_t[:n_taps_ffe]/sum(abs(w_t[:n_taps_ffe]))
