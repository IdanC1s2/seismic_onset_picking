import os
import numpy as np
import scipy
from matplotlib import pyplot as plt


def load_data(path):
    """A simple method used to load the dataset from a given path.

    Args:
        path : path to .mat file

    Returns:
        data, fs, geometry
    """
    matfile = scipy.io.loadmat(path)
    data = matfile.get('data')
    fs = matfile.get('fs')[0,0]
    geometry = matfile.get('geometry')
    return data, fs, geometry

def prep_baseline(data, fs ,extend_duration=3):
    """Given a data matrix, it prepares the data for addition of noise baseline.
        The preparation includes padding the left of each sensor with a specified
        amount of zeros and setting a new time vector and baseline index range.

    Args:
        data (_type_): data matrix.
        fs (_type_): sampling frequency.
        extend_duration (int, optional): Number of seconds used for padding.

    Returns:
        data_extended, t_vec_new, baseline_samples, baseline_indices
    """
    data_extended = extend_data_left(data, n_pad=extend_duration*fs)
    t_vec_new = np.arange(-extend_duration*fs, data.shape[1]) / fs
    baseline_samples = int(extend_duration*fs)
    baseline_indices = list(range(baseline_samples))
    
    return data_extended, t_vec_new, baseline_samples, baseline_indices


def add_noise(data, snr_dB, fs, noise_type='WGN'):
    """Given a data matrix, it adds noise the each channel according to the specified noise type 
    and SNR level.

    Args:
        data: Data matrix. (n_sensors, sig_len).
        snr_dB: Signal to Noise Ration (dB).
        fs: Sampling frequency (Hz).
        noise_type: Type of noise. Two options: 'WGN' or 'Pink.

    Returns:
        noisy: Noisy data matrix
    """

    n = data.shape[1]
    
    noisy = np.zeros_like(data)
    for ch in range(data.shape[0]):
        signal = data[ch, :]
        

        if noise_type == 'WGN':
            noise = np.random.randn(n)
        elif noise_type == 'Pink':
            noise = pink_noise(n)

        
        snr = 10**(snr_dB/10)  
        E_s = np.sum(signal**2)
        E_n = np.sum(noise**2)
        sigma = np.sqrt(E_s/(snr*E_n))  # gain coeff
        noisy[ch,:] = signal + sigma * noise

    return noisy

# To create pink noise:
def noise_psd(N, psd = lambda f: 1):
        X_white = np.fft.rfft(np.random.randn(N))
        S = psd(np.fft.rfftfreq(N))
        # Normalize S
        S = S / np.sqrt(np.mean(S**2))
        X_shaped = X_white * S
        return np.fft.irfft(X_shaped)

def PSDGenerator(f):
    return lambda N: noise_psd(N, f)

@PSDGenerator
def pink_noise(f):
    return 1/np.where(f == 0, float('inf'), np.sqrt(f))


def extend_data_left(data, n_pad=2000):
    "Pad the multichannel data matrix from left side using n_pad zeros."
    padded_data = np.pad(data, ((0, 0), (n_pad,0)), mode='constant', constant_values=0)
    return padded_data





