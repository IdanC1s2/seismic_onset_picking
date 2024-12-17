import numpy as np
from scipy.signal import stft


def estimate_noise_floor(noise_segment, fs, window, nperseg, noverlap):
    """This function is used to estimate the noise-floor level of a signal, 
    based on a segment that is filled only with noise.

    Args:
        noise_segment: List of indices filled only with noise
        fs: sampling frequency (Hz)
        window: window type
        nperseg: number of samples per window (window size)
        noverlap: number of overlapping samples between consecutive windows.

    Returns:
        noise_floor : A vector in the fourier domain representing the average energy level
        of each noise frequency.
    """
    # Compute the STFT of the noise floor segment
    _, _, Zxx_noise = stft(noise_segment, fs=fs,window=window, nperseg=nperseg, noverlap=noverlap)

    # Average over the time axis
    noise_floor = np.mean(np.abs(Zxx_noise)**2, axis=1)

    return noise_floor

def compute_fft_energy(sig):
    """Compute the max energy in the FFT domain.
    Args:
        sig: Signal (time domain)

    Returns:
        max_energy
    """
    
    fft_result = np.fft.fft(sig)
    energy = np.abs(fft_result/len(sig)) ** 2  # Energy
    max_energy = np.max(energy)
    return max_energy


def find_earliest_occurrences_stft_energy_thresh(sig,fs, baseline_indices, window='hann',
                                                 window_sz=128, frame_jump=16, peak_thresh_dB=30,
                                                 noise_floor_factor=20):
    """ This function detects the earliest occurrence in a signal using stft energy thresholding.
    It starts by estimating the noise-floor level using a range of baseline in the signal. Then, if
    a frequency dominates more that noise_floor_factor * its noise_floor_level,
    then it might be an occurrence. 
    A peak_threshold is also used to make sure we don't pick too early in cases where there is no
    noise in the system.


    Args:
        sig: A signal (single channel)
        fs: Sampling frequency (Hz)
        baseline_indices (list): indeices range of the signal that is known as baseline and is only
        used for noise-floor estimation. 
        window: Window type. Default is 'hann'.
        window_sz: Window size. Default is 128.
        frame_jump: The jump between consecutive stft frames The smaller it is, the higher resolution
        we can get in time for an occurrence, but the more computational load it creates.
        Default is 16.
        peak_thresh_dB: The ratio a from the biggest peak that a frequency bin should surpass
        in order to be an occurrence. Default is 30 dB.
        noise_floor_factor: The number of time a frequency has to be higher than its
        noise-floor level in order to create an occurrence. Default is 20.

    Returns:
        onset_bin_idx, onset_time
    """
    
    baseline_samples = len(baseline_indices)
    window_sz = 128  # Window length for STFT
    frame_jump = 16  # samples jump from one frame to the next frame
    noverlap = window_sz - frame_jump # Overlap between segments
    
    # Number of stft time-bins used for noise floor estimation
    n_baseline_bins = (baseline_samples - window_sz) // noverlap + 1  

    # Estimate the noise floor from the first samples of the signal
    noise_floor_segment = sig[:baseline_samples]
    noise_floor = estimate_noise_floor(noise_floor_segment, fs, window, window_sz, noverlap)

    # Find maximum energy level of the entire fft:
    max_fft_energy = compute_fft_energy(sig)
    peak_atten_ratio = 10 ** (peak_thresh_dB/10)

    # The energy_threshold is a fraction of the maximum fft energy
    energy_threshold = max_fft_energy / peak_atten_ratio

    # Timeshift to avoid the baseline part later.
    timeshift = baseline_samples / fs

    
    # Compute STFT
    f, t, Zxx = stft(sig, fs=fs, nperseg=window_sz, noverlap=noverlap)
    stft_energies = (np.abs(Zxx) ** 2) 
    stft_energies_per_samp = (np.abs(Zxx/window_sz) ** 2) 

    # Shift the baseline part to negative times.
    t = t - timeshift

    for i in range(n_baseline_bins, Zxx.shape[1]):

            # Check when any frequency component exceeds its noise floor by a significant factor

            # A frequency is dominant if it is more than factor times the estimated noise floor.
            # Also - the same frequency's energy_per_sample in the stft should be larger than the
            # threshold defined by the maximum energy_per_sample in the fft.
            if np.any(stft_energies[:,i] > noise_floor * noise_floor_factor) and \
            np.any(stft_energies_per_samp[:,i] > energy_threshold):  # TODO Works sometimes... probably energy_threshold needs to be larger
                            
                onset_time = t[i]
                onset_bin_idx = i
                # print('entered')
                break
    
    return onset_bin_idx, onset_time



def early_occurences_stft_multichannel(data, fs, baseline_indices, window='hann', window_sz=128, frame_jump=16, peak_thresh_dB=60, noise_floor_factor=20):
    """ Read the documentation for the previous function. This function only extends it to all of the
    data matrix channels"""
    onset_times = []
    onset_indices = []
    n_sensors = data.shape[0]


    for i in range(n_sensors):
        sig = data[i,:]
        onset_bin_idx, onset_time = find_earliest_occurrences_stft_energy_thresh(sig, fs, baseline_indices, window,
                                                        window_sz , frame_jump, peak_thresh_dB, noise_floor_factor)
        
        onset_idx_original = int(onset_time * fs)

        onset_times.append(onset_time)
        onset_indices.append(onset_idx_original)

    return onset_times, onset_indices




def find_all_occurrences_stft_energy_thresh(sig,fs, baseline_indices, window='hann',
                                                 window_sz=128, frame_jump=16, peak_thresh_dB=30,
                                                 noise_floor_factor=20):
    
    """ This function, similarly to the function find_earliest_occurrences_stft_energy_thresh(),
    detects occurrences in a signal using stft energy thresholding.
    The difference is that this function detects the entire occurrence and doesn't stop once 
    its beginning has been detected!
    It starts by estimating the noise-floor level using a range of baseline in the signal. Then, if
    a frequency dominates more that noise_floor_factor * its noise_floor_level,
    then it might be an occurrence. 
    A peak_threshold is also used to make sure we don't pick too early in cases where there is no
    noise in the system.


    Args:
        sig: A signal (single channel)
        fs: Sampling frequency (Hz)
        baseline_indices (list): indeices range of the signal that is known as baseline and is only
        used for noise-floor estimation. 
        window: Window type. Default is 'hann'.
        window_sz: Window size. Default is 128.
        frame_jump: The jump between consecutive stft frames The smaller it is, the higher resolution
        we can get in time for an occurrence, but the more computational load it creates.
        Default is 16.
        peak_thresh_dB: The ratio a from the biggest peak that a frequency bin should surpass
        in order to be an occurrence. Default is 30 dB.
        noise_floor_factor: The number of time a frequency has to be higher than its
        noise-floor level in order to create an occurrence. Default is 20.

    Returns:
        onset_bin_idx (list), onset_time (list) 
    """

    baseline_samples = len(baseline_indices)
    window_sz = 128  # Window length for STFT
    frame_jump = 16  # samples jump from one frame to the next frame
    noverlap = window_sz - frame_jump # Overlap between segments
    
    # Number of stft time-bins used for noise floor estimation
    n_baseline_bins = (baseline_samples - window_sz) // noverlap + 1  

    # Estimate the noise floor from the first samples of the signal
    noise_floor_segment = sig[:baseline_samples]
    noise_floor = estimate_noise_floor(noise_floor_segment, fs, window, window_sz, noverlap)

    # Find maximum energy level of the entire fft:
    max_fft_energy = compute_fft_energy(sig)
    peak_atten_ratio = 10 ** (peak_thresh_dB/10)

    # The energy_threshold is a fraction of the maximum fft energy
    energy_threshold = max_fft_energy / peak_atten_ratio

    # Timeshift to avoid the baseline part later.
    timeshift = baseline_samples / fs

    
    # Compute STFT
    f, t, Zxx = stft(sig, fs=fs, nperseg=window_sz, noverlap=noverlap)
    stft_energies = (np.abs(Zxx) ** 2) 
    stft_energies_per_samp = (np.abs(Zxx/window_sz) ** 2) 

    # Shift the baseline part to negative times.
    t = t - timeshift
    time_bin_occurence = []
    time_sample_occurence = set()
    for i in range(n_baseline_bins, Zxx.shape[1]):

            # Check when any frequency component exceeds its noise floor by a significant factor

            # A frequency is dominant if it is more than factor times the estimated noise floor.
            # Also - the same frequency's energy_per_sample in the stft should be larger than the
            # threshold defined by the maximum energy_per_sample in the fft.
            if np.any(stft_energies[:,i] > noise_floor * noise_floor_factor) and \
            np.any(stft_energies_per_samp[:,i] > energy_threshold):  # TODO Works sometimes... probably energy_threshold needs to be larger
                            
                start_index = i * frame_jump
                end_index = start_index + window_sz - 1

                time_sample_occurence.update(list(range(start_index, end_index+1)))
                time_bin_occurence.append(i)

    time_sample_occurence = np.array(list(time_sample_occurence))

    return time_bin_occurence, time_sample_occurence

    


def all_early_occurences_stft_multichannel(data, fs, baseline_indices, window='hann', window_sz=128, frame_jump=16, peak_thresh_dB=60, noise_floor_factor=20):
    """ Read the documentation for the previous function. This function only extends it to all of the
    data matrix channels"""
    occurence_times = {}
    occurence_indices = {}
    n_sensors = data.shape[0]


    for i in range(n_sensors):
        sig = data[i,:]
        time_bin_occurence, time_sample_occurence = find_all_occurrences_stft_energy_thresh(sig, fs, baseline_indices, window,
                                                        window_sz , frame_jump, peak_thresh_dB, noise_floor_factor)
        

        occurence_indices[i] = time_sample_occurence
        occurence_times[i] = time_sample_occurence / fs

    return occurence_times, occurence_indices



