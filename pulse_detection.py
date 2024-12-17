import numpy as np
from scipy.fftpack import fft
import scipy
from scipy import signal



def calc_tkeo(signal):
    """ Calculate the Teager–Kaiser energy operator for a signal. """
    return signal[1:-1]**2 - signal[:-2] * signal[2:]



def find_earliest_occurrences_with_tkeo_impulse(data, fs, peak_thresh_dB=10):  # TODO - Document
    """
    Find the earliest occurrence time using TKEO and adaptive thresholding.

    Parameters:
    - data: 2 x N numpy array, signals measured by the two microphones.
    - fs: Sampling frequency (Hz). 
    - peak_thresh_dB: The ratio (dB) between the max peak and the beginning of a new occurrence.

    Returns:
    - times: List of earliest times for each sensor.
    - indices: List of indices corresponding to the earliest times.
    """
    times = []
    indices = []

    for mic_data in data:
        # Normalize the signal
        normalized_signal = mic_data / np.max(np.abs(mic_data))
        
        # Compute TKEO energy
        tkeo_energy = calc_tkeo(normalized_signal)
                
        peak_atten_ratio = 10 ** (peak_thresh_dB / 10)  
        tkeo_thresh = tkeo_energy.max() / peak_atten_ratio
        
        # Find the first index where energy exceeds the threshold
        occurrence_index = np.argmax(tkeo_energy > tkeo_thresh) + 1  # Offset for TKEO trimming
        occurrence_time = occurrence_index / fs
        
        times.append(occurrence_time)
        indices.append(occurrence_index)
    
    return times, indices





def find_all_occurrences_with_tkeo_impulse(data, fs, peak_thresh_dB=10):  # TODO - Document
    """
    Find *all* (not just initial) of the earliest occurrence times using TKEO thresholding.

    Parameters:
    - data: 2 x N numpy array, signals measured by the two microphones.
    - fs: Sampling frequency (Hz). 
    - peak_thresh_dB: The ratio (dB) between the max peak and the beginning of a new occurrence.

    Returns:
    - times: List of earliest times for each microphone [time_mic1, time_mic2].  # TODO
    - indices: List of indices corresponding to the earliest times.
    """
    times = {}
    indices = {}
    n_sensors = data.shape[0]

    for i in range(n_sensors):
        sig = data[i,:]
        # times[i] = []
        # indices[i] = []

        # Normalize the signal
        normalized_signal = sig / np.max(np.abs(sig))
        
        # Compute TKEO energy
        tkeo_energy = calc_tkeo(normalized_signal)
                
        
        peak_atten_ratio = 10 ** (peak_thresh_dB / 10)  
        tkeo_thresh = tkeo_energy.max() / peak_atten_ratio


        # Find the first index where energy exceeds the threshold
        occurrence_indices = np.argwhere(tkeo_energy > tkeo_thresh)[:,0]  # Offset for TKEO trimming
        occurrence_times = occurrence_indices / fs
        
        times[i] = occurrence_times
        indices[i] = occurrence_indices

        # plt.figure()
        # plt.plot(sig)
        
        # plt.figure()
        # plt.plot(tkeo_energy)
        # plt.axhline(tkeo_thresh, color='r', linestyle='--')
        # plt.scatter(occurrence_indices, 0.25 * np.ones_like(occurrence_indices))

        # peak_indices, matched_signal = detect_narrow_peaks_template_matching(tkeo_energy, gaussian_width=3, threshold=0.3)

        # plt.figure()
        # plt.plot(tkeo_energy)

    
    return times, indices


from scipy.signal import find_peaks, gaussian, convolve

def detect_narrow_peaks_template_matching(tkeo_signal, gaussian_width=5, threshold=0.5):
    """Unused Method"""
    # Step 1: Create a steep Gaussian template
    template_size = 2 * gaussian_width + 1  # Total size of the Gaussian template
    gaussian_template = gaussian(template_size, std=gaussian_width)
    gaussian_template /= np.sum(gaussian_template)  # Normalize the template to sum to 1
    
    # Step 2: Convolve the TKEO signal with the Gaussian template
    matched_signal = convolve(tkeo_signal, gaussian_template, mode='same')
    
    # Step 3: Find peaks in the matched signal that exceed the threshold
    peaks, _ = find_peaks(matched_signal, height=threshold)


    
    return peaks, matched_signal



###############################################################################################

# # Elbow method function: detect where the change falls below 5%
# def find_elbow(values, threshold=0.05):
#     changes = np.diff(values)  # Compute first derivative (changes between steps)
#     for i in range(1, len(changes)):
#         # Check if the new change is less than threshold * previous change
#         if abs(changes[i]) < threshold * abs(changes[i - 1]):
#             return i + 1  # Return the index (components count starts from 1)
#     return len(values)




