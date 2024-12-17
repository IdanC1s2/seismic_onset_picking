from matplotlib import pyplot as plt
import numpy as np
from scipy.fftpack import fft
import scipy
from scipy import signal
from pulse_detection import calc_tkeo


def plot_sig_in_time(sig, t_vec):
    """Plot the signal in time domain

    Args:
        sig : signal.
        t_vec : 1D array of the time values.

    Returns:
        fig: The generated figure.
    """

    fig = plt.figure()
    plt.plot(t_vec, sig)
    plt.xlabel('Time (s)')
    return fig

def plot_tkeo(sig, t_vec):
    """Plot the TKEO of a signal in time domain

    Args:
        sig : signal.
        t_vec : 1D array of the time values.

    Returns:
        fig: The generated figure.
    """
    tkeo = calc_tkeo(sig)
    t_vec = t_vec[1:-1]  # outer edges thrown by tkeo computation
    fig = plt.figure()
    plt.plot(t_vec, tkeo)
    plt.xlabel('Time (s)')
    return fig



def plot_sig_in_freq(sig, f_vec, n_samples):
    fig = plt.figure()
    signal_pow = np.abs((fft(sig)/n_samples))**2
    signal_pow_left = signal_pow[:len(f_vec)]
    plt.plot(f_vec, signal_pow_left)
    plt.xlabel('Frequency (Hz)')
    return fig


def seismic_wiggle(data_norm, t_vec, picked_indices=None, title=None):
    """Plots seismic wiggle traces for normalized sensor data with optional onset markings.

    Args:
        data_norm : Data matrix, normalized by the max value.
        t_vec (_type_): 1D array of the time values.
        picked_indices: Onset indices for each trace. 
        Can be a single index or multiple indices per trace.
        title: Title for the plot

    Returns:
        fig: The generated figure.
    """
    
    fig,ax = plt.subplots()
    n_sensors = data_norm.shape[0]
    tdiff = t_vec[1] - t_vec[0]
    
    for i in range(0, n_sensors):
        offset = i
        x = offset + data_norm[i,:]
        y = t_vec

        ax.plot(x,y,'k-')
        ax.fill_betweenx(y,offset,x,where=(x>offset),color='k')

        #  Plot picked indices if provided
        if picked_indices is not None:
            # In the case where in each trace we pick multiple onsets:
            if isinstance(picked_indices[i], np.ndarray):
                color_id = 0
                colors = ['b', 'g', 'r', 'm', 'y']
                for picked_idx in picked_indices[i]:
                    picekd_time = picked_idx * tdiff
                    plt.plot([offset - 0.5, offset + 0.5], [picekd_time, picekd_time],
                              colors[color_id]+'--')
                    color_id += 1
            else:
                picked_idx = picked_indices[i]
                picekd_time = picked_idx * tdiff
                plt.plot([offset - 0.5, offset + 0.5], [picekd_time, picekd_time], 'r--')

    plt.xlim([-1,n_sensors + 1])
    plt.ylim([t_vec[0], t_vec[-1]])
    
    if title:
        plt.title(title)

    plt.xlabel('Trace')
    plt.xlabel('Time (s)')
    plt.show()

    return fig

