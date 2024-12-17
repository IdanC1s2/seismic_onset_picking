import numpy as np
from matplotlib import pyplot as plt

def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=1):
    ''' Unused Method
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    # n = sig.shape[0] + refsig.shape[0] - 1
    N = sig.shape[0]
    Ncorr = 2 * N - 1
    nextpow2 = int(np.ceil(np.log2(Ncorr)))
    nfft = 2**nextpow2
    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=nfft)
    REFSIG = np.fft.rfft(refsig, n=nfft)
    R = SIG * np.conj(REFSIG)

    # cc = np.fft.irfft(R / np.abs(R), n=(interp * nfft))
    cc = np.fft.irfft(np.exp(1j * np.angle(R)));
    cc = np.fft.fftshift(cc)
    # r12_temp = np.fft.irfft(np.exp(1j * np.angle(R)));

    # max_shift = int(interp * nfft / 2)

    # max_shift = n
    # if max_tau:
    #     max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    # cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
    # lags = np.arange(-max_shift,max_shift+1)

    lags = np.arange(-(Ncorr - 1) / 2,(Ncorr-1) / 2 + 1)
    # lags = lags / fs

    cc = cc[nfft//2 - (Ncorr-1)//2 : nfft//2 + (Ncorr-1)//2 + 1]
    # nfft//2 is the middle point

    # find max cross correlation index
    # shift = np.argmax(np.abs(cc)) - max_shift

    # tau = shift / float(interp * fs)

    idx = np.argmax(np.abs(cc))
    tau = lags[idx]

    return tau, cc, lags   # returns tau as number of samples to shift


def gcc_phat_v2(sig, refsig, fs=1, max_tau=None, interp=16):
    ''' Unused Method
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''
    interp = 1  # TODO remove
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)


    # Phase transform into correlation:
    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))  # TODO
    # cc = np.fft.irfft(R, n=(interp * n))
    # cc = np.fft.irfft(R / (np.abs(R) + 10), n=(interp * n))
    # cc = np.fft.irfft(R / np.sqrt(np.abs(R)), n=(interp * n))
    # cc = np.fft.fftshift(np.fft.ifft(exp(1i*angle(R12))),1)
    # cc = np.fft.ifft(np.exp(1j*np.angle(R)))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(np.ceil(interp * fs * max_tau)), max_shift)

    cc_shifted = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc_shifted)) - max_shift

    tau = shift / float(interp * fs)

    # fig, axes = plt.subplots(4,1)
    # axes[0].plot(refsig, 'bs-')
    # axes[0].plot(sig, 'r^-')
    # axes[0].legend(['refsig', 'sig'])

    # axes[1].plot(cc, 'rs-')
    # axes[2].plot(cc_shifted, 'rs-')

    # axes[3].plot(np.abs(R))


    return tau, cc_shifted



def parbolic_interpolation(cc, lags, fs=None):
    """Unused Method"""
    l0 = np.argmax(cc)
    R = np.array([cc[l0-1],cc[l0],cc[l0+1]])
    L = np.array([[lags[l0-1]**2,lags[l0-1],1],[lags[l0]**2,lags[l0],1],[lags[l0+1]**2,lags[l0+1],1]])
    A = np.linalg.inv(L) @ R

    tdoa_interp = -A[1] / 2 / A[0]

    return tdoa_interp



def calculate_lag_and_align(sig, refsig, fs, do_visualize=False):
    """
    Calculate the lag between two signals, remove the lag, and plot the aligned signals.

    Parameters:
    - sig: 1D numpy array, first signal.
    - refsig: 1D numpy array, reference signal.
    - fs: Sampling frequency (Hz).

    Returns:
    - aligned_sig: Signal 1 aligned with the reference signal.
    - aligned_refsig: The reference signal aligned with Signal 1 (shortened for alignment).
    - lag_samples: Lag between the signals in samples.
    - lag_time: Lag between the signals in seconds.
    """
    # Compute cross-correlation
    correlation = np.correlate(sig, refsig, mode='full')
    lag_samples = np.argmax(correlation) - (len(refsig) - 1)
    lag_time = lag_samples / fs  # Convert lag to time
    
    # Shift sig1 to align with sig2
    if lag_samples > 0:
        aligned_sig = sig[lag_samples:]  # Shift sig1 forward (to left)
        aligned_refsig = refsig[:len(aligned_sig)]  # Truncate sig2 to match length
    else:
        aligned_sig = sig[:len(sig) + lag_samples]  # Truncate the end of sig1
        aligned_refsig = refsig[-lag_samples:]  # Shift sig2 forward to match


    # If lag>0 it means that sig lags after refsig by 'lag' time points.
    # Therefore, if sig has onset in some t0, in order to go to the reference time axis,
    # we simply convert it to t0-lag .

    
    # Plot the results
    if do_visualize:
        plt.figure(figsize=(10, 6))
        
        # Before alignment
        plt.subplot(2, 1, 1)
        plt.plot(sig, label='Signal 1')
        plt.plot(refsig, label='Signal 2')
        plt.title('Before Alignment')
        plt.legend()
        
        # After alignment
        plt.subplot(2, 1, 2)
        plt.plot(aligned_sig, label='Aligned Signal 1')
        plt.plot(aligned_refsig, label='Aligned Signal 2')
        plt.title('After Alignment')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    return aligned_sig, aligned_refsig, lag_samples, lag_time
