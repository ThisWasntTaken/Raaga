import numpy as np
from math import ceil, floor
from scipy.signal import stft
from scipy.fft import dct
from Signal import AudioSignal
import matplotlib.pyplot as plt

def STFT(x, samplingRate = 1.0, windowLength = 0.05, windowEvery = 0.01, window = 'hann'):
    # Calculate everything in samples
    windowLengthSamples = windowLength * x.samplingRate
    windowEverySamples = ceil(windowEvery * x.samplingRate)
    adjustedWindowEvery = windowEverySamples / x.samplingRate

    # Length of y is going to be the number of windows in the stream
    # We pad the input signal to complete integral number of windows
    # Hence ceil of the division below
    ylength = 1 + ceil(x.length/windowEverySamples)

    # Width of y is half of the window length - nyquist frequency
    # is at the midpoing of the window length. +1 for the midpoint
    # of symmetry
    ywidth = 1 + floor(windowLengthSamples/2)

    y = np.ndarray(shape = (ywidth, ylength, x.channels), dtype=np.float32)

    for i in range(0, x.channels):
        f, t, Zxx = stft(
            x.audioData[:,i], 
            fs=samplingRate, 
            window=window, 
            nperseg=windowLengthSamples, 
            noverlap=windowLengthSamples-windowEverySamples, 
            nfft=None, 
            detrend=False, 
            return_onesided=True, 
            boundary='zeros', 
            padded=True, 
            axis=-1)
        y[:, :,i] = np.abs(Zxx)
    
    # Construct an audio signal which is 2 dimensional to represent the 
    # frequency in the second dimension
    ySignal = AudioSignal(y, 1/adjustedWindowEvery, ylength, x.channels, 2)
    ySignal.dimensionAxes.append(f * x.samplingRate)
    ySignal.time = t*x.samplingPeriod
    return ySignal
