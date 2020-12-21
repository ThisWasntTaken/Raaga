import numpy as np
from scipy.signal import stft
from scipy.fft import dct
from MP3 import AudioSignal
import matplotlib.pyplot as plt

from python_speech_features import mfcc

def STFT(x, samplingRate = 1.0, wlength = 2400, woverlap = 2272, window = 'hann'):
    ylength = 1+int((x.length+woverlap-1)/woverlap)
    y = np.ndarray(shape = (woverlap+1, ylength, x.channels), dtype=np.float32)
    for i in range(0, x.channels):
        #t, f, y[:, :,i] = stft(
        f, t, Zxx = stft(
            x.audioData[:,i], 
            fs=samplingRate, 
            window=window, 
            nperseg=wlength, 
            noverlap=woverlap, 
            nfft=None, 
            detrend=False, 
            return_onesided=True, 
            boundary='zeros', 
            padded=True, 
            axis=-1)
        #plt.pcolormesh(f, t, np.abs(Zxx), vmin=0, vmax=1.0, shading='gouraud')

        winstep = 0.01
        mfcc_feat = mfcc(x.audioData[:,i],x.samplingRate, winstep=winstep)
        t = np.arange(0, x.length - x.samplingRate*winstep - 1, x.samplingRate*winstep)
        f = np.arange(0, 13, 1)
        
        #plt.pcolormesh(t, f*x.samplingRate, np.abs(Zxx))
        #plt.pcolormesh(t, f*x.samplingRate, np.abs(mfcc_feat))
        #plt.pcolormesh(t, f, np.abs(np.transpose(mfcc_feat)))
        for j in range(0,13):
            plt.scatter(t,mfcc_feat[:,j])
            plt.title('STFT Magnitude')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.show()

        a = 0

    return AudioSignal(y, x.samplingRate, ylength, x.channels)

def DCT(x, samplingRate = 1.0, wlength = 256, woverlap = 128):
    return dct(
        x.audioData[:,0], 
        fs=samplingRate, 
        window='hann', 
        nperseg=wlength, 
        noverlap=woverlap, 
        nfft=None, 
        detrend=False, 
        return_onesided=True, 
        boundary='zeros', 
        padded=True, 
        axis=0)