import numpy as np
class AudioSignal:
    def __init__(self, audioData = np.float32([]), samplingRate = 0, length = 0, channels = 1, dimensions = 1):
        self.audioData = audioData
        self.samplingRate = samplingRate
        self.samplingPeriod = 1/samplingRate
        self.channels = channels
        self.dimensions = dimensions
        self.length = length
        self.timeLength = length/samplingRate
        self.time = np.arange(0, self.timeLength, self.samplingPeriod, dtype=np.float32)
        self.dimensionAxes = []
