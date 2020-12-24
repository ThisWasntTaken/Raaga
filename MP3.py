import os
import pathlib
import numpy as np
import pydub
from Signal import AudioSignal

def read(f, normalized = False):
    # Read MP3 to numpy array
    # Fix pydub ffmpeg file paths : https://stackoverflow.com/questions/51219531/pydub-unable-to-locte-ffprobe
    pydub.AudioSegment.converter = "W:\\Tools\\ffmpeg\\ffmpeg.exe"
    pydub.AudioSegment.ffmpeg = "W:\\Tools\\ffmpeg\\ffmpeg.exe"
    pydub.AudioSegment.ffprobe = "W:\\Tools\\ffmpeg\\ffprobe.exe"
    audioSegment = pydub.AudioSegment.from_mp3(f)
    audioData = np.array(audioSegment.get_array_of_samples())
    if audioSegment.channels == 2:
        audioData = audioData.reshape((-1,2)) # Unknown length - so resize as required for 2 columns
        audioShape = audioData.shape
    elif audioSegment.channels == 1:
        audioData = audioData.reshape((-1,1)) # Unknown length - so resize as required for 1 column
        audioShape = (audioData.shape[0], 1)
    if normalized:
        return AudioSignal(np.float32(audioData) / 2**15, audioSegment.frame_rate, audioShape[0], audioShape[1], 1)
    else:
        return AudioSignal(np.float32(audioData), audioSegment.frame_rate, audioShape[0], audioShape[1], 1)
