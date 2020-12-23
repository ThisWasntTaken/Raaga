import sys
import os
import argparse
import numpy as np
from bokeh.plotting import figure as bkfigure
from bokeh.io import show as bkshow
from bokeh.layouts import column as bkcolumn
from bokeh.models.mappers import LinearColorMapper
from bokeh.palettes import Inferno11
import bokeh
from Signal import AudioSignal
import MP3
import Transforms

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Read an mp3 file and plot out its pitch')
    parser.add_argument('-i', dest='input', type=str, help='Input file path', default='')
    parser.add_argument('-n', dest='normalize', action='store_true', help='Normalize input values', default='')
    parser.add_argument('-t', dest='transform', type=str, help='Use different transforms on the input audio signal', default='stft')
    parser.add_argument('-s', dest='sample', type=int, help='Sampling rate in Hz to use on the input audio signal while transforming', default='100')
    parser.add_argument('-w', dest='window', type=float, help='Sampling window in s to use on the input audio signal for stft', default='0.05')
    options = parser.parse_args()

    # Error check
    if options.input == '':
        print("No input given. BYE!\n")
        return 1
    elif not os.path.isfile(options.input):
        print (f"Given input path {options.input} does not exist!")
        return 2

    # Read input file into frame rate and data
    try:
        inSignal = MP3.read(options.input, options.normalize)
    except:
        print ("Reading MP3 failed")
        return 3

    figures = []
    # Plot the data for quick visualization
    if options.transform == 'none':
        for i in range(0, inSignal.channels):
            if i == 0:
                figures.append(bkfigure(plot_width = 1200, plot_height = 600, x_axis_label = 'Time', y_axis_label = 'Amp'))
            else:
                figures.append(bkfigure(plot_width = 1200, plot_height = 600, x_axis_label = 'Time', y_axis_label = 'Amp', x_range=figures[0].x_range, y_range=figures[0].y_range))
            figures[i].line(inSignal.time, inSignal.audioData[:,i])
    elif options.transform == 'stft':
        # STFT over the signal
        fSignal = Transforms.STFT(inSignal, windowEvery = 1/options.sample, windowLength = options.window)
        for i in range(0, inSignal.channels):
            if i == 0:
                figures.append(bkfigure(plot_width = 1200, plot_height = 400, x_axis_label = 'Time', y_axis_label = 'Frequency'))
            else:
                figures.append(bkfigure(plot_width = 1200, plot_height = 400, x_axis_label = 'Time', y_axis_label = 'Frequency', x_range=figures[0].x_range, y_range=figures[0].y_range))
            channelAmp = np.max(fSignal.audioData[:,:,i])
            figures[i].image(image=[fSignal.audioData[:,:,i]], x=0, y=0, dw=fSignal.time[-1], dh=fSignal.dimensionAxes[0][-1], color_mapper=LinearColorMapper(high=channelAmp, low=0, palette=Inferno11))
    else:
        print("Unrecognized transform given!")
        return 4
    
    bkshow(bkcolumn(*figures))
    return 0

if __name__ == "__main__":
    import time
    start = time.process_time()
    returnCode = main()
    print("DONE processing : Took ", time.process_time() - start, "s\n")
    sys.exit(returnCode)