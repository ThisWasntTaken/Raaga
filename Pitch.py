import sys
import os
import argparse
import MP3
import numpy as np
from bokeh.plotting import figure as bkfigure
from bokeh.io import show as bkshow
from bokeh.layouts import column as bkcolumn
import bokeh
import Transforms

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Read an mp3 file and plot out its pitch')
    parser.add_argument('-i', dest='input', type=str, help='Input file path', default='')
    parser.add_argument('-n', dest='normalize', action='store_true', help='Normalize input values', default='')
    parser.add_argument('-p', dest='plot', type=str, help='Plot signal name', default='input')
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

    # STFT over the signal
    fSignal = Transforms.STFT(inSignal)

    figures = []
    if options.plot == 'input':
        # Plot the data for quick visualization
        time = np.arange(0, inSignal.time, inSignal.samplingPeriod, dtype=np.float32)

        figures.append(bkfigure(plot_width = 1200, plot_height = 600, x_axis_label = 'Time'))
        figures[0].line(time, inSignal.audioData[:,0])
        for i in range(1, inSignal.channels):
            figures.append(bkfigure(plot_width = 1200, plot_height = 600, x_axis_label = 'Time', x_range=figures[0].x_range))
            figures[i].line(time, inSignal.audioData[:,i])
    else:
        time = np.arange(0, fSignal.time, inSignal.samplingPeriod, dtype=np.float32)
        pass
    
    bkshow(bkcolumn(*figures))

if __name__ == "__main__":
    sys.exit(main())