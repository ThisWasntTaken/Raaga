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
    # Plot the data for quick visualization
    if options.plot == 'input':
        for i in range(0, inSignal.channels):
            if i == 0:
                figures.append(bkfigure(plot_width = 1200, plot_height = 600, x_axis_label = 'Time', y_axis_label = 'Amp'))
            else:
                figures.append(bkfigure(plot_width = 1200, plot_height = 600, x_axis_label = 'Time', y_axis_label = 'Amp', x_range=figures[0].x_range))
            figures[i].line(inSignal.time, inSignal.audioData[:,i])
    elif options.plot == 'stft':
        for i in range(0, inSignal.channels):
            if i == 0:
                figures.append(bkfigure(plot_width = 1200, plot_height = 400, x_axis_label = 'Time', y_axis_label = 'Frequency'))
            else:
                figures.append(bkfigure(plot_width = 1200, plot_height = 400, x_axis_label = 'Time', y_axis_label = 'Frequency', x_range=figures[0].x_range, y_range=figures[0].y_range))
            channelAmp = np.max(fSignal.audioData[:,:,i])
            figures[-1].image(image=[fSignal.audioData[:,:,i]], x=0, y=0, dw=fSignal.time[-1], dh=fSignal.dimensionAxes[0][-1], color_mapper=LinearColorMapper(high=channelAmp, low=0, palette=Inferno11))
        pass
    
    bkshow(bkcolumn(*figures))
    return 0

if __name__ == "__main__":
    sys.exit(main())