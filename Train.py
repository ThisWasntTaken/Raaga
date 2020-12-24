import sys
import os
import argparse
import numpy as np
import torch

import NetworkModel


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train on a given dataset to recognize pitch')
    parser.add_argument('-i', dest='inputModel', type=str, help='Input data set file path', default='')
    parser.add_argument('-m', dest='modelType', type=str, help='Set the model type', default='Model1')
    parser.add_argument('-r', dest='learningRate', type=float, help='Set the learning rate', default='0.001')
    parser.add_argument('-e', dest='epochs', type=int, help='Set the epochs', default='20')
    parser.add_argument('-g', dest='gpu', action='store_true', help='Set the training to run on gpu')
    parser.add_argument('-o', dest='output', type=str, help='Set the output path', default='model.pth')

    options = parser.parse_args()

    # Error check
    if options.inputModel == '':
        print("No input given. BYE!\n")
        return 1
    elif not os.path.isdir(options.inputModel):
        print (f"Given input path {options.inputModel} does not exist!")
        return 2

    if options.gpu:
        device = torch.cuda.device('cuda')
    else:
        device = torch.device('cpu')

    if options.modelType == 'Model1':
        modelClass = NetworkModel.Model1

    NetworkModel.train(
        root_dir = 'assets\\nsynth_test', 
        model_class = modelClass, 
        epochs = options.epochs, 
        learning_rate = options.learningRate, 
        device = device, 
        save_path = options.output
    )
    
    return 0

if __name__ == "__main__":
    import time
    start = time.process_time()
    returnCode = main()
    print("DONE : Took ", time.process_time() - start, "s\n")
    sys.exit(returnCode)
