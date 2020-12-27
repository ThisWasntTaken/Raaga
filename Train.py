import sys
import os
import argparse
import numpy as np
import torch

import NetworkModel


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train on a given dataset to recognize pitch')
    parser.add_argument('-i', dest='inputData', type=str, help='Input data set file path', default='')
    parser.add_argument('-m', dest='modelType', type=str, help='Set the model type', default='Model1')
    parser.add_argument('-R', dest='loadDataToRam', action='store_true', help='Preload all data into ram to speedup training')
    parser.add_argument('-C', dest='chunkDataToRam', action='store_true', help='Load chunked data of size -M (in MB) into ram to speedup training')
    parser.add_argument('-M', dest='memoryLimit', type=int, help='Load chunked data of this size (in MB) into ram to speedup training', default='2048')
    parser.add_argument('-r', dest='learningRate', type=float, help='Set the learning rate', default='0.001')
    parser.add_argument('-v', dest='validationSplit', type=float, help='Set the validation split', default='0.1')
    parser.add_argument('-e', dest='epochs', type=int, help='Set the epochs', default='20')
    parser.add_argument('-g', dest='gpu', action='store_true', help='Set the training to run on gpu')
    parser.add_argument('-w', dest='windowStep', type=int, help='Set the window step to use when performing stft', default='4000')
    parser.add_argument('-b', dest='batchSize', type=int, help='Set the batch size in training', default='16')
    parser.add_argument('-o', dest='output', type=str, help='Set the output path', default='model.pth')

    options = parser.parse_args()

    # Error check
    if options.inputData == '':
        print("No input given. BYE!\n")
        return 1
    elif not os.path.isdir(options.inputData):
        print (f"Given input path {options.inputData} does not exist!")
        return 2

    if options.gpu:
        device = torch.device('cuda')
        print ("Using device", torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print ("Using device CPU")

    if options.modelType == 'Model1':
        modelClass = NetworkModel.Model1
    elif options.modelType == 'Model2':
        modelClass = NetworkModel.Model2

    if options.loadDataToRam:
        datasetClass = NetworkModel.NSynthRamLoadedDataSet
    elif options.chunkDataToRam:
        datasetClass = NetworkModel.NSynthChunkedDataSet
    else:
        datasetClass = NetworkModel.NSynthDataSet

    NetworkModel.train(
        root_dir = options.inputData, 
        model_class = modelClass, 
        dataset_class = datasetClass,
        epochs = options.epochs, 
        learning_rate = options.learningRate, 
        memoryLimitInMB = options.memoryLimit,
        device = device, 
        validationSplit = options.validationSplit,
        save_path = options.output,
        batchSize = options.batchSize,
        windowStep = options.windowStep
    )
    
    return 0

if __name__ == "__main__":
    import time
    start = time.process_time()
    returnCode = main()
    print("DONE : Took ", time.process_time() - start, "s\n")
    sys.exit(returnCode)
