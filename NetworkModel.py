# All DNN models and training functions here

# PyTorch imports
import torch
from torch.nn import Conv1d, MaxPool1d, Softmax, Module, Linear
from torch.utils.data import Dataset, BatchSampler, Sampler, DataLoader
from torch.optim import SGD, Adam
from torch.fft import fft
import matplotlib.pyplot as plt
# Signal processing imports
from scipy.io import wavfile
from scipy.signal import stft
import numpy as np

# Few handy utils
from math import ceil
import random
import json
import tqdm
import os

# Plotting utility imports
from PlotterUtils import AverageMeter, VisdomLinePlotter

class ModelBase(Module):
    InputShape = (2000, 1)

# conv1d -> maxPooling1d (2) -> fc layer -> relu -> fc layer
class Model1(ModelBase):
    ConvolutionKernelSize = 5
    ConvolutionPadding = 2
    PoolingKernelSize = 2
    InputToOutputRatio = 2
    def __init__(self):
        super(Model1, self).__init__()
        self.conv1 = Conv1d(in_channels = ModelBase.InputShape[1], out_channels = ModelBase.InputShape[1], kernel_size = Model1.ConvolutionKernelSize, padding = Model1.ConvolutionPadding)
        self.maxpool1 = MaxPool1d(Model1.PoolingKernelSize)
        self.fcInputShape = (ModelBase.InputShape[0] // Model1.PoolingKernelSize,  Model1.InputShape[1])
        self.fc1 = Linear(self.fcInputShape[0] *  self.fcInputShape[1], 
                          self.fcInputShape[0] *  self.fcInputShape[1])
        self.fc2 = Linear(self.fcInputShape[0] *  self.fcInputShape[1], 
                          self.fcInputShape[0] *  self.fcInputShape[1])
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = fft(x,dim=2,norm='forward').abs()[:,:,0:ModelBase.InputShape[0]]
        x = torch.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = x.view(-1, self.fcInputShape[0] *  self.fcInputShape[1])
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# conv1d -> maxPooling1d (2) -> fc layer -> relu
class Model2(ModelBase):
    ConvolutionKernelSize = 5
    ConvolutionPadding = 2
    PoolingKernelSize = 2
    InputToOutputRatio = 2
    def __init__(self):
        super(Model2, self).__init__()
        self.conv1 = Conv1d(in_channels = ModelBase.InputShape[1], out_channels = ModelBase.InputShape[1], kernel_size = Model2.ConvolutionKernelSize, padding = Model2.ConvolutionPadding)
        self.maxpool1 = MaxPool1d(Model2.PoolingKernelSize)
        self.fcInputShape = (ModelBase.InputShape[0] // Model2.PoolingKernelSize,  Model2.InputShape[1])
        self.fc1 = Linear(self.fcInputShape[0] *  self.fcInputShape[1], 
                          self.fcInputShape[0] *  self.fcInputShape[1])
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

    def forward(self, x):
        x = fft(x,dim=2,norm='forward').abs()[:,:,0:ModelBase.InputShape[0]]
        x = torch.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = x.view(-1, self.fcInputShape[0] *  self.fcInputShape[1])
        x = torch.relu(self.fc1(x))
        return x


def passthru(x): return x
class NSynthDataSet(Dataset):
    """NSynth Music Frequencies dataset"""

    WavFileTime = 4.0
    SamplingFrequency = 16000
    WindowLength = (ModelBase.InputShape[0] - 1) * 2
    FrequencyBinsCount = ModelBase.InputShape[0]
    Sigma = 2.0 # sqrt(2)*sigma


    def __init__(self, root_dir, transform = passthru, filterString = '', model = Model1, windowStep = 4000):
        with open(os.path.join(root_dir, 'examples.json'), 'r') as f:
            self.labelMap = json.loads(f.read())
            if filterString:
                self.labelMap = { i : 440.0 * (2.0 ** ((self.labelMap[i]['pitch'] - 69)/12)) for i in self.labelMap if filterString in i }
            else:
                self.labelMap = { i : 440.0 * (2.0 ** ((self.labelMap[i]['pitch'] - 69)/12)) for i in self.labelMap }
        self.labelVector = list(self.labelMap.keys())
        self.frequencyValues = { i : None for i in self.labelMap.values() }
        self.windowStep = windowStep
        self.outputBinsCount = ModelBase.InputShape[0] // model.InputToOutputRatio
        for f in self.frequencyValues:
            self.frequencyValues[f] = np.linspace(0, NSynthDataSet.SamplingFrequency/2, self.outputBinsCount)
            self.frequencyValues[f] = np.exp(-((self.frequencyValues[f]-f)**2 / (NSynthDataSet.Sigma**2)))

        self.windowsPerWav = int(NSynthDataSet.WavFileTime * NSynthDataSet.SamplingFrequency // self.windowStep)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labelMap) * ceil(NSynthDataSet.WavFileTime * NSynthDataSet.SamplingFrequency / self.windowStep)

    def __getitem__(self, idx):
        wav_file_name = os.path.join(self.root_dir, 'audio', self.labelVector[idx[0] // self.windowsPerWav] + '.wav')
        _, data = wavfile.read(wav_file_name)
        idx = np.array(idx)
        idx = (idx % self.windowsPerWav)

        _, __, Zxx = stft(
            data, 
            fs=NSynthDataSet.SamplingFrequency, 
            window='hann', 
            nperseg=NSynthDataSet.WindowLength, 
            noverlap=NSynthDataSet.WindowLength-self.windowStep, 
            nfft=None, 
            detrend=False, 
            return_onesided=True, 
            boundary='zeros', 
            padded=True, 
            axis=-1)


        ## Dimensions will be WindowLength/2 x self.windowsPerWav
        # We need dimensions of WindowLength/2 x len(idx)
        Y = np.ndarray(shape = (self.windowsPerWav, 1, NSynthDataSet.FrequencyBinsCount), dtype=np.float32)
        Y[:,0,:] = abs(Zxx[:,idx]).transpose()
        return self.transform(Y), self.frequencyValues[self.labelMap[self.labelVector[idx[0] // self.windowsPerWav]]]

class NSynthRamLoadedDataSet(Dataset):
    """NSynth Music Frequencies dataset - but all data loaded onto RAM at init"""

    def __init__(self, root_dir, transform = passthru, filterString = '', model = Model1, windowStep = 4000, device = torch.device('cpu')):
        with open(os.path.join(root_dir, 'examples.json'), 'r') as f:
            self.labelMap = json.loads(f.read())
            if filterString:
                self.labelMap = { i : 440.0 * (2.0 ** ((self.labelMap[i]['pitch'] - 69)/12)) for i in self.labelMap if filterString in i }
            else:
                self.labelMap = { i : 440.0 * (2.0 ** ((self.labelMap[i]['pitch'] - 69)/12)) for i in self.labelMap }
        self.labelVector = list(self.labelMap.keys())
        self.frequencyValues = { i : None for i in self.labelMap.values() }
        self.windowStep = windowStep
        self.outputBinsCount = ModelBase.InputShape[0] // model.InputToOutputRatio
        for f in self.frequencyValues:
            self.frequencyValues[f] = np.linspace(0, NSynthDataSet.SamplingFrequency/2, self.outputBinsCount)
            self.frequencyValues[f] = np.exp(-((self.frequencyValues[f]-f)**2 / (NSynthDataSet.Sigma**2)))

        self.windowsPerWav = int(NSynthDataSet.WavFileTime * NSynthDataSet.SamplingFrequency // self.windowStep)
        self.windowsTotal  = int(self.windowsPerWav * len(self.labelVector))
        self.root_dir = root_dir
        self.transform = transform
        self.inputs = np.ndarray(shape=(self.windowsTotal, 1, NSynthDataSet.FrequencyBinsCount), dtype=np.float32)

        for wavFileIdx in tqdm.tqdm(range(len(self.labelVector))):
            wavFilePath = os.path.join(self.root_dir, 'audio', self.labelVector[wavFileIdx] + '.wav')
            _, data = wavfile.read(wavFilePath)
            _, __, Zxx = stft(
                data, 
                fs=NSynthDataSet.SamplingFrequency, 
                window='hann', 
                nperseg=NSynthDataSet.WindowLength, 
                noverlap=NSynthDataSet.WindowLength-self.windowStep, 
                nfft=None, 
                detrend=False, 
                return_onesided=True, 
                boundary='zeros', 
                padded=True, 
                axis=-1
            )
            wavFileWindowStart, wavFileWindowEnd = int(wavFileIdx*self.windowsPerWav), int(wavFileIdx*self.windowsPerWav + self.windowsPerWav)
            self.inputs[wavFileWindowStart:wavFileWindowEnd,0,:] = self.transform(abs(Zxx[:,0:self.windowsPerWav]).transpose())

    def __len__(self):
        return len(self.labelMap) * ceil(NSynthDataSet.WavFileTime * NSynthDataSet.SamplingFrequency / self.windowStep)

    def __getitem__(self, idx):
        idx = np.array(idx)
        return self.inputs[idx, :, :], self.frequencyValues[self.labelMap[self.labelVector[idx[0] // self.windowsPerWav]]]

class NSynthChunkedDataSet(Dataset):
    """NSynth Music Frequencies dataset - but data loaded onto RAM at demand in *chunks*"""

    def __init__(self, root_dir, chunkSize, transform = passthru, filterString = '', model = Model1, windowStep = 4000, device = torch.device('cpu')):
        with open(os.path.join(root_dir, 'examples.json'), 'r') as f:
            self.labelMap = json.loads(f.read())
            if filterString:
                self.labelMap = { i : 440.0 * (2.0 ** ((self.labelMap[i]['pitch'] - 69)/12)) for i in self.labelMap if filterString in i }
            else:
                self.labelMap = { i : 440.0 * (2.0 ** ((self.labelMap[i]['pitch'] - 69)/12)) for i in self.labelMap }
        self.chunkSize = chunkSize
        self.labelVector = list(self.labelMap.keys())
        self.frequencyValues = { i : None for i in self.labelMap.values() }
        self.windowStep = windowStep
        self.outputBinsCount = ModelBase.InputShape[0] // model.InputToOutputRatio
        self.device = device

        self.frequencyValues = np.ndarray(shape=(len(self.labelVector), self.outputBinsCount), dtype=np.float32)
        for labelIdx in range(len(self.labelVector)):
            self.frequencyValues[labelIdx,:] = np.linspace(0, NSynthDataSet.SamplingFrequency/2, self.outputBinsCount)
            self.frequencyValues[labelIdx,:] = np.exp(-((self.frequencyValues[labelIdx,:]-self.labelMap[self.labelVector[labelIdx]])**2 / (NSynthDataSet.Sigma**2)))
        self.frequencyValues = torch.from_numpy(self.frequencyValues).to(self.device)

        self.windowsPerWav = int(NSynthDataSet.WavFileTime * NSynthDataSet.SamplingFrequency // self.windowStep)
        self.signalLength = (self.windowsPerWav * self.windowStep) + NSynthDataSet.WindowLength
        self.root_dir = root_dir
        self.transform = transform
        self.wavFilePerChunk = self.chunkSize // self.windowsPerWav
        self.windowsTotal = int(len(self.labelVector) * self.windowsPerWav)
        self.currentlyAvailableChunkIdx = -1 # INVALID

    def __len__(self):
        return len(self.labelMap) * ceil(NSynthDataSet.WavFileTime * NSynthDataSet.SamplingFrequency / self.windowStep)

    def __getitem__(self, idx):
        self.loadChunk(idx[0])
        idx = np.array(idx)
        labelIdx = idx // self.windowsPerWav
        wavIdx = labelIdx % self.wavFilePerChunk
        offset = (idx % self.windowsPerWav) * self.windowStep
        return torch.stack([ self.inputs[j, :, i:i+NSynthDataSet.WindowLength] for i,j in zip(offset, wavIdx) ]), self.frequencyValues[labelIdx,:]

    def loadChunk(self, startIdx):
        chunkIdx = startIdx // self.chunkSize
        if chunkIdx == self.currentlyAvailableChunkIdx: return
        
        self.currentlyAvailableChunkIdx = chunkIdx
        # Clean up old inputs
        self.inputs = None
        self.inputs = np.zeros(shape=(self.wavFilePerChunk, 1, self.signalLength), dtype=np.float32)
        print ("Loading wav files : ", chunkIdx * self.wavFilePerChunk, ":", chunkIdx * self.wavFilePerChunk + self.wavFilePerChunk)

        for wavFileIdx in range(chunkIdx * self.wavFilePerChunk, min(chunkIdx * self.wavFilePerChunk + self.wavFilePerChunk, len(self.labelVector))):
            wavFilePath = os.path.join(self.root_dir, 'audio', self.labelVector[wavFileIdx] + '.wav')
            _, data = wavfile.read(wavFilePath)
            self.inputs[wavFileIdx % self.wavFilePerChunk, 0, 0:len(data)] = data
        self.inputs = torch.from_numpy(self.inputs).to(self.device)

class WavRandomSampler(Sampler):
    def __init__(self, chunkSize, chunkIndices, lastChunkIndex, lastChunkSize):
        self.countGroups = len(chunkIndices)
        if lastChunkIndex in chunkIndices:
            self.fullSize = chunkSize * (self.countGroups - 1) + lastChunkSize
        else:
            self.fullSize = chunkSize * self.countGroups
        self.sequence = []
        groups = np.random.choice(chunkIndices, size=self.countGroups, replace=False)
        for g in groups:
            # Choose a group
            actualChunkSize = chunkSize
            if lastChunkIndex in chunkIndices : actualChunkSize = lastChunkSize
            indicesInGroup = np.random.choice(range(0, actualChunkSize), size=actualChunkSize, replace=False)
            for i in indicesInGroup :
                # Choose indices in the group
                self.sequence.append(g*chunkSize + i)
    def __len__(self):
        return self.fullSize
    def __iter__(self):
        return iter(self.sequence)
        
def train(
    root_dir = 'assets\\nsynth_test', 
    model_class = Model1, 
    dataset_class = NSynthDataSet,
    epochs = 20, 
    learning_rate = 1e-3, 
    batchSize = 16,
    memoryLimitInMB = 1024,
    validationSplit = 0.1,
    device = torch.device('cpu'), 
    save_path = 'model.pth', 
    windowStep = 4000):

    # Calculate chunk size based on the memory limit allowed
    windowsPerWav = int(NSynthDataSet.WavFileTime * NSynthDataSet.SamplingFrequency // windowStep)

    if dataset_class == NSynthChunkedDataSet:
        chunkSize = ceil(int(memoryLimitInMB * 1024 * 1024 / NSynthDataSet.FrequencyBinsCount / 4) / windowsPerWav) * windowsPerWav
        data_set = NSynthChunkedDataSet(root_dir = root_dir, chunkSize = chunkSize, model = model_class, windowStep = windowStep, device=device)
    else:
        data_set = dataset_class(root_dir = root_dir, model = model_class,  windowStep = windowStep)

    # Initialize the sampler to split between train and validation sets
    allChunkIndices = list(range(0, ceil(len(data_set) / chunkSize)))
    lastChunkSize = len(data_set) % chunkSize
    validationChunkLen = int(np.ceil(validationSplit * len(allChunkIndices)))
    validationChunkIndices = np.random.choice(allChunkIndices, size=validationChunkLen, replace=False)
    trainChunkIndices = list(set(allChunkIndices) - set(validationChunkIndices))

    trainWavRandomSampler = WavRandomSampler(chunkSize, trainChunkIndices, allChunkIndices[-1], lastChunkSize)
    validationWavRandomSampler = WavRandomSampler(chunkSize, validationChunkIndices, allChunkIndices[-1], lastChunkSize)
    
    # Data loader object to load wav files on demand
    trainDataLoader = DataLoader(
        data_set,
        shuffle = False,
        num_workers = 0,
        batch_size = None, # Specially needed - else the auto_collation makes batch sampling useless!
        sampler = BatchSampler(
            trainWavRandomSampler,
            batch_size = batchSize,
            drop_last = False
        )
    )
    validationDataLoader = DataLoader(
        data_set,
        shuffle = False,
        num_workers = 0,
        batch_size = None, # Specially needed - else the auto_collation makes batch sampling useless!
        sampler = BatchSampler(
            validationWavRandomSampler,
            batch_size = batchSize,
            drop_last = False
        )
    )

    # Plotter object to plot the losses
    plotter = VisdomLinePlotter(env_name='Loss plot')

    # Port the model to device
    model = model_class()
    model = model.to(device)

    # Initialize the optimizer and data set first
    optimizer = Adam(model.parameters(), lr = learning_rate)
    
    # Epoch loop
    for epoch in tqdm.tqdm(range(1, epochs + 1)):
        # Object to aggregate losses over the epoch
        trainLosses = AverageMeter()
        validationLosses = AverageMeter()


        # Iterate over the dataset from training
        model.train(True)
        for batch_idx, batch in enumerate(trainDataLoader):
            # Get the inputs from the dataset
            inputs, labels = batch

            # Zero the parameter gradients 
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model.forward(inputs)
            loss = (outputs - labels).pow(2).mean().sqrt()
            trainLosses.update(loss.data.cpu().numpy(), labels.size(0))
            loss.backward()
            optimizer.step()

        # Update the Plot for the losses in this epoch
        plotter.plot('loss', 'train', 'Class Loss', epoch, trainLosses.avg)
        print('Epoch', epoch, ' : training DONE. TrainingLoss =', trainLosses.avg)

        model.eval()
        for batch_idx, batch in enumerate(validationDataLoader):
            # Get the inputs from the dataset
            inputs, labels = batch

            # forward
            outputs = model.forward(inputs)
            loss = (outputs - labels).pow(2).mean().sqrt()
            validationLosses.update(loss.data.cpu().numpy(), labels.size(0))

        # Update the Plot for the losses in this epoch
        plotter.plot('loss', 'validation', 'Class Loss', epoch, validationLosses.avg)
        print('Epoch', epoch, ' : validation DONE. ValidationLoss =', validationLosses.avg)
    
    torch.save(model.state_dict(), save_path)

def test(
    root_dir = 'assets\\nsynth_test', 
    model_class = Model1, 
    dataset_class = NSynthDataSet,
    memoryLimitInMB = 1024,
    device = torch.device('cpu'), 
    load_path = 'model.pth', 
    windowStep = 4000):

    # Calculate chunk size based on the memory limit allowed
    windowsPerWav = int(NSynthDataSet.WavFileTime * NSynthDataSet.SamplingFrequency // windowStep)

    if dataset_class == NSynthChunkedDataSet:
        chunkSize = ceil(int(memoryLimitInMB * 1024 * 1024 / NSynthDataSet.FrequencyBinsCount / 4) / windowsPerWav) * windowsPerWav
        data_set = NSynthChunkedDataSet(root_dir = root_dir, chunkSize = chunkSize, model = model_class, windowStep = windowStep, device=device)
    else:
        data_set = dataset_class(root_dir = root_dir, model = model_class, windowStep = windowStep)

    # Initialize the sampler to split between train and validation sets
    allChunkIndices = list(range(0, ceil(len(data_set) / chunkSize)))
    lastChunkSize = len(data_set) % chunkSize
    if (lastChunkSize == 0) : lastChunkSize = chunkSize

    testWavRandomSampler = WavRandomSampler(chunkSize, allChunkIndices, allChunkIndices[-1], lastChunkSize)
    
    testDataLoader = DataLoader(
        data_set,
        shuffle = False,
        num_workers = 0,
        batch_size = None, # Specially needed - else the auto_collation makes batch sampling useless!
        sampler = BatchSampler(
            testWavRandomSampler,
            batch_size = 1,
            drop_last = False
        )
    )

    # Plotter object to plot the losses for each window
    plotter = VisdomLinePlotter(env_name='Accuracy plot')

    # Port the model to device
    model = model_class()
    model.load_state_dict(torch.load(load_path))
    model = model.to(device)

    model.eval()
    frequencyBinRange = torch.from_numpy((NSynthDataSet.SamplingFrequency/2.0/data_set.outputBinsCount) *  np.array(range(data_set.outputBinsCount))).to(device)
    for batch_idx, batch in enumerate(testDataLoader):
        # Get the inputs from the dataset
        inputs, labels = batch

        # forward
        outputs = model.forward(inputs)

        #outputs = outputs.detach().cpu().numpy()
        #expected = labels.detach().cpu().numpy()

        #print(outputs,expected)
        #plt.plot(outputs[0,:])
        #plt.plot(expected[0,:])
        #plt.show()
        #return 0

        frequencyWeightedSum = torch.sum((outputs * frequencyBinRange),axis=1)
        outputSum = torch.sum(outputs, axis=1)
        observed = (frequencyWeightedSum/outputSum).detach().cpu().numpy()[0]
        plotter.plot('Hz', 'Observed', 'Frequencies', batch_idx, observed)

        expectedFrequencyWeightedSum = torch.sum((labels * frequencyBinRange),axis=1)
        expectedSum = torch.sum(labels, axis=1)
        expected = (expectedFrequencyWeightedSum/expectedSum).detach().cpu().numpy()[0]
        plotter.plot('Hz', 'Expected', 'Frequencies', batch_idx, expected)
