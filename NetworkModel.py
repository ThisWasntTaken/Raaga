# All DNN models and training functions here

import torch
from torch.nn import Conv1d, MaxPool1d, Softmax, Module, Linear
from torch.utils.data import Dataset, BatchSampler, Sampler, DataLoader
from torch.optim import SGD, Adam
from math import ceil
from scipy.io import wavfile
from scipy.signal import stft
import numpy as np
import random
import json
import tqdm
import os


class ModelBase(Module):
    InputShape = (2000, 1)

class Model1(ModelBase):
    ConvolutionKernelSize = 5
    ConvolutionPadding = 2
    PoolingKernelSize = 2
    def __init__(self):
        super(Model1, self).__init__()
        self.conv1 = Conv1d(in_channels = ModelBase.InputShape[1], out_channels = ModelBase.InputShape[1], kernel_size = Model1.ConvolutionKernelSize, padding = Model1.ConvolutionPadding)
        self.maxpool1 = MaxPool1d(Model1.PoolingKernelSize)
        self.fcInputShape = (ModelBase.InputShape[0] // Model1.PoolingKernelSize,  Model1.InputShape[1])
        self.fc1 = Linear(self.fcInputShape[0] *  self.fcInputShape[1], 
                          self.fcInputShape[0] *  self.fcInputShape[1])
        self.fc2 = Linear(self.fcInputShape[0] *  self.fcInputShape[1], 
                          self.fcInputShape[0] *  self.fcInputShape[1])
        torch.nn.init.xavier_uniform(self.conv1.weight)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        torch.nn.init.xavier_uniform(self.fc2.weight)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = x.view(-1, self.fcInputShape[0] *  self.fcInputShape[1])
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def passthru(x): return x
class NSynthDataSet(Dataset):
    """NSynth Music Frequencies dataset"""

    WavFileTime = 4.0
    SamplingFrequency = 16000
    WindowStep = 4000
    WindowLength = ModelBase.InputShape[0] * 2
    FrequencyBinsCount = ModelBase.InputShape[0]
    OutputBinsCount = ModelBase.InputShape[0] // 2
    Sigma = 2.0 # sqrt(2)*sigma


    def __init__(self, root_dir, transform = passthru, filterString = ''):
        with open(os.path.join(root_dir, 'examples.json'), 'r') as f:
            self.labelMap = json.loads(f.read())
            if filterString:
                self.labelMap = { i : 440.0 * (2.0 ** ((self.labelMap[i]['pitch'] - 69)/12)) for i in self.labelMap if filterString in i }
            else:
                self.labelMap = { i : 440.0 * (2.0 ** ((self.labelMap[i]['pitch'] - 69)/12)) for i in self.labelMap }
        self.labelVector = list(self.labelMap.keys())
        self.frequencyValues = { i : None for i in self.labelMap.values() }
        for f in self.frequencyValues:
            self.frequencyValues[f] = np.linspace(0, NSynthDataSet.SamplingFrequency/2, NSynthDataSet.OutputBinsCount)
            self.frequencyValues[f] = np.exp(-((self.frequencyValues[f]-f)**2 / (NSynthDataSet.Sigma**2)))

        self.windowsPerWav = int(NSynthDataSet.WavFileTime * NSynthDataSet.SamplingFrequency // NSynthDataSet.WindowStep)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labelMap) * ceil(NSynthDataSet.WavFileTime * NSynthDataSet.SamplingFrequency / NSynthDataSet.WindowStep)

    def __getitem__(self, idx):
        idx = [idx]
        wav_file_name = os.path.join(self.root_dir, 'audio', self.labelVector[idx[0] // self.windowsPerWav] + '.wav')
        _, data = wavfile.read(wav_file_name)
        idx = np.array(idx)
        idx = (idx % self.windowsPerWav)

        _, __, Zxx = stft(
            data, 
            fs=NSynthDataSet.SamplingFrequency, 
            window='hann', 
            nperseg=NSynthDataSet.WindowLength, 
            noverlap=NSynthDataSet.WindowLength-NSynthDataSet.WindowStep, 
            nfft=None, 
            detrend=False, 
            return_onesided=True, 
            boundary='zeros', 
            padded=True, 
            axis=-1)


        ## Dimensions will be WindowLength/2 x self.windowsPerWav
        # We need dimensions of WindowLength/2 x len(idx)
        return self.transform(abs(Zxx[:,idx].transpose())), self.frequencyValues[self.labelMap[self.labelVector[idx[0] // self.windowsPerWav]]]

class WavRandomSampler(Sampler):
    def __init__(self, indexRange, groupSize):
        self.indexRange = indexRange
        self.groupSize = groupSize
        self.groupRange = (self.indexRange[0], int(self.indexRange[1] / self.groupSize))
        #print ("SAMPLER INPUTS :",self.indexRange, self.groupSize, self.groupRange)
        self.sequence = []
        for _ in range(self.groupRange[0], self.groupRange[1]):
            # Choose a group
            group = random.randint(self.groupRange[0], self.groupRange[1]-1)
            for __ in range(0, self.groupSize):
                # Choose indices in the group
                indexInGroup = random.randint(0, self.groupSize-1)
                self.sequence.append(group*self.groupSize + indexInGroup)

    def __len__(self):
        return self.indexRange[1] - self.indexRange[0]

    def __iter__(self):
        return iter(self.sequence)
        
def train(root_dir = 'assets\\nsynth_test', model_class = Model1, epochs = 20, learning_rate = 1e-3, device = torch.device('cpu'), save_path = 'model.pth'):
    model = model_class().to(device)
    optimizer = Adam(model.parameters(), lr = learning_rate)
    data_set = NSynthDataSet(root_dir = root_dir)

    for _ in tqdm.tqdm(range(1, epochs + 1)):
        data_loader = DataLoader(
            data_set,
            shuffle = False,
            num_workers = 1,
            batch_sampler = BatchSampler(
                WavRandomSampler((0, len(data_set)),data_set.windowsPerWav),
                batch_size = data_set.windowsPerWav,
                drop_last = False
            )
        )

        for batch_idx, batch in enumerate(data_loader):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = (outputs - labels).pow(2).mean().sqrt()
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), save_path)