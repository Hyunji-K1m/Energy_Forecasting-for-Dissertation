import torch
import numpy as np
from torch.autograd import Variable
import pandas as pd

def smape(actual, predicted):
    denominator = (torch.abs(actual) + torch.abs(predicted))
    diff = torch.abs(actual - predicted) / denominator
    return 200 * torch.mean(diff)

def mase(actual, predicted):
    n = actual.size(0)
    d = torch.mean(torch.abs(actual[1:] - actual[:-1])) 
    epsilon = 1e-10
    d = d + epsilon  
    errors = torch.abs(actual - predicted)
    return torch.mean(errors / d)

def crps(actual, predicted):
    return torch.mean((predicted - actual) ** 2)

def crps_sum(actual, predicted):
    return torch.sum((predicted - actual) ** 2)

def rmse(actual, predicted):
    return torch.sqrt(torch.mean((predicted - actual) ** 2))

def relative_rmse(actual, predicted, train_data):
    rmse_value = rmse(actual, predicted)
    rmse_naive = torch.sqrt(torch.mean((train_data[1:] - train_data[:-1]) ** 2))
    return rmse_value / rmse_naive


class Data_utility(object):
    def __init__(self, file_name, train, valid, cuda, horizon, window, normalize = 2):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS device")
        else:
            self.device = torch.device("cpu")
            print("Using CPU device")
        
        self.P = window
        self.h = horizon
        data = pd.read_csv(file_path)
        self.rawdat = data.drop(columns=['start_date']).values
        self.dat = np.zeros(self.rawdat.shape, dtype=np.float32)
        self.n, self.m = self.dat.shape
        self.min_val = np.zeros(self.m)
        self.max_val = np.zeros(self.m)
        self.scale = np.ones(self.m, dtype=np.float32)
        self.normalize = normalize
        #self.normalize = 2
        self.scale = np.ones(self.m, dtype=np.float32)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train+valid) * self.n), self.n)
        
        self.scale = torch.from_numpy(self.scale).float().to(self.device)
    
    def _normalized(self, normalize):
        # MinMax 정규화
        if normalize == 0:
            self.dat = self.rawdat
            
        if normalize == 1:
            self.dat = (self.rawdat - np.min(self.rawdat, axis=0)) / (np.max(self.rawdat, axis=0) - np.min(self.rawdat, axis=0))

        if normalize == 2:
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:,i]))
                self.dat[:,i] = (self.rawdat[:,i] - np.min(self.rawdat[:,i])) / (np.max(self.rawdat[:,i]) - np.min(self.rawdat[:,i]))
        
    def _split(self, train, valid, test):
        train_set = range(self.P+self.h-1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)
        
    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m), dtype=torch.float32)
        Y = torch.zeros((n, self.m), dtype=torch.float32)
        
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :]).float()
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :]).float()

        return [X.to(self.device), Y.to(self.device)]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            yield X,Y #Variable(X), Variable(Y)
            start_idx += batch_size
