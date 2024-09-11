import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import yaml
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_config():
    with open('FILE.yaml', 'r') as file:
        return yaml.safe_load(file)
config = load_config()

class Forecasting_Dataset(Dataset):
    def __init__(self, datatype, mode="train"):
        self.history_length = 365
        self.pred_length = 30

        self.seq_length = self.history_length + self.pred_length

        data = pd.read_csv(datatype)
        data['start_date'] = pd.to_datetime(data['start_date'])
        data = data[~((data['start_date'].dt.month == 2) & (data['start_date'].dt.day == 29))]
        self.data_pivot = data.pivot(index='start_date', columns='series_name', values='daily_series_value')


        assert self.data_pivot.shape[1] == 320, "Expected 320 series (features) in the dataset."

        scaler = MinMaxScaler()
        self.main_data = scaler.fit_transform(self.data_pivot.values)
        self.mean_data = scaler.data_min_
        self.std_data = scaler.data_max_
        
        # Adjust to shape (time, features) -> (time, features, 1) -> (time, features)
        self.main_data = self.main_data[:, :, np.newaxis]  # -> shape: (time, features, 1)
        self.main_data = np.squeeze(self.main_data, axis=-1)  # -> shape: (time, features)

        # Mask data creation
        self.mask_data = np.ones_like(self.main_data)
        total_length = len(self.main_data)
        self.test_length = self.pred_length*4 #int(0.15 * total_length)  
        self.valid_length =self.pred_length*4 #int(0.15 * total_length)  

        if mode == 'train':
            start = 0
            end = total_length - self.seq_length - self.valid_length - self.test_length + 1
            end = max(0, end) 
            self.use_index = np.arange(start, end, 1)
        elif mode == 'valid':
            start = total_length - self.seq_length - self.valid_length - self.test_length + self.pred_length
            end = total_length - self.seq_length - self.test_length + self.pred_length
            end = min(total_length - self.seq_length, end)  
            self.use_index = np.arange(start, end, self.pred_length)
        elif mode == 'test':
            start = total_length - self.seq_length - self.test_length + self.pred_length
            end = total_length - self.seq_length + self.pred_length
            end = min(total_length - self.seq_length, end)  
            self.use_index = np.arange(start, end, self.pred_length)



    def __getitem__(self, orgindex):
        index = self.use_index[orgindex]
        target_mask = self.mask_data[index:index+self.seq_length].copy()
        target_mask[-self.pred_length:] = 0.  # pred mask for test pattern strategy

        timepoints = self.data_pivot.index[index:index+self.seq_length].values
        timepoints = timepoints.astype('datetime64[s]').astype(np.int64)  # datetime64 -> int64
        timepoints = np.expand_dims(timepoints, axis=-1)  # (seq_length, 1)
        timepoints = np.tile(timepoints, (1, self.main_data.shape[1]))  # (seq_length, features)

            # B, K, L -> B, L, K
        observed_data = torch.tensor(self.main_data[index:index+self.seq_length], dtype=torch.float32).transpose(0, 1)  # -> (features, seq_length) -> (seq_length, features)
        observed_mask = torch.tensor(self.mask_data[index:index+self.seq_length], dtype=torch.float32).transpose(0, 1)
        gt_mask = torch.tensor(target_mask, dtype=torch.float32).transpose(0, 1)
        timepoints = torch.tensor(timepoints, dtype=torch.float32).transpose(0, 1)

        s = {
            'observed_data': self.main_data[index:index+self.seq_length],  # (seq_length, features)
            'observed_mask': self.mask_data[index:index+self.seq_length],  # (seq_length, features)
            'gt_mask': target_mask,                                        # (seq_length, features)
            'timepoints': timepoints,                                      # (seq_length, features)
            'feature_id': np.arange(self.main_data.shape[1]) * 1.0,
        }

        s['observed_data'] = s['observed_data'].transpose(0, 1)  # (features, seq_length) -> (seq_length, features)
        s['observed_mask'] = s['observed_mask'].transpose(0, 1)
        s['gt_mask'] = s['gt_mask'].transpose(0, 1)
        s['timepoints'] = s['timepoints'].transpose(0, 1)

        return s
    
    def __len__(self):
        return len(self.use_index)


def get_dataloader(datatype,device,batch_size=8):
    dataset = Forecasting_Dataset(datatype,mode='train')
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = Forecasting_Dataset(datatype,mode='valid')
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = Forecasting_Dataset(datatype,mode='test')
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=0)
    
    scaler = torch.from_numpy(dataset.std_data.astype(np.float32)).to(device)
    mean_scaler = torch.from_numpy(dataset.mean_data.astype(np.float32)).to(device)

    return train_loader, valid_loader, test_loader, scaler, mean_scaler
