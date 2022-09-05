import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Resize 


class TrafficDataset(Dataset):
    def __init__(self, X, Y):
        super(TrafficDataset, self).__init__()
        self.X = (X + 1) / 2
        self.Y = (Y + 1) / 2
        self.mean = 0
        self.std = 1

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index, ::]).float()
        labels = torch.tensor(self.Y[index, ::]).float()
        return data, labels

class MorUAV(Dataset):
    def __init__(self, data, n_input_frames, n_output_frames):
        super(MorUAV, self).__init__()
        self.data = data
        self.n_input_frames = n_input_frames
        self.n_output_frames = n_output_frames
        self._data_process()
        self.mean = 0
        self.std = 1

    # 对每个视频片段进行截取，使其长度是length的整数倍
    def _data_process(self):
        print('data process start...')
        new_data = []
        self.length = self.n_input_frames + self.n_output_frames
        self.len = 0
        for i in self.data:
            cut = i.shape[0] - (i.shape[0] % self.length)
            new_data.append(i[:cut])
            self.len += cut
        self.data = np.concatenate(tuple(new_data), axis=0)
        self.len = int(self.len / self.length)
        print('data process end with len ',self.len)

    def __len__(self):
        return self.len
        
    def __getitem__(self, index):
        start = index * self.length
        end = start + self.length
        item = self.data[start:end]
        data = item[:self.n_input_frames] 
        label = item[self.n_input_frames:]
        data = torch.tensor(data.transpose(0,3,1,2)).float()
        label = torch.tensor(label.transpose(0,3,1,2)).float()
        T,C,_,_ = data.shape
        data = data.resize_(T,C,64,64)
        T_,C_,_,_ = label.shape
        label = label.resize_(T_,C_,64,64)
        data = data/255
        label = label/255
        return data, label

def load_data(
        batch_size, val_batch_size,
        data_root, num_workers):

    train_data = np.load(data_root+'mor_uav/train.npy', allow_pickle=True)
    val_data = np.load(data_root+'mor_uav/val.npy',allow_pickle=True)
    test_data = np.load(data_root+'mor_uav/test.npy',allow_pickle=True)
    
    train_set = MorUAV(train_data, 10, 5)
    val_set = MorUAV(val_data, 10, 5)
    test_set = MorUAV(test_data, 10, 5)

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_val = torch.utils.data.DataLoader(
        val_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return dataloader_train, dataloader_val, dataloader_test, 0, 1

# if __name__=="__main__":
#     data_root = '/workspace/xly-FramePrediction/SimVP/data/'
#     batch_size = val_batch_size = 16
#     num_workers = 8
#     train_loader, vali_loader, test_loader, data_mean, data_std = load_data(batch_size, val_batch_size, data_root, num_workers)
#     for data, label in train_loader:
#         print(data.shape)
#         print(label.shape)
#         break