import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import os


class SRDataset(Dataset):
    def __init__(self, scale_factor = 2, datasets={}, has_mv=True) -> None:
        super().__init__()
        train_txt = open(f"{datasets['train']['txt_path']}", "r")
        self.train_list = [i[:-1] for i in train_txt.readlines()]
        self.lr_dir = datasets['train']['lr_dir']
        self.hr_dir = datasets['train']['hr_dir']
        self.scale_factor = scale_factor
        self.has_mv = has_mv
    
    def __getitem__(self, index):
        target_dir = self.train_list[index // 5]
        target_frame = (index % 5) + 1
        lr_path = f"{self.lr_dir}/{target_dir:0>4}"
        hr_path = f"{self.hr_dir}/{target_dir:0>4}"
        f1_lr = (cv2.imread(f"{lr_path}/im{target_frame}.png")[...,::-1].astype('float')/255).transpose(2, 0, 1)
        f2_lr = (cv2.imread(f"{lr_path}/im{target_frame+1}.png")[...,::-1].astype('float')/255).transpose(2, 0, 1)
        f3_lr = (cv2.imread(f"{lr_path}/im{target_frame+2}.png")[...,::-1].astype('float')/255).transpose(2, 0, 1)
        f1_hr = (cv2.imread(f"{hr_path}/im{target_frame}.png")[...,::-1].astype('float')/255).transpose(2, 0, 1)
        f2_hr = (cv2.imread(f"{hr_path}/im{target_frame+1}.png")[...,::-1].astype('float')/255).transpose(2, 0, 1)
        f3_hr = (cv2.imread(f"{hr_path}/im{target_frame+2}.png")[...,::-1].astype('float')/255).transpose(2, 0, 1)
        if self.has_mv:
            f2_mv = (np.load(f"{lr_path}/mv{target_frame+1}.npy")).astype('float').transpose(2, 0, 1)
            f3_mv = (np.load(f"{lr_path}/mv{target_frame+2}.npy")).astype('float').transpose(2, 0, 1)
            x = np.concatenate([f1_lr, f2_mv, f2_lr, f3_mv, f3_lr])
        else:
            x = np.concatenate([f1_lr, f2_lr, f3_lr])
        y = np.concatenate([f1_hr, f2_hr, f3_hr])
        return x, y
    
    def __len__(self):
        return len(self.train_list) * 5
