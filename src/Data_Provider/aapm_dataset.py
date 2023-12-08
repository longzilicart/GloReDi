import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np 
from datetime import datetime
import h5py
import glob
import sys
sys.path.append('..')
from utils.ct_tools import *


class AAPM_Myo_Dataset(Dataset):
    """aapm myo dataset
    args:
        root_path: deep_lesion data root path
        mode: train/test
    kwargs:
        image_list_path: a predefined list. divide train/test set. if None, collect all image
        dataset_shape: reshape the image
    return:
        return CT image in miu(attenuation coefficient)
    """
    def __init__(self, root_path, mode, image_list_path=None, split=True, dataset_shape=512, num_train=5410, num_val=526):
        assert mode in ['train', 'val', 'test']
        self.root_path = root_path
        # get image path list
        if image_list_path is None:
            img_path_list = sorted([os.path.join(root_path, _) for _ in os.listdir(root_path) if '.npy' in _])
        else:
            with open(image_list_path, 'r') as f:
                img_path_list = sorted([os.path.join(root_path, i.strip('\n')) for i in f])
        
        # split train val set by simple_split
        if mode in ['train', 'val']:
            if split:
                train_path_list, val_path_list = self.simple_split(img_path_list, num_train, num_val)
                self.img_path_list = train_path_list if mode == 'train' else val_path_list
            else:
                self.img_path_list = img_path_list
        else:  # val as test
            _, val_path_list = self.simple_split(img_path_list, num_train, num_val)
            self.img_path_list = val_path_list
        print('finish loading AAPM-myo ' + mode + ' dataset, total images {}'.format(len(self.img_path_list)))
            
        # dataset
        self.dataset_shape = dataset_shape
        print('shape of the input data: {}'.format(self.dataset_shape))
        
        # init cttool
        self.cttool = CT_Preprocessing()

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        hu_array = np.load(img_path)
        width, height = hu_array.shape
        if width != self.dataset_shape or height != self.dataset_shape:
            hu_array = cv2.resize(hu_array, (self.dataset_shape, self.dataset_shape), cv2.INTER_CUBIC)
        img_miu = self.cttool.HU2miu(hu_array)
        img_miu = torch.from_numpy(img_miu).unsqueeze(0).float()
        return img_miu

    def __len__(self):
        return len(self.img_path_list)

    @staticmethod
    def simple_split(path_list, num_train=None, num_val=None):
        num_imgs = len(path_list)
        if num_train is None or num_val is None:
                num_train = int(num_imgs * 0.8)
                num_val = num_imgs - num_train
        if len(path_list) >= num_train + num_val:
            train_list = path_list[:num_train]
            val_list = path_list[-num_val:]
            print('dataset:{}, training set:{}, val set:{}'.format(len(path_list), len(train_list), len(val_list)))
        else:
            raise ValueError(f'aapm_myo dataset simple_split() error. num_imgs={num_imgs}, while num_train={num_train}, num_val={num_val}.')
        return train_list, val_list



if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import tqdm

    # [2] explicit_simple_split
    root_path = '/home/longzili/workspace/DATA/AAPM/cvpr_train'
    # image_list_path = './image_list.txt'
    aapm_dataset = AAPM_Myo_Dataset(root_path, 'val', dataset_shape=256, num_train=5000, num_val=410)
    print(aapm_dataset.__len__())
    val_loader = DataLoader(aapm_dataset, batch_size=1, num_workers=2)
    pbar = tqdm.tqdm(val_loader, ncols=60)
    for i, data in enumerate(pbar):
        pass
