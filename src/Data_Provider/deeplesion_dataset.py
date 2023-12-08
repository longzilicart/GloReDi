# We use the first N data. However, we found redundancy with similar images per patient in each folder.
# 【TODO】 Hence, we recommend shuffling the dataset to enhance training diversity, which is crucial for smaller datasets.


import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np 
from PIL import Image
from datetime import datetime
import h5py
import glob
# import cv2
import sys
sys.path.append('..')
from utils.ct_tools import *



class Deep_Lesion_Dataset(Dataset):
    """Deeplesion Dataset Class
    Introduction: This class handles the Deeplesion dataset which includes CT images from 11 zip files (01 to 11). The 'img_list_path' includes 337898 images in total. From this, 40000 images are used for training and 1000 images are used for testing.

    Arguments:
        root_path (str): The root path where the Deeplesion data is stored.
        mode (str): Specifies whether the data will be used for 'train' or 'test'.

    Keyword Arguments:
        image_list_path (str): A predefined list that separates the train/test set. If not provided, it will collect all images.
        dataset_shape (tuple): Used to reshape the image.

    Returns:
        CT images in terms of miu (attenuation coefficient).
    """
    def __init__(self, root_path, mode, image_list_path=None, split=True, dataset_shape=512, num_train=40000, num_val=1000, train_rel_path = None, val_rel_path = None):
        assert mode in ['train', 'val', 'test']
        self.root_path = root_path
        # TODO it contains the first N data, recommand to shuffle the dataset
        if train_rel_path is None:
            train_rel_path = './Data_Provider/deeplesion_train_list.txt'
        if val_rel_path is None:
            val_rel_path = './Data_Provider/deeplesion_test_list.txt'
        if mode in ['train']:
            with open(train_rel_path, 'r') as f:
                img_path_list = [os.path.join(root_path, i.strip('\n')) for i in f]
            print('finish loading deeplesion train dataset, total images {}'.format(len(img_path_list)))
        elif mode in ['val', 'test']:
            with open(val_rel_path, 'r') as f:
                img_path_list = [os.path.join(root_path, i.strip('\n')) for i in f]
            print('finish loading deeplesion test dataset, total images {}'.format(len(img_path_list)))
        self.img_path_list = img_path_list
        # ++++ new version ++++

        # dataset
        self.dataset_shape = dataset_shape
        print('shape of the input data: {}'.format(self.dataset_shape))
        
        # init cttool
        self.cttool = CT_Preprocessing()

    def __getitem__(self, idx):
        img_path = self.img_path_list[idx]
        pil_image = Image.open(img_path)
        width, height = pil_image.size
        if width != self.dataset_shape or height != self.dataset_shape:
            pil_image = pil_image.resize((self.dataset_shape, self.dataset_shape))
        deeplesion_img = np.array(pil_image, dtype=np.float32) - 32768
        deeplesion_img[deeplesion_img>3072] = 3072 # follow dudonet preprocessing
        deeplesion_img[deeplesion_img<-1024] = -1024
        deeplesion_img_miu = self.cttool.HU2miu(deeplesion_img)
        deeplesion_img_miu = torch.from_numpy(deeplesion_img_miu).unsqueeze(0)
        return deeplesion_img_miu

    def __len__(self):
        return len(self.img_path_list)

    @staticmethod
    def simple_split(path_list, num_train=40000, num_val=1000):
        if len(path_list) > num_train + num_val:
            train_list = path_list[:num_train]
            val_list = path_list[-num_val:]
            print('dataset:{}, training set:{}, testing set:{}'.format(len(path_list), len(train_list), len(val_list)))
        else:
            raise ValueError('deeplesion dataset simple_split() error, list length is {}, pls split train val by hand'.format(len(path_list)))
        return train_list, val_list

    @staticmethod
    def explicit_simple_split(path_list):
        if len(path_list) > 41000:
            train_list = path_list[:40000]
            val_list = path_list[40001:40001+1000]
            print('dataset:{}, training set:{}, testing set:{}'.format(len(path_list), len(train_list), len(val_list)))
        else:
            raise ValueError('deeplesion dataset simple_split() error, list length is {}, pls split train val by hand'.format(len(path_list)))
        return train_list, val_list






if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import tqdm


    # [2] explicit_simple_split
    root_path = '/home/longzili/workspace/DATA/Deeplesion_Data'
    image_list_path = './image_list.txt'
    # deep_lesion_dataset = Deep_Lesion_Dataset(root_path, 'train', dataset_shape=256, image_list_path=image_list_path)
    # deep_lesion_dataset = Deep_Lesion_Dataset(root_path, 'val', dataset_shape=256, image_list_path=image_list_path)
    deep_lesion_dataset = Deep_Lesion_Dataset(root_path, 'test', dataset_shape=256, image_list_path=image_list_path)
    print(deep_lesion_dataset.__len__())
    val_loader = DataLoader(deep_lesion_dataset, batch_size=1,
                                  num_workers=2)
    pbar = tqdm.tqdm(val_loader, ncols = 60)
    for i, data in enumerate(pbar):
        # print(i)
        pass

