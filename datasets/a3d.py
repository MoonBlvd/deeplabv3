import torch
import os
import sys
import torch
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
sys.path.append('/u/bryao/work/Documents/deeplabv3/utils')
from PIL import Image
import cv2
import glob
from build_samplers import make_data_sampler, make_batch_data_sampler

import json
from tqdm import tqdm
import pdb

class A3DDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.W = 640 #1280
        self.H = 320 #720
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        img_root = '/home/data/vision7/A3D_2.0/frames/'
        self.samples = []
        all_img_folders = glob.glob(os.path.join(img_root, '*'))
        for img_folder in tqdm(all_img_folders):
            video_name = img_folder.split('/')[-1]
            all_files = sorted(glob.glob(os.path.join(img_folder, 'images', '*.jpg')))
            for idx, file in enumerate(all_files):
                self.samples.append((video_name, 
                                        idx, 
                                        file
                                        ))
            
                      
    def __getitem__(self, index):
        video_name, idx, file_name = self.samples[index]

        frame = Image.open(file_name)
        # resize
        frame = F.resize(frame, (self.H, self.W))
        frame = F.to_tensor(frame)
        # DeeplabV3, normalize
        frame = F.normalize(frame, mean=self.mean, std=self.std)
        return video_name, idx, frame

    def __len__(self):
        return len(self.samples)

def make_dataloader(root, 
                    shuffle=False, 
                    is_train=False,
                    distributed=False,
                    batch_per_gpu=1,
                    num_workers=0,
                    max_iters=10000):
    dataset = A3DDataset(root)

    sampler = make_data_sampler(dataset, shuffle=shuffle, distributed=distributed, is_train=is_train)
    batch_sampler = make_batch_data_sampler(dataset, 
                                            sampler, 
                                            aspect_grouping=False, 
                                            batch_per_gpu=batch_per_gpu,
                                            max_iters=max_iters, 
                                            start_iter=0, 
                                            dataset_name='A3D')

    dataloader =  DataLoader(dataset, 
                            num_workers=num_workers, 
                            batch_sampler=batch_sampler)

    return dataloader
