# NOTE: Jan 5th, run inference on A3D 2.0
import os
root = '' # "/root/deeplabv3"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='1,3'
import sys
sys.path.append(os.path.join(root, "datasets"))
sys.path.append(os.path.join(root, "model"))
sys.path.append(os.path.join(root, "utils"))
from deeplabv3 import DeepLabV3
from utils import label_img_to_color
import cv2
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from a3d import make_dataloader

import time
from tqdm import tqdm

device = 'cuda'
save_dir = '/home/data/vision7/A3D_2.0/segmentations/'
root = '/home/data/vision7/A3D_2.0/'
batch_per_gpu = 20
num_workers = 20

model_id = 1
model = DeepLabV3(model_id, project_dir=root, is_train=False)
model.load_state_dict(torch.load(os.path.join("/u/bryao/work/Documents/deeplabv3/pretrained_models/model_13_2_2_2_epoch_580.pth")))
model = model.to(device)
model.eval()

val_dataloader = make_dataloader(root, 
                                shuffle=False, 
                                is_train=False,
                                distributed=False,
                                batch_per_gpu=batch_per_gpu,
                                num_workers=num_workers,
                                max_iters=None)

for iters, batch in enumerate(tqdm(val_dataloader)):
    video_name, ids, images = batch
    images = images.to(device)

    outputs = model(images)
    outputs = F.upsample(outputs, size=(720, 1280), mode="bilinear")
    outputs.softmax(dim=1)
    
    outputs = outputs.data.cpu().numpy()
    pred_label_imgs = np.argmax(outputs, axis=1) # (shape: (batch_size, 1024, 2048))
    pred_label_imgs = pred_label_imgs.astype(np.uint8)

#     viz = label_img_to_color(pred_label_imgs[0])
    for vid, idx, seg in zip(video_name, ids, pred_label_imgs):
        file_dir = os.path.join(save_dir, vid)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        file_name = os.path.join(file_dir, str(int(idx)).zfill(6)+'.png')
        cv2.imwrite(file_name, seg)
    
        
        