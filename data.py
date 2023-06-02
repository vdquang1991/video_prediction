import os
import glob
import random
import torch
import numpy as np
import csv
from PIL import Image
import cv2
from torch.utils.data import DataLoader, Dataset

def get_data(csv_file):
    """Load our data from file."""
    with open(csv_file, 'r') as fin:
        reader = csv.reader(fin)
        data = list(reader)
    return data

def clean_data(data, CLIPS_LENGTH, MAX_FRAMES=3000):
    """Limit samples to greater than the sequence length and fewer
    than N frames. Also limit it to classes we want to use."""
    data_clean = []
    for item in data:
        if int(item[3]) >= CLIPS_LENGTH and int(item[3]) <= MAX_FRAMES:
            data_clean.append(item)
    return data_clean

def get_frames_for_sample(root_path, sample):
    """Given a sample row from the data file, get all the corresponding frame
    filenames."""
    path = os.path.join(root_path, sample[0], sample[1])
    folder_name = sample[2]
    images = sorted(glob.glob(os.path.join(path, folder_name + '/*jpg')))
    num_frames = sample[3]
    return images, int(num_frames)

def read_images(frames, start_idx, num_frames_per_clip):
    img_data = []
    for i in range(start_idx, start_idx + num_frames_per_clip):
        img = Image.open(frames[i])
        img = np.asarray(img)
        img_data.append(img)
    return img_data

def data_process(tmp_data, crop_size, is_train):
    img_datas = []
    if is_train and random.random()>0.5:
        flip = True
    else:
        flip = False

    for j in range(len(tmp_data)):
        img = Image.fromarray(tmp_data[j].astype(np.uint8))
        if img.width > img.height:
            scale = float(crop_size) / float(img.height)
            img = np.array(cv2.resize(np.array(img), (int(img.width * scale), crop_size))).astype(np.float32)
        else:
            scale = float(crop_size) / float(img.width)
            img = np.array(cv2.resize(np.array(img), (crop_size, int(img.height * scale)))).astype(np.float32)
        if j == 0:
            if is_train:
                crop_x = random.randint(0, int(img.shape[0] - crop_size))
                crop_y = random.randint(0, int(img.shape[1] - crop_size))
            else:
                crop_x = int((img.shape[0] - crop_size) / 2)
                crop_y = int((img.shape[1] - crop_size) / 2)
        img = img[crop_x:crop_x + crop_size, crop_y:crop_y + crop_size, :]
        img = np.array(cv2.resize(img, (crop_size, crop_size))).astype(np.float32)
        img = img / 255.
        if flip:
            img = np.flip(img, axis=1)

        img_datas.append(img)
    return img_datas

class VideoDataset(Dataset):
    def __init__(self, root_path, data, n_past=2, n_future=10, image_size=64, n_channel=3, is_train=True):
        super(VideoDataset, self).__init__()
        self.root_path = root_path
        self.data = data
        self.image_size = image_size
        self.n_past = n_past
        self.n_future = n_future
        self.n_channel = n_channel
        self.is_train = is_train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        frames, num_frames = get_frames_for_sample(self.root_path, row)
        start_idx = random.randint(0, num_frames - self.n_past - self.n_future)
        # print('start_idx / num_frames: = %d / %d at row %s' %(start_idx, num_frames, row))
        clip = read_images(frames, start_idx, self.n_past + self.n_future)
        clip = data_process(clip, self.image_size, is_train=self.is_train)
        clip = np.asarray(clip)
        if self.n_channel == 1:
            clip = clip[...,0]
            clip = np.resize(clip, new_shape=(clip.shape[0], clip.shape[1], clip.shape[2], 1))
        if self.is_train and random.random() > 0.5:
            clip = clip[::-1,...].copy()
        if self.is_train:
            return torch.from_numpy(clip)
        else:
            return (torch.from_numpy(clip), row, start_idx)




