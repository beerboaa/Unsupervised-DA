import torch.utils.data as data
import torchvision.transforms as transforms
import torch
from PIL import Image
import ntpath
import os
import numpy as np
import random

class TrainDataset(data.Dataset):
    """
    data_dir : path to data
    direction : AtoB etc.
    image_size : input size
    """
    def __init__(self, sub_dir, opt):
        self.data_dir = os.path.abspath(opt.data_dir)
        self.s_dir = sub_dir
        self.image_size = opt.image_size

        # a list of all image paths
        self.s_image_paths = os.listdir(os.path.join(self.data_dir, self.s_dir))
        self.s_image_paths = [os.path.join(self.data_dir, self.s_dir, p) for p in self.s_image_paths]
        self.s_image_paths = sorted(self.s_image_paths)
        self.s_data_size = len(self.s_image_paths)

        temp = os.listdir(self.data_dir)
        self.t_dir = temp[len(temp) - temp.index(self.s_dir) - 1]
        self.t_image_paths = os.listdir(os.path.join(self.data_dir, self.t_dir))
        self.t_image_paths = [os.path.join(self.data_dir, self.t_dir, p) for p in self.t_image_paths]
        self.t_image_paths = sorted(self.t_image_paths)
        self.t_data_size = len(self.t_image_paths)

        self.data_size = max(self.s_data_size, self.t_data_size)

        # pre-processing for training
        if opt.image_size == 224:
            self.transform = transforms.Compose([
                             transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            self.transform = transforms.Compose([
                             transforms.Resize(opt.image_size, Image.BICUBIC),
                             transforms.ToTensor(),
                             transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

    def __getitem__(self, index):
        if self.s_data_size >= self.t_data_size:
            s_image_path = self.s_image_paths[index % self.data_size]
            t_image_path = self.t_image_paths[random.randint(0, self.t_data_size - 1)]
        else:
            s_image_path = self.s_image_paths[random.randint(0, self.s_data_size - 1)]
            t_image_path = self.t_image_paths[index % self.data_size]

        if self.image_size == 224:
            s_image = Image.open(s_image_path).convert('RGB')
            t_image = Image.open(t_image_path).convert('RGB')
        else:
            s_image = Image.open(s_image_path).convert('RGB')
            t_image = Image.open(t_image_path).convert('RGB')
            # s_image = Image.open(s_image_path).convert('L')
            # t_image = Image.open(t_image_path).convert('L')

        s_image = self.transform(s_image)
        s_label = torch.from_numpy(np.array(int(ntpath.basename(s_image_path).split('_')[0])))
        t_image = self.transform(t_image)
        t_label = torch.from_numpy(np.array(int(ntpath.basename(t_image_path).split('_')[0])))

        return s_image, s_label, t_image, t_label

    def __len__(self):
        return max(self.s_data_size, self.t_data_size)

class TestDataset(data.Dataset):
    def __init__(self, sub_dir, opt):
        self.data_dir = os.path.abspath(opt.data_dir)
        self.sub_dir = sub_dir
        self.image_size = opt.image_size

        # a list of all image paths
        self.image_paths = os.listdir(os.path.join(self.data_dir, self.sub_dir))
        self.image_paths = [os.path.join(self.data_dir, self.sub_dir, p) for p in self.image_paths]
        self.image_paths = sorted(self.image_paths)
        self.data_size = len(self.image_paths)

        # pre-processing for training
        if opt.image_size == 224:
            self.transform = transforms.Compose([
                             transforms.Resize(224),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            self.transform = transforms.Compose([
                             transforms.Resize(opt.image_size, Image.BICUBIC),
                             transforms.ToTensor(),
                             transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

    def __getitem__(self, index):
        image_path = self.image_paths[index % self.data_size]
        if self.image_size == 224:
            image = Image.open(image_path).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
            # image = Image.open(image_path).convert('L')

        image = self.transform(image)
        label = torch.from_numpy(np.array(int(ntpath.basename(image_path).split('_')[0])))

        return image, label

    def __len__(self):
        return self.data_size