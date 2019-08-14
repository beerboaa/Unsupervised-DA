import torch.utils.data as data
import torchvision.transforms as transforms
import torch
from PIL import Image
import ntpath
import os
import numpy as np

class Dataset(data.Dataset):
    """
    data_dir : path to data
    mode : 'train' or 'test'
    image_size : input size
    """
    def __init__(self, data_dir, mode):
        self.data_dir = os.path.abspath(data_dir)
        self.mode = mode

        self.image_paths = os.listdir(os.path.join(self.data_dir, self.mode))
        self.image_paths = [os.path.join(self.data_dir, self.mode, p) for p in self.image_paths]
        self.image_paths = sorted(self.image_paths)
        self.data_size = len(self.image_paths)
        if mode == 'train':
            self.transform = transforms.Compose([
                             transforms.RandomResizedCrop(224),
                             transforms.RandomHorizontalFlip(),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            self.transform = transforms.Compose([
                             transforms.Resize(224),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    def __getitem__(self, index):
        image_path = self.image_paths[index % self.data_size]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = torch.from_numpy(np.array(int(ntpath.basename(image_path).split('_')[0])))

        if torch.cuda.is_available():
            image = image.to('cuda')
            label = label.to('cuda')

        return image, label

    def __len__(self):
        return self.data_size