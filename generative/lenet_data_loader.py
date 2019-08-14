import torch.utils.data as data
import torchvision.transforms as transforms
import torch
from PIL import Image
import ntpath
import os
import numpy as np

class Dataset(data.Dataset):
    """
    data_dir : path to data containing subfolder of train or test
    mode : 'train' or 'test'

    """
    def __init__(self, data_dir, mode, image_size):
        self.data_dir = os.path.abspath(data_dir)
        self.mode = mode

        self.image_paths = os.listdir(os.path.join(self.data_dir, self.mode))

        self.image_paths = [os.path.join(self.data_dir, self.mode, p) for p in self.image_paths]
        self.image_paths = sorted(self.image_paths)
        self.data_size = len(self.image_paths)
        self.transform = transforms.Compose([
                                          transforms.Resize(image_size, Image.BICUBIC),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5),
                                                               (0.5, 0.5, 0.5))])


    def __getitem__(self, index):

        image_path = self.image_paths[index % self.data_size]
        image = Image.open(image_path).convert('L')
        image = self.transform(image)
        label = torch.from_numpy(np.array(int(ntpath.basename(image_path).split('_')[0])))

        if torch.cuda.is_available():
            image = image.to('cuda')
            label = label.to('cuda')

        return image, label

    def __len__(self):
        return self.data_size



