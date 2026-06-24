# dataloaders
# image data will be processed using dataloaders
# map data will be requested through an api

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import matplotlib.pyplot as plt

#

class ImageDataset(Dataset):
  def __init__(self):
    super().__init__()
    
    self.size = 0

    
  def __len__(self):
    return self.size
  
  def __getitem__(self, idx):
    # returns [X, Y]
    # X: Image Data 
    
    
  