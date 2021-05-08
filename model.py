import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.autograd import Variable

import os
import argparse

import os
import time
from google.colab.patches import cv2_imshow
import pickle
import cv2
import numpy as np
import glob
import urllib
from urllib.request import URLopener
from skimage.transform import rotate


from PIL import Image
import shutil

dtype = torch.cuda.FloatTensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available()==False:
  dtype=torch.FloatTensor
from matplotlib import pyplot as plt

class My_Classifier(nn.Module):

  def __init__(self ,  mode = "train" ,  num_classes = 10  , input_channels = 3):
    super().__init__()
    self.mode = mode
    self.conv1=nn.Conv2d(input_channels,32,3,stride=2,padding=1 ) ## 128*128*32
    self.relu1=nn.ReLU()

    self.conv2=nn.Conv2d(32,32,3,stride=2,padding=1 )  ## 64*64*32
    self.bn2=nn.BatchNorm2d(32,momentum=0.5)
    self.relu2=nn.ReLU()
    
    self.conv3=nn.Conv2d(32,64,3,stride=2,padding=1 )  ## 32*32*64
    self.bn3=nn.BatchNorm2d(64,momentum=0.5)
    self.relu3=nn.ReLU()
    
    self.conv4=nn.Conv2d(64,128,3,stride=2,padding=1 )  ## 16*16*128
    self.bn4=nn.BatchNorm2d(128,momentum=0.5)
    self.relu4=nn.ReLU()
    
    self.conv5=nn.Conv2d(128,256,3,stride=2,padding=1 )  ## 8*8*256
    self.bn5=nn.BatchNorm2d(256,momentum=0.5)
    self.relu5=nn.ReLU()

    self.conv6=nn.Conv2d(256,256,3,stride=2,padding=1 )  ## 4*4*256
    self.bn6=nn.BatchNorm2d( 256 ,momentum=0.5)
    self.relu6=nn.ReLU()

    self.conv7=nn.Conv2d(256,512,3,stride=2,padding=1 )  ## 2*2*512
    self.bn7=nn.BatchNorm2d(512,momentum=0.5)
    self.relu7=nn.ReLU()

    self.linear = nn.Linear(2*2*512 , num_classes )


  def init_weights(self):
    for name, module in self.named_modules():
      if isinstance(module, nn.Conv2d ) or isinstance(module , nn.Linear ):
        nn.init.xavier_uniform_(module.weight.data)

        if module.bias is not None:
          module.bias.data.zero_()


  
  def forward(self , x ):
 
    out = self.conv1(x)
    out = self.relu1(out)
    
    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu2(out)
                    
    out = self.conv3(out)
    out = self.bn3(out)
    out = self.relu3(out)
    
    out = self.conv4(out)
    out = self.bn4(out)
    out = self.relu4(out)
    
    out = self.conv5(out)
    out = self.bn5(out)
    out = self.relu5(out)
                    
    out = self.conv6(out)
    out = self.bn6(out)
    out = self.relu6(out)
                    
    out = self.conv7(out)
    out = self.bn7(out)
    out = self.relu7(out)


    out = out.view(out.size(0), -1)
    out = self.linear(out)
    
    return out


