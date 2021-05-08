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

from utils import *
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('-img','--img_path', type=str,default='./images/')
parser.add_argument('--trained_model', type=str,default='./weights/latest_model_8.pth')


def test():

  with torch.no_grad():
    net1=My_Classifier().to(device)
    net1.load_state_dict(torch.load(trained_model))
    net1.eval()
    cfg1 = set_config()
    cfg1.mode = "detect"
    img_list=glob.glob(img+'*') 
    cfg1.batch_size = len(img_list)
    dataset1 = vis_cat_dataloader(cfg1,img)
    data_loader1= DataLoader(dataset1,cfg1.batch_size,collate_fn= test_collate)

    step = 0
    L = nn.CrossEntropyLoss()

    correct = 0
    tot = 0
    avg_loss = 0
    for i,(images) in enumerate(data_loader1):
      if torch.cuda.is_available():
        images= images.to(device = device , dtype = torch.float ).cuda().detach()
      else:
        images = images.to(device = device , dtype = torch.float  ).detach()
      ######## Obtain Prediction by passing from model

      pred = net1(images)

    ########  Counting no. of correct classifications
      _, predicted = pred.max(1)
      correct_class=predicted[0]
      correct_class=cfg1.id_to_label[correct_class]

      print("Predicted Class is - ",correct_class)
      print(images.shape)
  
  
if  __name__ == "__main__":
  test()
