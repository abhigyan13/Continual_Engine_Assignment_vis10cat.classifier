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

class set_config:

  def __init__(self):
    self.cuda=torch.cuda.is_available()
    self.lr=0.0009 
    self.test_path = '/content/drive/MyDrive/project_Abhigyan/image2/'
    self.batch_size = 8
    self.mode='train'
    self.resume= True
    self.path='/content/drive/MyDrive/project_Abhigyan/'
    self.epoch_ul = 15
    self.wpath= self.path+ 'weights/'
    self.hpath = self.path+'weights/history/'
    self.loss_path = '/content/drive/MyDrive/project_Abhigyan/weights/history'
    self.save_test = True
    self.label_to_id = { "AreaGraph" :  0 ,"BarGraph" :  1 ,"LineGraph" :  2 , "Map" :  3 , "ParetoChart" :  4 ,"PieChart" :  5 ,"RadarPlot" :  6 ,"ScatterGraph" :  7 , "Table" :  8 , "VennDiagram" :  9}
    self.id_to_label = {    0:"AreaGraph"  , 1 : "BarGraph" , 2 : "LineGraph" , 3 :  "Map" , 4 :  "ParetoChart" , 5 :  "PieChart" , 6 : "RadarPlot" ,7 : "ScatterGraph" , 8: "Table"  , 9 : "VennDiagram"}
    self.img_size = 256
    self.test_categ = None

class vis_cat_dataloader(data.Dataset):

  def __init__(self , config   , path = "/content/drive/MyDrive/project_Abhigyan/image/"   ):
    
    self.mode = config.mode
    self.size = config.img_size
    self.cfg = config


    if self.mode == "train":
      self.image_path = glob.glob( path+'*')
    if self.mode == "test":
      self.test_path = glob.glob( self.cfg.test_path+'*')
    if self.mode =="detect":
      self.t_path = glob.glob(path+'*')
  
  def __getitem__(self , index ):

    if self.mode == 'test':
      image = cv2.imread(self.test_path[index])
      categ = self.test_path[index].split("-")[-1].split(".")[0]
      y = self.cfg.label_to_id[categ]

    elif self.mode == 'train':

      image = cv2.imread(self.image_path[index])
      categ = self.image_path[index].split("-")[-1].split(".")[0]
      y = self.cfg.label_to_id[categ]  
    
    else:
      image = cv2.imread(self.t_path[index])
      y = 0

    image = cv2.resize(image , (self.size , self.size)) #Resize to Conv Imput Size
    image= image.astype( np.float64 )
    image/=255  #Normalize
    image = torch.tensor( image.transpose(2,0,1) )  #Convert to tensor
    

    return image , y  
  
  def __len__(self):
    if self.mode == "test":
      return len(self.test_path)
    elif self.mode == "train":
      return len(self.image_path)
    else:
      return len(self.t_path)

def train_collate(batch):

  img_list,y =[],[]
  for i,sample in enumerate(batch):
    img_list.append(torch.tensor(sample[0]) )
    y.append(torch.tensor(sample[1] ) )

  imgs=torch.stack(img_list)
  labels= torch.stack(y)

  return imgs , labels

def test_collate(batch):

  img_list =[]
  for i,sample in enumerate(batch):
    img_list.append(torch.tensor(sample[0]) )

  imgs=torch.stack(img_list)

  return imgs

def save_latest(net,opt  , step , path =  "/content/drive/MyDrive/project_Abhigyan/weights/"):
  weight = glob.glob(path+"lat*" )
  assert len(weight) <= 1, 'Error, multiple latest weight found.'
  if weight:
    os.remove(weight[0])
  
  optt = glob.glob(path+"opt*" )
  assert len(optt) <= 1, 'Error, multiple latest optimizer found.'
  if optt:
    os.remove(optt[0])

  print(f'\nSaving the latest model  as \'latest_{step}.pth\'.\n')
  torch.save(net.state_dict(), f'{path}latest_model_{step}.pth')
  torch.save(opt.state_dict(), f'{path}optimizer_model_{step}.pth')  


def save_loss_acc(lossv , accur ,  path = '/content/drive/MyDrive/project_Abhigyan/weights/history/'):

  loss_hist=glob.glob(path+'loss*')
  acc_hist= glob.glob(path+'acc*')
  assert len(loss_hist)<=1, "Multiple files of Loss History"
  assert len(acc_hist)<=1, "Multiple files of Accuracy History"
  if loss_hist:
    os.remove(loss_hist[0])
  if acc_hist:
    os.remove(acc_hist[0])
  
  open_file = open(path+"loss.pkl", "wb")
  pickle.dump(lossv , open_file)
  open_file.close()

  open_file = open(path+"accuracy.pkl", "wb")
  pickle.dump(accur , open_file)
  open_file.close()

