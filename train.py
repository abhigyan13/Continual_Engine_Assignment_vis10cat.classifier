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

def train():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  cfg = set_config()

  torch.autograd.set_detect_anomaly(True)
  net = My_Classifier().to(device)
  net.train()
  losses = []
  accuracies = []

  optimizer =optim.SGD(net.parameters(),lr=cfg.lr,momentum=0.9)

  ###### Load weights if resuming training else initialize 

  if cfg.resume:
    wtemp = glob.glob("/content/drive/MyDrive/project_Abhigyan/weights/latest*")
    opttemp = glob.glob("/content/drive/MyDrive/project_Abhigyan/weights/opt*")
    if torch.cuda.is_available():
        net_state_dict = torch.load(wtemp[0])
        optim_state_dict = torch.load(opttemp[0])
    else:
        net_state_dict = torch.load(path, map_location='cpu')
        optim_state_dict = torch.load(path, map_location = 'cpu')
    net.load_state_dict(net_state_dict)
    optimizer.load_state_dict(optim_state_dict)
    start_step = int(wtemp[0].split('.pth')[0].split('_')[-1])+1
    open_file=open(cfg.loss_path +'loss.pkl','rb')
    losses=pickle.load(open_file)
    open_file.close()

    open_file=open(cfg.loss_path +'accuracy.pkl','rb')
    accuracies=pickle.load(open_file)
    open_file.close()

    print(f'\nResume training with \'{start_step}\'.\n')

  else:
    
    net.init_weights()
    print(f'\nTraining from begining, weights initialized \n')
    start_step = 0 
    #cfg.resume = True

  dataset = vis_cat_dataloader( cfg )
  data_loader= DataLoader(dataset,cfg.batch_size,shuffle=True,collate_fn=train_collate)

  training = True
  time_last = time.time()
  step = start_step
  L = nn.CrossEntropyLoss()

  while training:
    correct = 0
    tot = 0
    avg_loss = 0
    for i,(images,targets) in enumerate(data_loader):
            
      if torch.cuda.is_available():
        images = images.to(device = device , dtype = torch.float ).cuda().detach()
        targets = targets.to(device = device , dtype = torch.long  ).cuda().detach()

      
    ######## Obtain Prediction by passing from model
      pred = net(images)

    ######## Compute cross entrop loss
      loss =  L( pred , targets )

    ######## A step of gradient upadate
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      time_this = time.time()
      batch_time = time_this-time_last
      
      avg_loss+= loss.item()

  ########  Counting no. of correct classifications
      _, predicted = pred.max(1)
      tot += targets.size(0)
      correct += predicted.eq(targets).sum().item()

      if i%8==0:
        print("Time for batch " , i , "/", len(data_loader) , "= " , batch_time , ", Loss  = " , loss.item() )
      time_last = time.time()
    
    avg_loss/=len(data_loader)
    accuracy =  100.*correct/tot
    accuracies.append(accuracy)
    losses.append(avg_loss)

    print("Epoch no ",step ,"  completed , Avg Loss = " , avg_loss ,  " Accuracy = " , accuracy  , " TEST Accuracy " , accuracy_test )
  
  ####### Save weights after epoch and loss history
    
    save_latest(net, optimizer ,  step = step)
    save_loss_acc(losses , accuracies)

    print('Showing loss plot')
    plt.plot(np.array(losses), 'r')
    plt.show()

    print('Showing accuracy plot')
    plt.plot(np.array(accuracies), 'r')
    plt.show()

    step+=1
    if step > cfg.epoch_ul:
      break


if  __name__ == "__main__":
  train()