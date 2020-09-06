import torch
import pickle
import matplotlib as plt
from tdnn import *
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import os

path = os.getcwd()

with open('test1.pkl','rb') as f:
    X,y = pickle.load(f)
num_epoch = 100
batch = 10   
trainloader = DataLoader(list(zip(X,y)), shuffle=False, batch_size=batch)
net = nn.Sequential()
net.add_module('frame1', TDNN(input_dim=24, output_dim=512, context_size=5, dilation=1))#layer 1
net.add_module('frame2', TDNN(input_dim=512, output_dim=512, context_size=3, dilation=2))#layer 2
net.add_module('frame3', TDNN(input_dim=512, output_dim=512, context_size=3, dilation=3))#layer 3
net.add_module('frame4', TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1))#layer 4
net.add_module('frame5', TDNN(input_dim=512, output_dim=1500, context_size=1, dilation=1))#layer 5
net.add_module('pool',StatsPooling())#pooling
net.add_module('segment6',Segment(input_dim=3000, output_dim=512))#segment1
net.add_module('segment7',Segment(input_dim=512, output_dim=512))#segment2
net.add_module('softmax',Softmax(input_dim = 512, output_dim = len(y[0])))#softmax
 

if os.path.exists('trained_net.pth'):
    net.load_state_dict(torch.load('trained_net.pth'))
    
    
for epoch in range(1,num_epoch+1):
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        input_, label = data
        #inputs, labels = inputs.cuda(), labels.cuda()
        #output= net(inputs).cuda()
        inputs = input_.type(torch.float)
        inputs.requires_grad_(True)
        labels = label.type(torch.float)
        output = net(inputs)
        loss = nn.MSELoss(reduction='sum')
        l = (loss(output,labels))
        optimizer = optim.Adam(net.parameters(), lr=0.005)
        net.zero_grad()
        l.backward()
        optimizer.step()
        print('epoch %d, loss: %f, batch_size = %d' % (epoch, l.item(),batch))
    if (epoch % 50) ==0:
        torch.save(net.state_dict(),'trained_net.pth')
        print('model saved, epoch: %d'%(epoch))
        
        
        
