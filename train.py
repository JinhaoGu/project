import torch
import pickle
#import matplotlib as plt
from tdnn import *
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import sys



path = os.getcwd()
batch = int(sys.argv[1])
l_rate = float(sys.argv[2])


with open('test1.pkl','rb') as f:
    X,y = pickle.load(f)
num_epoch = 100
 
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
 

if os.path.exists('trained_net103.pth'):
    net.load_state_dict(torch.load('trained_net103.pth'))



# basic_optim = optim.SGD(net.parameters(),lr = 1e-5)  
# optimizer = ScheduledOptim(basic_optim)  
# lr_mult = (1 / 1e-5) ** (1 / 100)
# lr = []
# losses = []
# best_loss = 1e9

for epoch in range(1,num_epoch+1):
    for i, data in enumerate(trainloader, 0):
        start = time.time()
        # get the inputs; data is a list of [inputs, labels]
        input_, label = data
        #inputs, labels = inputs.cuda(), labels.cuda()
        #output= net(inputs).cuda()
        inputs = input_.type(torch.float)
        inputs.requires_grad_(True)
        labels = label.type(torch.float)
        output = net(inputs)
        #loss = nn.MSELoss(reduction='sum')
        l = torch.sum(output*labels)/(batch)#(loss(output,labels))
        optimizer = optim.Adam(net.parameters(), lr=l_rate)
        net.zero_grad()
        with torch.autograd.detect_anomaly():
             l.backward()
    
        optimizer.step()
        end = time.time()
        # lr.append(optimizer.lr)
        # losses.append(loss.data[0])
        # optimizer.set_learning_rate(optimizer.lr * lr_mult)
        runtime = end-start
        print('epoch %d, batch_size = %d,iteration: %d, loss: %f, runtime: %f s, lr=%f' % (epoch,batch ,i+1,l.item(),runtime,l_rate))
        # if loss.data[0] < best_loss:
        #     best_loss = loss.data[0]
        # if loss.data[0] > 4 * best_loss or optimizer.learning_rate > 1.:
        #     break
    if l <=0.01:
        break
    if (epoch % 10) ==0:
        torch.save(net.state_dict(),"".join(('trained_net',str(epoch+3),'.pth')))
        print('model saved, epoch: %d'%(epoch))

torch.save(net.state_dict(),'final_net.pth')
print('final model saved')
print('========================')


 
# plt.figure()
# plt.xticks(np.log([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]), (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1))
# plt.xlabel('learning rate')
# plt.ylabel('loss')
# plt.plot(np.log(lr), losses)
# plt.show()
# plt.figure()
# plt.xlabel('num iterations')
# plt.ylabel('learning rate')
# plt.plot(lr)     
        
