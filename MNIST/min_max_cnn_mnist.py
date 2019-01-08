import os
import re
import sys

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms
import torch.optim as optim

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_batch_size = 64
test_batch_size = 100

numepochs = 20

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=train_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=test_batch_size, shuffle=True)



class NormalCNN(nn.Module):
  def __init__(self):
    super(NormalCNN,self).__init__()
    self.conv1 = nn.Conv2d(1,64,5,1,2)
    self.conv2 = nn.Conv2d(64,64,5,1,2)
    self.conv3 = nn.Conv2d(64,64,5,1,2)
    self.pool = nn.MaxPool2d(kernel_size=3,stride=2)
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(2304,64)
    self.fc2 = nn.Linear(64,10)
  def forward(self,x):
    out = self.relu(self.conv1(x))
    out = self.pool(out)
    out = self.relu(self.conv2(out))
    out = self.pool(out)
    out = self.relu(self.conv3(out))
    out = out.view(-1,2304)
    out = self.relu(self.fc1(out))
    out = self.fc2(out)
    return F.log_softmax(out,dim=1)

normal_cnn = NormalCNN().to(device)



p = torch.rand(1,1,28,28).to(device)
outp = normal_cnn(p)
print(outp.shape)



def get_accuracy(net,loader):
  net.eval()
  correct = 0
  total = 0
  for _,(imgs,lbls) in enumerate(loader):
    imgs = imgs.to(device)
    lbls = lbls.to(device)
    
    pred = net(imgs)
    _,predictions = torch.max(pred,1)
    correct+=torch.sum(predictions==lbls).item()
    total+=imgs.size(0)
    
  return ((correct/total)*100)

normalcnn_optim = optim.Adam(normal_cnn.parameters())


normalcnnloss = []
normalcnn_trainaccuracy = []


normal_cnn.train()
for epoch in range(numepochs):
  avg_loss = 0.0
  count = 0
  normal_cnn.train()
  for _,(images,labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)
    normalcnn_optim.zero_grad()
    output = normal_cnn(images)
    loss = F.nll_loss(output,labels)
    avg_loss+=(loss.item())
    count+=1
    loss.backward()
    normalcnn_optim.step()
    
  avg_loss = avg_loss/count
  normalcnnloss.append(avg_loss)
  
  train_accuracy = get_accuracy(normal_cnn,train_loader)
  normalcnn_trainaccuracy.append(train_accuracy)
  print("Epoch {} Loss {} Train Accuracy {} ".format(epoch,loss,train_accuracy))
  
  
test_accuracy = get_accuracy(normal_cnn,test_loader)
print("Test Accuracy {} ".format(test_accuracy))


#---------------------MinMax Custom CNN Layer--------------------------------------#

class MinMaxCNNLayer(nn.Module):
  def __init__(self,infeatures,outfeatures,kernelsize,stride,paddinglength):
    super(MinMaxCNNLayer,self).__init__()
    self.infeatures = infeatures
    self.outfeatures = outfeatures
    self.kernelsize = kernelsize
    self.padding = paddinglength
    self.stride = stride

    self.cnn = nn.Conv2d(self.infeatures,self.outfeatures,self.kernelsize,self.stride,self.padding)
      
  def forward(self,x):
    conv1 = self.cnn(x)
    conv2 = (-1) * conv1
    conv3 = torch.cat((conv1,conv2),dim=1)
    return conv3


#---------------------------------------------------------------------------------------#

class MinMaxCNNNetwork(nn.Module):
  def __init__(self):
    super(MinMaxCNNNetwork,self).__init__()
    self.minmax1 = MinMaxCNNLayer(1,64,5,1,2)
    self.minmax2 = MinMaxCNNLayer(128,64,5,1,2)
    self.minmax3 = MinMaxCNNLayer(128,64,5,1,2)
    self.pool = nn.MaxPool2d(kernel_size=3,stride=2)
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(4608,64)
    self.fc2 = nn.Linear(64,10)
  def forward(self,x):
    out = self.relu(self.minmax1(x))
    out = self.pool(out)
    out = self.relu(self.minmax2(out))
    out = self.pool(out)
    out = self.relu(self.minmax3(out))
    out = out.view(-1,4608)
    out = self.relu(self.fc1(out))
    out = self.fc2(out)
    return F.log_softmax(out,dim=1)


minmaxcnn = MinMaxCNNNetwork().to(device)


q = torch.rand(1,1,28,28).to(device)
outq = minmaxcnn(p)
print(outq.shape)


minmaxcnn_optim = optim.Adam(minmaxcnn.parameters())


minmaxcnn_loss = []
minmaxcnn_trainacc = []


minmaxcnn.train()
for epoch in range(0,numepochs):
  avg_loss = 0.0
  count = 0
  minmaxcnn.train()
  for _,(images,labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)
    
    minmaxcnn_optim.zero_grad()
    
    outputs = minmaxcnn(images)
    _,preds = torch.max(outputs,1)
    
    loss = F.nll_loss(outputs,labels)
    avg_loss+=loss.item()
    count+=1
    
    loss.backward()
    minmaxcnn_optim.step()
    
  avg_loss = avg_loss/count
  minmaxcnn_loss.append(avg_loss)
  train_accuracy = get_accuracy(minmaxcnn,train_loader)
  minmaxcnn_trainacc.append(train_accuracy)
  print("Epoch {} Loss {} Train Accuracy {} ".format(epoch,avg_loss,train_accuracy))

  
test_accuracy = get_accuracy(minmaxcnn,test_loader)
print(test_accuracy)


import matplotlib.pyplot as plt

iterations = range(numepochs)

plt.plot(normalcnn_trainaccuracy,iterations,label='Normal CNN')
plt.plot(minmaxcnn_trainacc,iterations,label='Min-Max CNN')
plt.xlabel('Train Accuracy')
plt.ylabel('Epochs')
plt.legend()
plt.show()

plt.plot(normalcnnloss,iterations,label='Normal CNN')
plt.plot(minmaxcnn_loss,iterations,label='Min-Max CNN')
plt.xlabel('Train Loss')
plt.ylabel('Epochs')
plt.legend()
plt.show()
