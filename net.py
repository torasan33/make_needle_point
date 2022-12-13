import torch
import torch.nn.functional
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding = (1, 1))    
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding = (1, 1))    
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding = (1, 1))    
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding = (1, 1))    
        self.fc1 = nn.Linear(256*8*8, 2000)
        self.fc2 = nn.Linear(2000, 2000)
        self.fc3 = nn.Linear(2000, 3)

        self.dropout1 = torch.nn.Dropout2d(p=0.3)

        
    def forward(self, x):
        h = self.conv1_1(x)
        h = self.conv1_2(h)
        h = F.max_pool2d(h, kernel_size=(2,2), stride=2)
        h = F.relu(h)
        
        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h = F.max_pool2d(h, kernel_size=(2,2), stride=2)
        h = F.relu(h)

        h = h.view(-1, 256*8*8)

        h = self.fc1(h)
        h = self.fc2(h)
        h = self.dropout1(h)
        h = self.fc3(h)
        h = torch.softmax(h, dim=1)
        return h

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding = (1, 1))    
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding = (1, 1))    
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding = (1, 1))    
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding = (1, 1))    
        self.fc1 = nn.Linear(256*8*8, 3000)
        self.fc2 = nn.Linear(3000, 3000)
        self.fc3 = nn.Linear(3000, 3)

        self.dropout1 = torch.nn.Dropout2d(p=0.3)

        
    def forward(self, x):
        h = self.conv1_1(x)
        h = self.conv1_2(h)
        h = F.max_pool2d(h, kernel_size=(2,2), stride=2)
        h = F.relu(h)
        
        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h = F.max_pool2d(h, kernel_size=(2,2), stride=2)
        h = F.relu(h)

        h = h.view(-1, 256*8*8)

        h = self.fc1(h)
        h = self.fc2(h)
        h = self.dropout1(h)
        h = self.fc3(h)
        h = torch.softmax(h, dim=1)
        return h

class Deep_Net(nn.Module):
    def __init__(self):
        super(Deep_Net, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding = (1, 1))    
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding = (1, 1))    
        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding = (1, 1))    
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding = (1, 1))    
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding = (1, 1))    
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding = (1, 1))    
        self.fc1 = nn.Linear(128*4*4, 2000)
        self.fc2 = nn.Linear(2000, 2000)
        self.fc3 = nn.Linear(2000, 3)

        self.dropout1 = torch.nn.Dropout2d(p=0.3)

        
    def forward(self, x):
        h = self.conv1_1(x)
        h = self.conv1_2(h)
        h = F.max_pool2d(h, kernel_size=(2,2), stride=2)
        h = F.relu(h)
        
        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h = F.max_pool2d(h, kernel_size=(2,2), stride=2)
        h = F.relu(h)

        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = F.max_pool2d(h, kernel_size=(2,2), stride=2)
        h = F.relu(h)

        h = h.view(-1, 128*4*4)

        h = self.fc1(h)
        h = self.fc2(h)
        h = self.dropout1(h)
        h = self.fc3(h)
        h = torch.sigmoid(h)
        return h

class MHS_Net(nn.Module):
    #出力は針とその他のみ
    def __init__(self):
        super(MHS_Net, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3), padding = (1, 1))    
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding = (1, 1))    
        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding = (1, 1))    
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding = (1, 1))    
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding = (1, 1))    
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding = (1, 1))    
        self.fc1 = nn.Linear(128*4*4, 2000)
        self.fc2 = nn.Linear(2000, 2000)
        self.fc3 = nn.Linear(2000, 2)

        self.dropout1 = torch.nn.Dropout2d(p=0.3)

        
    def forward(self, x):
        h = self.conv1_1(x)
     
        h = self.conv1_2(h)
        h = F.max_pool2d(h, kernel_size=(2,2), stride=2)
        h = F.relu(h)
        
        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h = F.max_pool2d(h, kernel_size=(2,2), stride=2)
        h = F.relu(h)
        
        h = self.conv3_1(h) 
        h = self.conv3_2(h)
       
        h = F.max_pool2d(h, kernel_size=(2,2), stride=2)
        h = F.relu(h)        

        h = h.view(-1, 128*4*4)

        h = self.fc1(h)
        h = self.fc2(h)
        h = self.dropout1(h)
        h = self.fc3(h)
        h = torch.sigmoid(h)
        print(h)
        return h

class coor_Net(nn.Module):
    def __init__(self):
        super(coor_Net, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3,3), padding = (1, 1))    
        self.conv1_2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(3,3), padding = (1, 1))    
        self.conv2_1 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3,3), padding = (1, 1))    
        self.conv2_2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3,3), padding = (1, 1))    
        self.fc1 = nn.Linear(24*8*8+2, 2000)
        self.fc2 = nn.Linear(2000, 2000)
        self.fc3 = nn.Linear(2000, 2)

        self.dropout1 = torch.nn.Dropout2d(p=0.3)

        
    def forward(self, x, coor):
        h = self.conv1_1(x)
        h = self.conv1_2(h)
        h = F.max_pool2d(h, kernel_size=(2,2), stride=2)
        h = F.relu(h)
        
        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h = F.max_pool2d(h, kernel_size=(2,2), stride=2)
        h = F.relu(h)

        h = h.view(-1, 24*8*8)

        h = self.fc1(h)
        h = self.fc2(h)
        h = self.dropout1(h)
        h = self.fc3(h)
        h = torch.sigmoid(h)
        return h

