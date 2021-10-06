import torch.nn as nn 
import torch.nn.functional as F 
import torch
import numpy as np
from scipy import stats


#Information to calculate 1D CNN output size and maxpool output size.
#https://towardsdatascience.com/pytorch-basics-how-to-train-your-neural-net-intro-to-cnn-26a14c2ea29
#Maxpool layer uses same resource but kernel size and stride are set to be the same
#unless otherwise specified. Padding set to 0 by default.
class SummaryCNN(nn.Module):
    def __init__(self, current_injections=1, summary_features=8, hybrid=True): 
        super().__init__()

        self.__hybrid = hybrid
        self.__current_injections = current_injections

        # 1D convolutional layer
        #1024->1024 
        self.conv1 = nn.Conv1d(in_channels=current_injections, out_channels=5, kernel_size=9, padding=4,stride=1)
        # Maxpool layer that reduces 1024-> 512
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        #Second convolutional layer.
        #512 -> 512
        self.conv2 = nn.Conv1d(in_channels=5, out_channels=5, kernel_size=7, padding=3, stride=1)
        #Maxpool layer to reduce from 512 to 256. (Maybe try 128 here).
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        #Third convolutional layer.
        #256 -> 256
        self.conv3 = nn.Conv1d(in_channels=5, out_channels=5, kernel_size=5, padding=2, stride=1)
        #Maxpool layer to reduce from 256 to 128. (Maybe try 128 here).
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        #Fourth convolutional layer.
        #128 -> 64
        self.conv4 = nn.Conv1d(in_channels=5, out_channels=1, kernel_size=3, padding=1, stride=1)
        #Maxpool layer to reduce from 256 to 128. (Maybe try 128 here).
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully connected layer taking as input the 6 flattened output arrays from the maxpooling layer
        self.fc = nn.Linear(in_features=64, out_features=summary_features) 

    def forward(self, x):
        #Reshape input to the right size. Inlcluding:
        #1) batch size,
        #2) Number of input channels.
        #3) Size of input.
        # -1 means to retain the first dimensions size (batch size).
        x = x.view(-1, self.__current_injections, 1024)

        raw_data = x

        x = self.pool1(F.relu(self.conv1(x)))
        res = self.pool2(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = x + res
        res = self.pool3(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = x + res
        res = self.pool4(x)
        x = self.pool4(F.relu(self.conv4(x)))

        x = x.view(-1, 64)
        x = self.fc(x)

        #add mean, standard deviation, skew, 4th moment maybe
        #Caclulate each stat for each channel and concat into a vector
        #Then concat with final x.

        if self.__hybrid:
            x = torch.cat((x,self.calculate_hybrid_stats(raw_data)), 1)

        return x

    def calculate_hybrid_stats(self, x):
        data = x.numpy()
        mean     = np.mean(data,-1)
        variance = np.std(data,-1)
        skew     = stats.skew(data,-1)
       
        return torch.from_numpy(np.concatenate((mean,variance,skew), -1))
            