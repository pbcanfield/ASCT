import torch.nn as nn 
import torch.nn.functional as F 

#Information to calculate 1D CNN output size and maxpool output size.
#https://towardsdatascience.com/pytorch-basics-how-to-train-your-neural-net-intro-to-cnn-26a14c2ea29
#Maxpool layer uses same resource but kernel size and stride are set to be the same
#unless otherwise specified. Padding set to 0 by default.
class SummaryCNN(nn.Module):
    def __init__(self, current_injections=1, summary_features=8): 
        super().__init__()

        self.__current_injections = current_injections

        # 1D convolutional layer
        #1024->1024 
        self.conv1 = nn.Conv1d(in_channels=current_injections, out_channels=1, kernel_size=4, padding=2,stride=1)
        # Maxpool layer that reduces 1024-> 128
        self.pool = nn.MaxPool1d(kernel_size=8, stride=8)
        # Fully connected layer taking as input the 6 flattened output arrays from the maxpooling layer
        self.fc = nn.Linear(in_features=128, out_features=summary_features) 

    def forward(self, x):
        #Reshape input to the right size. Inlcluding:
        #1) batch size,
        #2) Number of input channels.
        #3) Size of input.
        # -1 means to retain the first dimensions size (batch size).
        x = x.view(-1, self.__current_injections, 1024)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 128)
        x = F.relu(self.fc(x))
        return x
