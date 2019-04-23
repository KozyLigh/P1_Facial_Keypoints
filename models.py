## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        
        ## Shape of a Convolutional Layer
        # K - out_channels : the number of filters in the convolutional layer
        # F - kernel_size
        # S - the stride of the convolution
        # P - the padding
        # W - the width/height (square) of the previous layer
        
        # Since there are F*F*D weights per filter
        # The total number of weights in the convolutional layer is K*F*F*D
        
        ## self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
        # output size = (W-F)/S +1 = (224-3)/1 +1 = 222
        self.conv1 = nn.Conv2d(1, 32, 5) 
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3) 
        self.conv2_bn = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3) 
        self.conv3_bn = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, 3) 
        self.conv4_bn = nn.BatchNorm2d(256)
        
        self.conv5 = nn.Conv2d(256, 512, 3) 
        self.conv5_bn = nn.BatchNorm2d(512)
        
        #Dropouts
        
        self.drop25 = nn.Dropout(p=0.25)
        self.drop40 = nn.Dropout(p=0.40)
        
        #MaxPool
        self.pool22 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(12800, 2048) 
        self.fc1_bn = nn.BatchNorm1d(2048)        
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 68*2)  
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # Convolutional and pooling layers
        x = self.pool22(F.leaky_relu(self.conv1(x)))
        x = self.conv1_bn(x)
        x = self.drop25(x)
        x = self.pool22(F.leaky_relu(self.conv2(x)))
        x = self.conv2_bn(x)
        x = self.drop25(x)
        x = self.pool22(F.leaky_relu(self.conv3(x)))
        x = self.conv3_bn(x)
        x = self.drop25(x)
        x = self.pool22(F.leaky_relu(self.conv4(x)))
        x = self.conv4_bn(x)
        x = self.drop25(x)
        x = self.pool22(F.leaky_relu(self.conv5(x)))
        x = self.conv5_bn(x)
        x = self.drop25(x)

        # Fully-Connected layers
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc1_bn(x)
        x = self.drop40(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
