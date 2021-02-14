import torch
import torch.nn.functional as f
import torch.nn as nn

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__() 

        self.num_actions = 8
        self.num_accepted = 30
        self.num_channels = self.num_actions+self.num_accepted+1
        

        self.conv2d = nn.Conv2d(
            in_channels = (50, 50, self.num_channels), 
            out_channels = (1, 50, self.num_channels), 
            kernel_size = (50, 1, 1))
        self.conv2d_1 = nn.Conv2d(
            in_channels = (50, 50, self.num_channels), 
            out_channels = (1, 48, self.num_channels), 
            kernel_size = (50, 2, 1))
        self.maxpool1d = nn.MaxPool1d(
            kernel_size = (1, 50, 1), 
            stride = 1)
        self.linear = nn.Linear(
            in_features = self.num_channels, 
            out_features = 8)
        self.softmax = nn.Softmax(
            name = Softmax, 
            dim = self.num_actions)

    def forward(self, input):
        conv2d_output = self.conv2d(input)
        conv2d_output = f.relu_(conv2d_output)
        conv2d_1_output = self.conv2d_1(input)
        conv2d_1_output = f.relu_(conv2d_1_output)
        maxpool1d_input = torch.cat((conv2d_output, conv2d_1_output), dim=0)
        maxpool1d_output = self.maxpool1d(maxpool1d_input)
        linear_output = self.linear(maxpool1d_output)
        linear_output = f.relu_(linear_output)
        softmax_output = self.softmax(linear_output)
        
        return softmax_output
    