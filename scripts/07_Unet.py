import torch
import torch.nn as nn

#reusable block for U_Net 2 convolution layer with 1 ReLU
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding =1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding =1)
        self.relu  = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x