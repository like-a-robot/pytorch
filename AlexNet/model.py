import torch
from torch import nn
from torchsummary import summary

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        