import torch
from torch import nn
from torch.nn import Dropout
from torchsummary import summary

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4,padding=2)
        self.pool = nn.MaxPool2d(kernel_size=3,stride = 2)
        self.conv2 = nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2)
        self.conv3 = nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=256 * 6 * 6,out_features=4096)
        self.fc2 = nn.Linear(in_features=4096,out_features=4096)
        self.fc3 = nn.Linear(in_features=4096,out_features=10)
        self.Relu = nn.ReLU()
        self.Dropout = nn.Dropout(p=0.5)


    def forward(self,x):
        x = self.Relu(self.conv1(x))
        x = self.pool(x)
        x = self.Relu(self.conv2(x))
        x = self.pool(x)
        x = self.Relu(self.conv3(x))
        x = self.Relu(self.conv4(x))
        x = self.Relu(self.conv5(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.Dropout(x)
        x = self.Relu(self.fc1(x))
        x = self.Dropout(x)
        x = self.Relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    print(summary(model,(1,224,224)))
