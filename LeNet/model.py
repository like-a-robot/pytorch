import torch
import torch.nn as nn
from torchsummary import summary

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16, kernel_size=5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features = 400,out_features=120)
        self.fc2 = nn.Linear(in_features = 120,out_features=84)
        self.fc3 = nn.Linear(in_features=84,out_features=10)
        self.Tanh = nn.Tanh()

    def forward(self,x):
        x = self.Tanh(self.conv1(x))
        x = self.pool(x)
        x = self.Tanh(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet()
    # print(summary(model,(1,32,32)))



