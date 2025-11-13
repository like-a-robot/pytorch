import torch
from torchsummary import summary
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self,in_channel,out_channel,flag=False,stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.Relu = nn.ReLU()
        if flag:
            self.conv3 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self,x):
        y = self.Relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)

        return self.Relu(x + y)

class ResNet(nn.Module):
    def __init__(self,Residual):
        super(ResNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        )
        self.block2 = nn.Sequential(
            Residual(in_channel=64, out_channel=64),
            Residual(in_channel=64, out_channel=64),
        )
        self.block3 = nn.Sequential(
            Residual(in_channel=64, out_channel=128,flag=True,stride=2),
            Residual(in_channel=128, out_channel=128),
        )
        self.block4 = nn.Sequential(
            Residual(in_channel=128, out_channel=256,flag=True,stride=2),
            Residual(in_channel=256, out_channel=256),
        )
        self.block5 = nn.Sequential(
            Residual(in_channel=256, out_channel=512,flag=True,stride=2),
            Residual(in_channel=512, out_channel=512),
        )
        self.block6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=512),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet(Residual).to(device)
    print(summary(model,(3,224,224)))



