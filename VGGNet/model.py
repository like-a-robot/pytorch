import torch
from torch import nn
from torchsummary import summary

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet,self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3, padding=1),nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3, padding=1),nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3, padding=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3, padding=1),nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, padding=1),nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3, padding=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, padding=1),nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1),nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1),nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1),nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, padding=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=7 * 7 * 512 , out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=10),
        )
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGGNet().to(device)
    # print(summary(model, (3,224,224)))