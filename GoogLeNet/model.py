import torch
import torch.nn as nn
from torchsummary import summary

class Inception(nn.Module):
    def __init__(self,in_channels, ch1, ch3reduce, ch3, ch5reduce, ch5, pool_proj):
        super(Inception,self).__init__()

        self.block1 = nn.Conv2d(in_channels=in_channels,out_channels=ch1,kernel_size=1)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=ch3reduce,kernel_size=1),
            nn.Conv2d(in_channels=ch3reduce,out_channels=ch3,kernel_size=3,padding=1),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=ch5reduce,kernel_size=1),
            nn.Conv2d(in_channels=ch5reduce,out_channels=ch5,kernel_size=5,padding=2),
        )
        self.block4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1,padding=1),
            nn.Conv2d(in_channels=in_channels,out_channels=pool_proj,kernel_size=1),
        )
        self.Relu = nn.ReLU()

    def forward(self,x):
        path_1 = self.Relu(self.block1(x))
        path_2 = self.Relu(self.block2(x))
        path_3 = self.Relu(self.block3(x))
        path_4 = self.Relu(self.block4(x))

        #(batch_size,channels,height,weight)
        return torch.cat((path_1,path_2,path_3,path_4),1)

class GoogLeNet(nn.Module):
    def __init__(self,Inception):
        super(GoogLeNet,self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=1),nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=192,kernel_size=3,padding=1),nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.b2 = nn.Sequential(
            Inception(192,64,96,128,16,32,32),
            Inception(256,128,128,192,32,96,64),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.b3 = nn.Sequential(
            Inception(480,192,96,208,16,48,64),
            Inception(512,160,112,224,24,64,64),
            Inception(512,128,128,256,24,64,64),
            Inception(512,112,144,288,32,64,64),
            Inception(528,256,160,320,32,128,128),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.b4 = nn.Sequential(
            Inception(832,256,160,320,32,128,128),
            Inception(832,384,192,384,48,128,128),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

    def forward(self,x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GoogLeNet(Inception).to(device)
    print(summary(model, (3,224,224)))