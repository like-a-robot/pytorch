import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import torch.utils.data as Data
from model import LeNet
import torch.nn as nn
import torch.optim as optim
import copy
import time
import pandas as pd

def train_val_data_process():
    train_data = FashionMNIST(root='./data',
                              transform=transforms.Compose([transforms.Resize(size=32),
                                                           transforms.ToTensor()]),
                              train=True,
                              download=True)
    train_data,val_train = Data.random_split(train_data,[round(len(train_data)*0.8),round(len(train_data)*0.2)])
    train_loader = Data.DataLoader(dataset=train_data,
                                   batch_size=16,
                                   shuffle=True,
                                   num_workers=2)
    val_loader = Data.DataLoader(dataset=val_train,
                                   batch_size=16,
                                   shuffle=False,
                                   num_workers=2)
    return train_loader,val_loader

def train_model_process(model, train_loader, val_loader, epochs):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(),lr = 0.001)
    criterion = nn.CrossEntropyLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    since = time.time()

    train_accs = []
    train_losses = []
    val_accs = []
    val_losses = []

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))

        train_loss = 0.0
        train_acc = 0.0
        train_num = 0

        val_loss = 0.0
        val_acc = 0.0
        val_num = 0

        for step,(inputs,labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            model.train()
            outputs = model(inputs)
            pred = torch.argmax(outputs,dim=1)

            loss = criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_acc += (pred == labels).sum().item()
            train_num += inputs.size(0)

        for step,(inputs,labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            model.eval()
            outputs = model(inputs)
            pred = torch.argmax(outputs,dim=1)

            val_loss += criterion(outputs,labels).item() * inputs.size(0)
            val_acc += (pred == labels).sum().item()
            val_num += inputs.size(0)

        train_losses.append(train_loss / train_num)
        train_accs.append(train_acc / train_num)
        val_losses.append(val_loss / val_num)
        val_accs.append(val_acc / val_num)

        print("Train loss %f, Train acc %f" % (train_losses[-1], train_accs[-1]))
        print("Val loss %f, Val acc %f" % (val_losses[-1], val_accs[-1]))

        if val_accs[-1] > best_acc:
            best_acc = val_accs[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print("Training complete in {:.0f}m {:.0f}s\n".format(time_elapsed // 60, time_elapsed % 60))

    torch.save(best_model_wts,'./best_model_wts.pth')
    train_process = pd.DataFrame(data={
        "epoch":range(1,epochs+1),
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
    })

    return train_process

if __name__ == '__main__':
    model = LeNet()
    train_loader,val_loader = train_val_data_process()
    # print(len(train_loader),len(val_loader))
    train_process = train_model_process(model,train_loader,val_loader,epochs=10)
    print(train_process)

