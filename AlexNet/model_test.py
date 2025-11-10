import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from model import AlexNet

def test_data_process():
    test_train = FashionMNIST(root='./data',
                              transform=transforms.Compose([transforms.Resize(224),
                                                            transforms.ToTensor()]),
                              train=False,
                              download=True)
    test_loader = DataLoader(dataset=test_train,
                             batch_size=16,
                             shuffle=True,
                             num_workers=2)
    return test_loader

def test_model_process(model,test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_acc = 0
    test_num = 0

    with torch.no_grad():
        for step, (input, labels) in enumerate(test_loader):
            input = input.to(device)
            labels = labels.to(device)

            model.eval()
            output = model(input)
            pred = torch.argmax(output,dim=1)

            test_acc += (pred == labels).sum().item()
            test_num += labels.size(0)

    test_acc = test_acc / test_num
    return test_acc

if __name__ == '__main__':
    model = AlexNet()
    model.load_state_dict(torch.load("./best_model_wts.pth"))

    test_loader = test_data_process()
    test_acc = test_model_process(model,test_loader)
    print("model accuracy is %f" % test_acc)


