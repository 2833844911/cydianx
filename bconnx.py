# 把pth模型转onnx类型


import torchvision.models as models
import torch
from torch import nn

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

with open('./classes.txt',encoding='utf-8') as f:
    t = f.read().split('\n')
alllb = len(t)
class mubModu(nn.Module):
    def __init__(self):
        super(mubModu, self).__init__()
        self.ks = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=150, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=150, out_channels=5, kernel_size=(3, 3), padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        d2 = self.ks(x)
        d2 = d2.permute(0, 2, 3, 1)
        d2 = d2.reshape((d2.shape[0], d2.shape[1], d2.shape[2], 1, 5))
        return d2


class flmodo(nn.Module):

    def __init__(self):
        super(flmodo, self).__init__()
        self.ret = models.resnet18(pretrained=False)
        # del self.ret.sgm
        self.ret.fc = nn.Linear(512, alllb)
        self.sgm = nn.Sigmoid()

    def forward(self,x):
        x = self.ret(x)
        x =self.sgm (x )
        return x


mymodo = torch.load('./mox2.pth', map_location=device)
mymodo.to(device)
mymodo.eval()

input_names = ['input']
output_names = ['output']

x = torch.randn(1, 3, 320, 192)

torch.onnx.export(mymodo, x, 'sbkuan.onnx', input_names=input_names, output_names=output_names, verbose='True')

mymod = torch.load('./flei3.pth',map_location=device)
mymod.eval()
x = torch.randn(1, 3, 80, 80)
input_names = ['input']
output_names = ['output']
torch.onnx.export(mymod, x, 'flei.onnx', input_names=input_names, output_names=output_names, verbose='True')


