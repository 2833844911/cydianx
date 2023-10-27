# 把pth模型转onnx类型


import torchvision.models as models
import torch
from torch import nn
if torch.cuda.is_available():
    if input("检测到可转gpu运行是否转(y/n):").strip() == 'y':
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
else:
    device = torch.device("cpu")

with open('./classes.txt',encoding='utf-8') as f:
    t = f.read().split('\n')
alllb = len(t)

class mubModu(nn.Module):
    def __init__(self):
        super(mubModu, self).__init__()
        self.ks = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=5, kernel_size=(3, 3), padding=1),
            nn.Sigmoid(),
        )


    def forward(self, x):
        d2 = self.ks(x)
        d2 = d2.permute(0, 2, 3, 1)
        d2 = d2.reshape((d2.shape[0], d2.shape[1], d2.shape[2], 1, 5))
        out = d2.squeeze(0)
        return out



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

x = torch.randn(1, 3, 320, 192).to(device)

torch.onnx.export(mymodo, x, 'sbkuan.onnx', input_names=input_names, output_names=output_names, verbose='True')

mymod = torch.load('./flei3.pth',map_location=device)
mymod.eval()
x = torch.randn(1, 3, 80, 80).to(device)
input_names = ['input']
output_names = ['output']
torch.onnx.export(mymod, x, 'flei.onnx', input_names=input_names, output_names=output_names, verbose='True')


