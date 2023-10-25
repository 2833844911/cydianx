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
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=12, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=12, out_channels=5, kernel_size=(3, 3), padding=1),
            nn.Sigmoid(),
        )

    def bbbiou(self, rec1, rec2):
        if self.pdisIn(rec1[0], rec1[1], rec1[2], rec1[3], rec2[0], rec2[1], rec2[2], rec2[3]) == False:
            return 0
        left_column_max = max(rec1[0], rec2[0])
        right_column_min = min(rec1[2], rec2[2])
        up_row_max = max(rec1[1], rec2[1])
        down_row_min = min(rec1[3], rec2[3])

        S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
        return S_cross / (S2+S1-S_cross)
    def pdisIn(self,x1, y1, x2, y2, x3, y3, x4, y4):
        if max(x1, x3) <= min(x2, x4) and max(y1, y3) <= min(y2, y4):
            return True
        else:
            return False
    def forward(self,x):
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

x = torch.randn(1, 3, 320, 192)

torch.onnx.export(mymodo, x, 'sbkuan.onnx', input_names=input_names, output_names=output_names, verbose='True')

mymod = torch.load('./flei3.pth',map_location=device)
mymod.eval()
x = torch.randn(1, 3, 80, 80)
input_names = ['input']
output_names = ['output']
torch.onnx.export(mymod, x, 'flei.onnx', input_names=input_names, output_names=output_names, verbose='True')


