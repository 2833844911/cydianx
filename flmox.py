from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import torchvision.models as models
import torch
from torch import nn
import os
from PIL import Image
import random
from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# 将模型移动到GPU上（如果有可用的话）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open('./classes.txt',encoding='utf-8') as f:
    t = f.read().split('\n')
alllb = len(t)
dasdj = t

def change_color_channels(image):
    # 在这里实现你的颜色通道转换逻辑
    # 假设输入图像为RGB格式，将红色和绿色通道交换
    if random.random() <0.2:
        r, g, b = image.split()
        image = Image.merge("RGB", (g, r, b))
    return image

class getData(Dataset):
    def __init__(self):
        super(getData, self).__init__()
        data = []
        path = './labels'
        for i in os.listdir(path):
            data.append(['./images/' + i.split('.')[0] + '.jpg', path + '/' + i])
        self.data = []
        for i in data:
            with open(i[1],encoding='utf-8') as f:
                dg = f.read()
                h = [i.strip().split(' ') for i in dg.split('\n')]
                if len(h[-1]) <= 1:
                    h.pop()
                for ik in h:
                    self.data.append([i[0], int(ik[0]), float(ik[1]), float(ik[2]), float(ik[3]), float(ik[4])])
        self.tpcl = transforms.Compose([
            # transforms.Resize( (320, 192)),
            transforms.Lambda(change_color_channels),
            transforms.ToTensor()
        ])
        self.len = len(self.data)
        self.alllb = alllb

    def __getitem__(self, item):
        dt = self.data[item]
        dst = Image.open( dt[0]).convert('RGB')
        # ko =  (320, 192)
        ko = dst.size
        # dst = self.tpcl(dst)
        hvyx= abs(int(ko[0] * (dt[2] - dt[4]/2)))
        hvyy= abs(int(ko[1] * (dt[3] - dt[5]/2)))
        hvzx= abs(int(ko[0] * (dt[2] + dt[4]/2)))
        hvzy= abs(int(ko[1] * (dt[3] + dt[5]/2)))
        # print(hvyx, hvyy, hvzx,hvzy)
        if hvzx - hvyx <=9:
            hvzx = hvyx +9
        if hvzy - hvyy <= 9:
            hvzy = hvzy + 9
        dst = dst.crop((hvyx, hvyy, hvzx, hvzy))
        dst = dst.resize((80, 80), Image.BICUBIC)
        dst = self.tpcl(dst)
        # dst.save(open('./k.jpg', 'wb'))
        # dads = torch.tensor([0]*self.alllb, dtype=torch.float)
        # dads[dt[1]] = 1
        dads = dt[1]


        # print(hvyx, hvyy, hvzx, hvzy)
        # print(dst[:, hvyx:hvzx, hvyy:hvzy].shape)
        return dst, dads


    def __len__(self):
        return self.len



class flmodo(nn.Module):
    def __init__(self):
        super(flmodo, self).__init__()
        self.ret = models.resnet18(pretrained=False)
        # del self.ret.sgm
        self.ret.fc = nn.Linear(512, alllb)

        self.ret.train()
        self.sgm = nn.Sigmoid()

    def forward(self,x):
        c =self.sgm ( self.ret(x))
        return c

mymod = flmodo()

# mymod = torch.load('./flei3.pth',map_location=device)
mymod.to(device)
sj = getData()
myloss = nn.CrossEntropyLoss()
optm = torch.optim.Adam(mymod.parameters(),lr=0.0005)

moll = 100000
dasdr =  nn.Softmax(dim=1)
for _ in range(100):
    dtr = DataLoader(sj, batch_size=100, shuffle=False)
    dtr = tqdm(dtr)

    allloss = 0
    csl = 0
    zka = 0
    mymod.train()
    for out, tar in dtr:
        out = out.to(device)
        tar = tar.to(device)
        optm.zero_grad()
        x = mymod(out)
        x =dasdr(x)
        loss = myloss(x, tar)
        fsda = loss.backward()
        optm.step()

        # zk = torch.sum(torch.argmax(x,dim = 1) == torch.argmax(tar,dim = 1))/out.shape[0]
        zk = torch.sum(torch.argmax(x,dim = 1) == tar)/out.shape[0]
        zka += zk.item()
        csl += 1
        allloss += loss.item()
        dtr.set_description("训练 loss {} acc {} epch {}".format(allloss / csl,zka / csl, _))
    if allloss/csl < moll:
        torch.save( mymod, './flei3.pth')
        print("保存模型===》")
        moll =allloss/csl

