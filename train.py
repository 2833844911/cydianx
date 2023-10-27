import torch
from torch import nn
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# 将模型移动到GPU上（如果有可用的话）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图片划分为5x3去识别
needJj = [5,3]
# 图片压缩为的大小
tpxz = ( 192,320)

def euclidean_distance(p1, p2):
    '''
    计算两个点的欧式距离
    '''
    x1, y1 = p1
    x2, y2 = p2
    return torch.sqrt((x2-x1)**2 + (y2-y1)**2)
class BBox:
    def __init__(self, xe, ye, re, be,dd = 0):
        '''
        定义框，左上角及右下角坐标
        '''
        if dd == 1:
            self.x, self.y, self.r, self.b = xe, ye, re, be
        else:
            if re/2 >xe and 1==0:
                x = 0
            else:
                x = xe - re/2

            if be/2 > ye and 1==0:
                y = 0
            else:
                y = ye - be/2

            if xe + re/2 > 1 and 1==0:
                r = 1
            else:
                r = xe + re/2
            if ye + be / 2 > 1 and 1==0:
                b = 1
            else:
                b = ye + be / 2

            self.x, self.y, self.r, self.b = x, y, r, b

    def __xor__(self, other):
        '''
        计算box和other的IoU
        '''
        cross = self & other
        union = self | other
        return cross / (union + 1e-6)

    def __or__(self, other):
        '''
        计算box和other的并集
        '''
        cross = self & other
        union = self.area + other.area - cross
        return union

    def __and__(self, other):
        '''
        计算box和other的交集
        '''
        xmax = min(self.r, other.r)
        ymax = min(self.b, other.b)
        xmin = max(self.x, other.x)
        ymin = max(self.y, other.y)
        cross_box = BBox(xmin, ymin, xmax, ymax, 1)
        if cross_box.width <= 0 or cross_box.height <= 0:
            return 0
        return cross_box.area

    def boundof(self, other):
        '''
        计算box和other的边缘外包框，使得2个box都在框内的最小矩形
        '''
        xmin = min(self.x, other.x)
        ymin = min(self.y, other.y)
        xmax = max(self.r, other.r)
        ymax = max(self.b, other.b)
        return BBox(xmin, ymin, xmax, ymax, 1)

    def center_distance(self, other):
        '''
        计算两个box的中心点距离
        '''
        return euclidean_distance(self.center, other.center)

    def bound_diagonal_distance(self, other):
        '''
        计算两个box的bound的对角线距离
        '''
        bound = self.boundof(other)
        return euclidean_distance((bound.x, bound.y), (bound.r, bound.b))

    @property
    def center(self):
        return (self.x + self.r) / 2, (self.y + self.b) / 2

    @property
    def area(self):
        return self.width * self.height

    @property
    def width(self):
        return self.r - self.x  # + 1

    @property
    def height(self):
        return self.b - self.y  # + 1

with open('./classes.txt',encoding='utf-8') as f:
    t = f.read().split('\n')
alllb = len(t)
class getData(Dataset):
    def __init__(self):
        super().__init__()
        self.data = []
        path = './labels'
        for i in os.listdir(path):
            self.data.append(['./images/'+i.split('.')[0]+'.jpg', path+'/'+i])
        self.jk = len(self.data)
        self.tpcl = transforms.Compose([
            transforms.Resize(tpxz),
            transforms.ToTensor()
        ])
        self.alllb = alllb


    def pdisIn(self,x1, y1, x2, y2, x3, y3, x4, y4):
        if max(x1, x3) <= min(x2, x4) and max(y1, y3) <= min(y2, y4):
            return True
        else:
            return False

    def niou(self,rec1, rec2):
        left_column_max = max(rec1[0], rec2[0])
        right_column_min = min(rec1[2], rec2[2])
        up_row_max = max(rec1[1], rec2[1])
        down_row_min = min(rec1[3], rec2[3])

        # S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
        return S_cross / S2

    def __getitem__(self, item):
        dt = self.data[item]
        lp = open(dt[1], encoding='utf-8')
        kj = lp.read()
        lp.close()
        h = [ i.strip().split(' ') for i in kj.split('\n')]
        if len(h[-1]) <= 1:
            h.pop()
        for i in h:
            if len(i) == 1:
                continue
            for ig in range(len(i)):
                i[ig] = float(i[ig])
        imge = Image.open(dt[0]).convert('RGB')
        img = self.tpcl(imge).permute(0, 2,1)
        xz = 1 / needJj[0]
        yz = 1 / needJj[1]

        target = torch.zeros((needJj[0],needJj[1],9,6)).to(device)

        for x in range(needJj[0]):
            for i in range(needJj[1]):
                sj = [xz*x, yz*i, xz*x+xz, yz*i+yz]
                for ges,ko in enumerate(h):
                    ges = 0
                    kol = [ko[1]-ko[3]/2, ko[2]-ko[4]/2, ko[1]+ko[3]/2, ko[2]+ko[4]/2]
                    lpijk = self.niou([sj[0], sj[1], sj[2], sj[3]], [kol[0], kol[1], kol[2], kol[3]])
                    if self.pdisIn(sj[0], sj[1], sj[2], sj[3], kol[0], kol[1], kol[2], kol[3]) == True and lpijk>0.1 and lpijk > target[x,i,ges,4]:
                        target[x,i,ges,0] = ko[1]
                        target[x,i,ges,1] = ko[2]
                        target[x,i,ges,2] = ko[3]
                        target[x,i,ges,3] = ko[4]
                        target[x,i,ges,4] = lpijk
                        # target[x, i, ges, 5:] = 0
                        target[x, i, ges, 5] = 1
                        # target[x,i,ges,int(ko[0])+6] = 1
                    # else:
                    #     target[x, i, ges, 4:] = 0
                        # target[x,i,ges,5] = 0
                    # break


        return img.to(device), target


    def __len__(self):
        return self.jk




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



class mbLoss(nn.Module):
    def __init__(self):
        super(mbLoss, self).__init__()
        self.jcs = nn.BCEWithLogitsLoss()

    def CIoU(self,a, b):
        v = 4 / (torch.pi ** 2) * (torch.atan(a.width / a.height) - torch.atan(b.width / b.height)) ** 2
        iou =self.IoU(a, b)
        alpha = v / (1 - iou + v)
        return 1 - (self.DIoU(a, b) - alpha * v), iou

    def DIoU(self,a, b):
        d = a.center_distance(b)
        c = a.bound_diagonal_distance(b)
        return self.IoU(a, b) - (d ** 2) / (c ** 2)
    def IoU(self,a, b):
        return a ^ b

    def forward(self,out, target):

        allloss = 0
        zsd = 0
        jbb = 0
        qit = 0
        huhuh = 0
        jsq = 0
        for bash in range(target.shape[0]):
            for xwz in range(target.shape[1]):
                zxdwx = (1 / target.shape[1]) * xwz + (1 / target.shape[1]) / 2
                for ywz in range(target.shape[2]):
                    zxdwy = (1/target.shape[2])*ywz + (1/target.shape[2])/2
                    dt = out[bash, xwz, ywz, :, :]
                    for qub in range(target.shape[3]):
                        st = target[bash, xwz, ywz, qub,:]
                        if st[5] > 0.8:
                            for jk in range(dt.shape[0]):
                                a = BBox(st[0], st[1], st[2], st[3])
                                b = BBox((dt[jk][0]-0.5) + zxdwx, (dt[jk][1]-0.5) +zxdwy, (dt[jk][2]-0.5) + 1/target.shape[1], (dt[jk][3]-0.5) + 1/target.shape[2] )
                                los, iou =  self.CIoU(a, b)
                                allloss += los
                                jbb += iou
                                jsq += 1
                                zsd += (1- dt[jk][4]) ** 2
                                allloss += (1- dt[jk][4])
                                huhuh += (1- dt[jk][4])

                        else:
                            for jk in range(dt.shape[0]):
                                zsd += dt[jk][4] ** 2

                                qit += 1
                        break

        return allloss/ jsq + zsd / (jsq+qit), jbb/ jsq, huhuh / (jsq)





data = getData()

mymodo = mubModu()
# mymodo = torch.load('./mox2.pth')
mymodo.to(device)
meLoss = mbLoss()
optm = torch.optim.Adam(mymodo.parameters(),lr=0.0005)
maxLoss = 10000

for i in range(100):

    sx = 0
    cs = 0
    csl = 0
    zxdss = 0
    datae = DataLoader(data, shuffle=True, batch_size=5)

    datad = tqdm(datae)
    for img,tar in datad:
        optm.zero_grad()
        out = mymodo(img)
        loss, ub, zxd =meLoss(out, tar)
        loss.backward()
        optm.step()
        sx += loss.item()
        cs += ub.item()
        zxdss += zxd.item()
        csl += 1

        datad.set_description("训练 loss {} epch {} iou {} zxd {}".format(sx/csl, i, cs/csl, zxdss/csl))

    if sx/csl < maxLoss:
        torch.save( mymodo, './mox2.pth')
        print("保存模型===》")
        maxLoss =sx/csl




