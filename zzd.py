import cv2
from PIL import Image
import time
import onnxruntime as ort
import numpy as np



class getTpInfo():
    def __init__(self):
        with open('./classes.txt', encoding='utf-8') as f:
            t = f.read().split('\n')
        self.alllb = t
        self.mymod = ort.InferenceSession('./flei.onnx')
        self.mymodo = ort.InferenceSession('./sbkuan.onnx')

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
        return S_cross / (S2 + S1 - S_cross)

    def pdisIn(self, x1, y1, x2, y2, x3, y3, x4, y4):
        if max(x1, x3) <= min(x2, x4) and max(y1, y3) <= min(y2, y4):
            return True
        else:
            return False


    def hetInfo(self,out):
        out = out[0]
        lzx = 1 / out.shape[0]
        lzy = 1 / out.shape[1]
        kd = []

        for i in range(out.shape[0]):
            zxdwx = lzx * i + lzx / 2
            for i2 in range(out.shape[1]):
                zxdwy = lzy * i2 + lzy / 2
                for k in range(out.shape[2]):
                    if out[i, i2, k, 4] > 0.9:
                        zxx = (out[i, i2, k, 0] - 0.5) + zxdwx
                        zxy = (out[i, i2, k, 1] - 0.5) + zxdwy
                        zxk = (out[i, i2, k, 2] - 0.5) + lzx
                        zxg = (out[i, i2, k, 3] - 0.5) + lzy
                        l = [zxx - zxk / 2, zxy - zxg / 2,
                             zxx + zxk / 2, zxy + zxg / 2, out[i, i2, k, 4]]
                        isokk = 1
                        for idx, ds in enumerate(kd):
                            if self.bbbiou([l[0], l[1], l[2], l[3]], [ds[0], ds[1], ds[2], ds[3]]) < 0.1:
                                continue
                            else:
                                isokk = 0
                                if ds[4] < l[4]:
                                    kd[idx] = l
                        if isokk == 1:
                            kd.append(l)

        return kd

    def sbmox(self,kd, kuan):

        bhbhb = kuan.size
        x = bhbhb[0]
        y = bhbhb[1]

        for dt in kd:
            hvyx =abs( int(x * dt[0]))
            hvyy = abs( int(y * dt[1]))
            hvzx =abs(  int(x * dt[2]))
            hvzy = abs( int(y * dt[3]))
            if hvzx - hvyx <= 7:
                hvzx = hvyx + 8
            if hvzy - hvyy <= 7:
                hvzy = hvzy + 8

            dst = kuan.crop((hvyx, hvyy, hvzx, hvzy))
            dst = dst.resize((80, 80), Image.BICUBIC)
            dst = np.array(dst).astype(np.float32)/255
            g = dst.transpose(2,0,1).reshape((1,3,80,80))
            out = self.mymod.run(None, {self.mymodo.get_inputs()[0].name: g})[0]
            k = np.argmax(out)
            dt.append(k)
        return kd

    def getimage(self,path):
        imge = Image.open(path).convert('RGB')
        dst = imge.resize((320, 192), Image.BILINEAR)
        dst.save('./l.jpg')
        dst = np.array(dst).astype(np.float32) / 255
        img = dst.transpose(2, 1, 0).reshape((1, 3, 320, 192))
        return img, imge

    def shibie(self,imgpa):
        tp, imge = self.getimage(imgpa)
        d = time.time()
        kuane = self.mymodo.run(None, {self.mymodo.get_inputs()[0].name: tp})
        kuan = self.hetInfo(kuane)
        kuan = self.sbmox(kuan, imge)
        print("识别用时",time.time()-d, "秒")

        #
        tp = cv2.imread(imgpa)
        y = tp.shape[0]
        x = tp.shape[1]

        font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体

        for idx, i in enumerate(kuan):
            cv2.rectangle(tp, (int(i[0] * x), int(i[1] * y)), (int(i[2] * x), int(i[3] * y)), (0, 0, 255), 2)
            print(i[5], '===>', self.alllb[i[5]])
            cv2.putText(tp, '{}'.format(i[5]), (int(i[0] * x), int(i[1] * y)), font, 0.8, (0, 0, 0), 1)
        cv2.imshow("image", tp)
        cv2.imwrite('./1_.png', tp)
        cv2.waitKey(0)


if __name__ == '__main__':
    imgpa = r'F:\bz\dianx\images\3.jpg'
    s = getTpInfo()
    s.shibie(imgpa)






