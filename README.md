

train.py 可以自己配置的参数
```python
# 图片划分为5x3去识别
needJj = [5,3] # 需要识别的图标越小 这里就调节越大 例如 needJj = [6,4]
# 图片压缩为的大小
tpxz = ( 192,320) # 需要注意防止失真
```


可切换分类模型 (zzd.py中)
```python
    def shibie(self,imgpa):
        tp, imge = self.getimage(imgpa)
        d = time.time()
        kuane = self.mymodo.run(None, {self.mymodo.get_inputs()[0].name: tp})
        kuan = self.hetInfo(kuane) # 获得框的信息 去剪切物体
        # 下面可以不要了 然后换你的模型去识别物体

```

运行 train.py

运行 flmox.py

运行 bconnx.py 把pth转onnx


测试 运行zzd.py


视频教程地址:https://www.bilibili.com/video/BV1o84y197c8/?vd_source=93dc7417d09246b7259569c65e2eb085


需要226算法并发或者其他模型可以联系这位大佬wx:z596918978