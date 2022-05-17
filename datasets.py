import paddle
import paddle.vision.transforms as T
import numpy as np
from PIL import Image


class ClassificationDatasets(paddle.io.Dataset):
    def __init__(self, datasets_path,img_size,mode='train'):
        if mode not in ['train', 'test', 'valid']: assert "Mode definition error"
        if mode == 'train':
            self.transforms = T.Compose([
                T.RandomResizedCrop(img_size), # 随机裁剪
                T.RandomHorizontalFlip(0.5),   # 随机水平翻转
                T.ToTensor(),                  # 数据的格式转换和标准化
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 归一化
            ])
        else:
            self.transforms = T.Compose([
                T.Resize(img_size),
                T.RandomCrop(img_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.data = []
        with open(datasets_path, 'r',encoding='utf-8') as f:
            for line in f.readlines():
                info = line.strip().split(';')
                if len(info) > 0:
                    self.data.append([info[0], info[1]])
            
    def __getitem__(self, index):
        label, image = self.data[index]
        image = Image.open(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = self.transforms(image)
        return image, np.array(label, dtype='int64')
    
    
    def __len__(self):
        return len(self.data)