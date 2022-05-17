
import paddle
from paddle.static import InputSpec
from datasets import ClassificationDatasets
import matplotlib.pyplot as plt
import numpy as np
from utils.utils import get_classes, show_img
from net import nets
from PIL import Image
import paddle.vision.transforms as T
from tqdm import tqdm


class Classification(object):
    _defaults = {
        "model_path": "./model_data/mobilenetv2", # 权重路径
        "classes_path": "./datasets/test", # 数据集路径
        "input_shape": [224, 224],  # 输入图片大小
        "backbone": "MobileNetv2", # 网络结构
    }
    
    
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
        
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        #---------------------------------------------------#
        #   获得种类
        #---------------------------------------------------#
        self.num_classes, self.class_names = get_classes(self.classes_path)
        # 使用gpu
        paddle.device.set_device('gpu')
        self.transforms = T.Compose([
            T.Resize(224),
            T.RandomCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.generate()
        
    def generate(self):
        self.model = nets[self.backbone](class_num=self.num_classes)
        self.model = paddle.Model(self.model, inputs=[InputSpec(shape=[3, self.input_shape[0], self.input_shape[1]], dtype='float32', name='image')])
        self.model.load(self.model_path)
        self.model.prepare()
        
    def detect_image(self, img):
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        image = np.expand_dims(np.array(self.transforms(img)), 0).astype(np.float32)
        # image = np.transpose(np.expand_dims(np.array(image)/255.0, 0), [0, 3, 1, 2]).astype(np.float32)
        # print(image)
        r = self.model.predict_batch([image])
        pred = self.class_names[np.argmax(r)]
        show_img(img, pred)
        return pred
    
    def detect_datasets(self, datasets_path, is_show=False):
        
        test_datasets = ClassificationDatasets(datasets_path, self.input_shape[0], mode='test')
        random_indexs = np.random.randint(len(test_datasets), size=3)
        r = self.model.predict(test_datasets)
        for idx in random_indexs:
            img = test_datasets[idx][0]
            pred = np.argmax(r[0][idx])
            r_label = test_datasets[idx][1]
            print('样本{}, 真实：{}, 预测：{}'.format(idx, self.class_names[r_label], self.class_names[pred]))
            if is_show:
                show_img(img, pred, r_label)
    
    def eval(self, datasets_path, mode='top1_acc'):
        if mode not in ['top1_acc', 'top5_acc']: assert "Mode definition error"
        test_datasets = ClassificationDatasets(datasets_path, self.input_shape[0], mode='test')
        indexs = len(test_datasets)
        r = self.model.predict(test_datasets)
        s = 0
        if mode == 'top1_acc':
            for idx in tqdm(range(indexs), desc='进行中', ncols=100):
                pred = np.argmax(r[0][idx])
                r_label = test_datasets[idx][1]
                if r_label == pred:
                    s += 1
            return s/indexs
        else:
            for idx in tqdm(range(indexs), desc='进行中', ncols=100):
                idxs = np.argsort(r[0][idx])[::-1][:5]
                r_label = test_datasets[idx][1]
                if r_label in idxs:
                    s += 1
            return s/indexs

            
        
        
        
        
        

    

    
        
    
