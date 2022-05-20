# 图像分类——Paddle实现

## 1. 环境配置

- paddlepaddle-gpu==2.3.0
- opencv-python==4.5.5.64
- tqdm==4.64.0
- numpy==1.19.3
- Pillow==9.1.0

## 2. 数据准备

数据集链接：https://pan.baidu.com/s/1zs9U76OmGAIwbYr91KQxgg
提取码：bhjx

数据文件夹格式如下：
```python
- datasets
	- test
		- flower0
		- flower1
		- flower2
		- flower3
		- ...
	- train 
		- flower0
		- flower1
		- flower2
		- flower3
		- ...
```
运行txt_annotation.py文件后，会在根目录下生成两个txt文件，分别为cls_test.txt以及cls_train.txt。


## 3. 训练步骤
1. 在train.py中实现，需要指定的参数如下：
```python
# ------------------------------------------------ #
# -----------------参数说明------------------------ #
# input_shape：输入图片大小
# epochs：训练轮次
# batch_size：训练批次
# train_path：训练集路径
# valid_path：测试集路径
# loss：选择的loss函数CE代表交叉熵损失、Poly代表Poly交叉熵损失
# network：网络结构的选择{MobileNetv2}
# model_path：预训练权重路径，则model_path=""，注意：不需要指定后缀名
# ------------------------------------------------ #
input_shape = [224, 224, 3]
epochs = 100
batch_size = 16
train_path = './cls_train.txt'
valid_path = './cls_test.txt'
loss = 'Poly'
network = "MobileNetv2"
model_path = ""
```

## 4. 预测步骤

**注意：需要数据准备之后进行此步骤**
1. 首先在classification.py文件中，指定参数，如下：

```python
_defaults = {
        "model_path": "./model_data/mobilenetv2", # 权重路径
        "classes_path": "./datasets/test", # 数据集路径
        "input_shape": [224, 224],  # 输入图片大小
        "backbone": "MobileNetv2", # 网络结构
    }
```

2. 运行predict.py，根据提示选择参数即可。

## 5. 评估步骤
**注意：需要数据准备之后进行此步骤**

1. 首先在classification.py文件中，指定参数，如下：

```python
_defaults = {
        "model_path": "./logs/57", # 权重路径
        "classes_path": "./datasets/test", # 数据集路径
        "input_shape": [224, 224],  # 输入图片大小
        "backbone": "MobileNetv2", # 网络结构
    }
```

2. 在eval.py文件中，根据提示选择参数即可。

## 6. 参考
https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/ppcls/arch/backbone/model_zoo/mobilenet_v2.py








