import paddle
from net.mobilenetv2 import MobileNetv2

if __name__ == '__main__':
    net = MobileNetv2(class_num=17, scale=0.5)
    model = paddle.Model(net)
    model.summary((1, 3, 224, 224))
    for i, (name, param) in enumerate(net.named_parameters()):
        print(i, name)
    
    # print(model.network)
    