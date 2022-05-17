
import paddle
from paddle.static import InputSpec
from datasets import ClassificationDatasets
from utils.optimizer import create_optimzer
from utils.utils import get_classes
from utils.loss import Poly1CrossEntropyLoss
from net import nets
from utils.callback import LossHistory, ModelCheckpoint
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------ #
# -----------------参数说明------------------------ #
# input_shape：输入图片大小
# epochs：训练轮次
# batch_size：训练批次
# train_path：训练集路径
# valid_path：测试集路径
# loss：选择的loss函数CE代表交叉熵损失、Poly代表Poly交叉熵损失
# network：网络结构的选择{MobileNetv2}
# model_path：预训练权重路径
# ------------------------------------------------ #
input_shape = [224, 224, 3]
epochs = 100
batch_size = 16
train_path = './cls_train.txt'
valid_path = './cls_test.txt'
loss = 'Poly'
network = "MobileNetv2"
model_path = "./model_data/57"

# ------------------------------------------------ #
# 使用gpu
paddle.device.set_device('gpu')
# ------------------------------------------------ #
if __name__ == '__main__':


    num_classes, _ = get_classes('./datasets/test')
    loss_dict = {
        'CE': paddle.nn.CrossEntropyLoss(reduction='mean'),
        'Poly': Poly1CrossEntropyLoss(num_classes=num_classes, reduction='mean')
    }
    train_datasets = ClassificationDatasets('./cls_train.txt', input_shape[0], mode='train')
    valid_datasets = ClassificationDatasets('./cls_test.txt', input_shape[0], mode='valid')
    print('训练数据集数量:', len(train_datasets))
    print('验证数据集数量:', len(valid_datasets))
    step_each_epoch = len(train_datasets) // batch_size

    net = nets[network](class_num=num_classes)
    
    model = paddle.Model(net, inputs=[InputSpec(shape=[3, input_shape[0], input_shape[1]], dtype='float32', name='image')])
    if model_path != "":
        model.load(model_path)
        print("导入模型权重文件。。。")
    

    model.prepare(
        create_optimzer(net.parameters(), step_each_epoch, epochs),
        loss_dict[loss],
        paddle.metric.Accuracy(topk=(1, 5))
    )
    # visualdl --logdir ./visualdl_log_dir --port 8080
    visualdl = paddle.callbacks.VisualDL(log_dir='./logs1')
    EarlyStopping = paddle.callbacks.EarlyStopping(save_best_model=False,patience=15)
    # modelcheckpoint = paddle.callbacks.ModelCheckpoint(save_dir='./logs')
    LRScheduler = paddle.callbacks.LRScheduler(by_epoch=True, by_step=False)
    loss_history = LossHistory('./metric')
    modelcheckpoint = ModelCheckpoint(save_dir='./logs')

    model.fit(
        train_datasets,
        valid_datasets,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=1,
        callbacks=[visualdl, EarlyStopping, modelcheckpoint, LRScheduler, loss_history]
    )
