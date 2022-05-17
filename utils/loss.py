import paddle
import numpy as np
import paddle.nn as nn

class Poly1CrossEntropyLoss(paddle.nn.Layer):
    
    def __init__(self, num_classes, epsilon=1.0, reduction="none"):
        
        super(Poly1CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
 
    def forward(self, input, label):
        
        labels_onehot = nn.functional.one_hot(label, num_classes=self.num_classes)
        labels_onehot = paddle.to_tensor(np.array(labels_onehot, dtype=np.array(input).dtype))
        pt = paddle.sum(labels_onehot * nn.functional.softmax(input, axis=-1), axis=-1)
        CE = nn.functional.cross_entropy(input=input, label=label, reduction="none")
        poly1 = CE + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        return poly1