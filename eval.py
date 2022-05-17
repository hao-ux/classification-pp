from classification import Classification

# ----------------------------------------- #
# ----------------评估--------------------- #

print('选择评估指标：')
print('可选择的评估指标：top1、top5')
eval_acc = input('请输入：')

classfication = Classification()
# 数据集路径
datasets_path = './cls_test.txt'

def top1ACC(datasets_path):
    print('使用top1ACC评价指标')
    acc = classfication.eval(datasets_path, mode='top1_acc')
    print("Final_Acc:{:.3f}".format(acc))
    
def top5ACC(datasets_path):
    print('使用top5ACC评价指标')
    acc = classfication.eval(datasets_path, mode='top5_acc')
    print("Final_Acc:{:.3f}".format(acc))
    
if __name__ == '__main__':
    
    if eval_acc == 'top1':
        top1ACC(datasets_path)
    elif eval_acc == 'top5':
        top5ACC(datasets_path)