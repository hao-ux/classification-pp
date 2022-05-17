from classification import Classification
from PIL import Image


classfication = Classification()

print('请选择预测模式：')
print('可选的模式有：')
print('predict_datasets ----> 负责随机预测测试集内的某些图片')
print('predict -------------> 负责预测单张图片')
mode = input('请输入：')
# 数据集路径
datasets_path = './cls_test.txt'

if __name__ == '__main__':

    if mode == 'predict':
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                class_name = classfication.detect_image(image)
                print(class_name)
    elif mode == 'predict_datasets':
        is_show = input('是否展示图片(是、否)：')
        if is_show == '是':
            is_show = True
        else:
            is_show = False
        classfication.detect_datasets(datasets_path, is_show=True)
    