'''
    拆分训练和验证集的方法
'''
import os
root = r'E:\zqw\PaperCode\data\ObjectDetection\kitti_original\training\\'
train_file = open(os.path.join(root, 'split','train.txt'), 'w')
val_file = open(os.path.join(root, 'split','val.txt'), 'w')
from sklearn.model_selection import train_test_split
file_ids = ["%06d\n" % i for i in range(0,7481)]
train_ids, val_ids = train_test_split(file_ids, test_size=0.3)
print(len(train_ids))
train_file.writelines(train_ids)
val_file.writelines(val_ids)
train_file.close()
val_file.close()
