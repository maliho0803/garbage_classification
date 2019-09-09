import glob
import os
import random

fd_train = open('/Users/zhoumi/Downloads/garbage_classify/train.txt', 'w')
fd_test = open('/Users/zhoumi/Downloads/garbage_classify/val.txt', 'w')
img_files = glob.glob('/Users/zhoumi/Downloads/garbage_classify/train_data/*jpg')

for img_file in img_files:
    class_file = img_file.replace('.jpg', '.txt')
    txt = open(class_file, 'r')
    label = txt.readlines()[0].split(' ')[-1]

    if random.uniform(0, 1) > 0.1:
        fd_train.write(img_file)
        fd_train.write(' ')
        fd_train.write(label)
        fd_train.write('\n')
    else:
        fd_test.write(img_file)
        fd_test.write(' ')
        fd_test.write(label)
        fd_test.write('\n')
    print(img_file)

fd_train.close()
fd_test.close()