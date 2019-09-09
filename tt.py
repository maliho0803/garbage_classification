import shutil
import os
import glob
src_path = '/Users/zhoumi/Downloads/garbage_classify_v2/train_data_v2/'
dst_path = '/Users/zhoumi/Downloads/garbage_classify_v2/val'
with open('/Users/zhoumi/Downloads/garbage_classify_v2/train_data_v2/val.txt', 'r') as fd:
    lines = fd.readlines()
    for line in lines:
        img_path = line.split(' ')[0].split('/')[-1]
        label = str(line.split(' ')[1])
        img_name = str(label).replace('\n', '') + '_' + img_path
        shutil.copy(os.path.join(src_path, line.split(' ')[0].split('/')[-1]), os.path.join(dst_path, img_name))

# img_paths = glob.glob('/Users/zhoumi/Downloads/garbage_classify/train_data/*jpg')
# dst_path = '/Users/zhoumi/Downloads/garbage_classify/new/'
# for img_path in img_paths:
#     txt_path = img_path.replace('.jpg', '.txt')
#     with open(txt_path, 'r') as fd:
#         line = fd.readlines()[0]
#         lable = line.split(' ')[-1]
#         if not os.path.exists(os.path.join(dst_path, str(lable))):
#             os.mkdir(os.path.join(dst_path, str(lable)))
#
#         shutil.copy(img_path, os.path.join(dst_path, str(lable), line.split(' ')[0][:-1]))
#     fd.close()

from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=40)


