from model import Baseline, ft_net, efficient_baseline
from PIL import Image
import glob
import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from efficientnet_pytorch import EfficientNet

use_ff = False
use_efficientnet = False
transform = T.Compose([T.Resize((224, 224)),
                             T.ToTensor(),
                             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

img_paths = glob.glob('/Users/zhoumi/Downloads/garbage_classify/val_data/*jpg')
# if use_ff == False:
#     if use_efficientnet == True:
#         model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=40)
#     else:
#         model = Baseline(num_classes=40)
# else:
#     model = ft_net(num_classes=40)
# model.load_state_dict(torch.load('./models/ff_best_model.pth', map_location=lambda storage, loc: storage))
model = torch.load('./models/best_model_v2_tri_old1.pth', map_location=lambda storage, loc: storage)
model = model.eval().cpu()
# print(model)

wrong = 0
for img_path in img_paths:
    label = int(img_path.split('/')[-1].split('_')[0])
    img = transform(Image.open(img_path))
    input = img[np.newaxis, :, :, :]
    # print(input.size())

    if use_ff == False:
        if use_efficientnet == True:
            pred_score = model(input)
        else:
            pred_score, _ = model(input)
        # print(pred_score)
        pred_label = torch.argmax(pred_score, dim=1).item()
    else:
        o1, o2, o3 = model(input)
        pred_label = torch.argmax((o1 + o2 + o3) / 3, dim=1).item()

    print(img_path.split('/')[-1], label, pred_label)
    if label != pred_label:
        # plt.imshow(Image.open(img_path))
        wrong +=1
        # plt.show()

print('acc：{}'.format(1- wrong/len(img_paths)))

# best_model1.pth acc：0.9082819986310746  resnet50
# tri_best_model.pth acc：acc：0.9103353867214237 resnet50
# ff_best_model.pth acc：0.8809034907597536 feature fusion
# effic_best_model acc：0.9226557152635182 effic4
# effic4_best_model.pth acc：0.9301848049281314 effic4

#new datasets
#best_modle_v2 acc：0.9340878828229028