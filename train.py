import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.models
from dataloader import *
import torchvision.transforms as T
from loss import CrossEntropyLabelSmooth, CenterLoss, TripletLoss
from torch.utils.data import DataLoader
from meters import AverageMeter
from model import resnet50, Baseline, ft_net, efficient_baseline
from torch.autograd import Variable
import torch
from bisect import bisect_right
from ramdom_erase import Cutout, RandomErasing
from samplers import RandomIdentitySampler, RandomIdentitySampler_new
from efficientnet_pytorch import EfficientNet


NUM_CLASSES = 40
MAX_EPOC = 60
BATCH_SIZE = 32
TEST_BATCH_SIZE = 1
use_triplet = True
use_ff = False
use_efficientnet = False

def adjust_lr(ep):
    lr = 1e-4
    if use_triplet == True:
        warmup_factor = 1
        if ep < 10:
            alpha = ep / 10
            warmup_factor = 0.01 * (1 - alpha) + alpha

        lr = lr * warmup_factor * 0.1 ** bisect_right([20, 40], ep)
    else:
        if ep <4:
            lr = 1e-4
        elif ep < 7:
            lr =1e-5
        else:
            lr = 1e-6

    return lr

# model = resnet50(num_classes=NUM_CLASSES, pretrained=True)
if use_ff == False:
    if use_efficientnet == True:
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=NUM_CLASSES)
        # model = torch.nn.DataParallel(model)
    else:
        model = efficient_baseline(num_classes=NUM_CLASSES, neck='bnneck')
else:
    model = ft_net(num_classes=NUM_CLASSES)
print(model)

train_transform = T.Compose([T.Resize((224, 224)),
                             T.RandomHorizontalFlip(),
                             # T.RandomVerticalFlip(),
                             # T.ColorJitter(0.5, 0.5, 0.5, 0.5),
                             T.Pad(10),
                             T.RandomCrop((224, 224)),
                             # T.RandomRotation(90),
                             T.ToTensor(),
                             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                             Cutout(probability=0.5, size=64, mean=[0.0, 0.0, 0.0]),
                             RandomErasing(probability=0.0, mean=[0.485, 0.456, 0.406])])

test_transform = T.Compose([T.Resize((224, 224)),
                             T.ToTensor(),
                             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_datasets = MyDataset(txt_path='/data/zhoumi/datasets/train_data/train.txt', transform=train_transform)
train = process_dir(txt_path='/data/zhoumi/datasets/train_data/train.txt')

if use_triplet == True:
    train_data = DataLoader(train_datasets, sampler=RandomIdentitySampler_new(train, NUM_CLASSES ,4),
                        batch_size=BATCH_SIZE, pin_memory=True, num_workers=8, drop_last=True)
else:
    train_data = DataLoader(train_datasets, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True)

test_data = DataLoader(MyDataset(txt_path='/data/zhoumi/datasets/train_data/val.txt', transform=test_transform),
                       batch_size=TEST_BATCH_SIZE, pin_memory=True)

optimizer = optim.Adam(params=model.parameters(), lr=1e-4, weight_decay=5e-4)
# optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)

#define loss function
xent_criterion = CrossEntropyLabelSmooth(NUM_CLASSES)
center_criterion = CenterLoss(NUM_CLASSES, feat_dim=1792)
triplet_criterion = TripletLoss(margin=0.3)

best_model = model
best_acc = 0
print(len(test_data) * TEST_BATCH_SIZE, len(train_data))

model = model.cuda()

for epoch in range(MAX_EPOC):
    lr = adjust_lr(epoch)
    for p in optimizer.param_groups:
        p['lr'] = lr

    for i, inputs in enumerate(train_data):
        model = model.train()
        images, labels = Variable(inputs[0].cuda()), Variable(inputs[1].cuda())
        if use_ff == False:
            if use_efficientnet == True:
                output = model(images)
            else:
                output, feat = model(images)
        else:
            output1, output2, output3 = model(images)
        if use_triplet == True:
            sofmax_loss = xent_criterion(output, labels)
            triplet_loss = triplet_criterion(feat, labels)[0]
            losses = sofmax_loss + triplet_loss + 0.0005 * center_criterion(feat, labels)
        else:
            if use_ff == False:
                losses = xent_criterion(output, labels)
            else:
                losses = (xent_criterion(output1, labels) + xent_criterion(output2, labels) + xent_criterion(output3, labels))/3
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            corrects = 0
            model = model.eval()
            for j, test in enumerate(test_data):
                t_images, t_labels = Variable(test[0].cuda()), Variable(test[1].cuda())

                if use_ff == False:
                    if use_efficientnet == True:
                        pred = torch.argmax(model(t_images), 1)
                    else:
                        _, pred = torch.max(model(t_images)[0], 1)
                else:
                    o1, o2, o3 = model(t_images)
                    _, pred = torch.max((o1 + o2 + o3)/3, 1)
                    print(pred, t_labels.data)

                corrects += torch.sum(pred == t_labels.data)

            acc = corrects.item() / len(test_data) / TEST_BATCH_SIZE
            if acc > best_acc:
                best_acc = acc
                best_model = model

            if use_triplet == True:
                print("epoch: {}, iter: {}, lr: {}, loss: {}, softmax_loss: {}, triplet_loss: {} acc: {}".format(epoch,
                 i, optimizer.param_groups[0]['lr'], losses.item(), sofmax_loss.item(), triplet_loss.item(), acc))
            else:
                print("epoch: {}, iter: {}, lr: {}, loss: {}, acc: {}".format(epoch,
                i, optimizer.param_groups[0]['lr'], losses.item(), acc))

torch.save(model, './best_model_v2_tri_center_old.pth')






