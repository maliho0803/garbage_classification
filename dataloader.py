from PIL import Image
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, txt_path = '../../data/garbage_classify/img_list.txt', transform = None):
        fd = open(txt_path, 'r')
        imgs = []

        for line in fd:
            line = line.rstrip()
            words = line.split(' ')
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.transforms = transform

    def __getitem__(self, item):
        fn, label = self.imgs[item]

        image = Image.open(fn)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.imgs)

def process_dir(txt_path = '../../data/garbage_classify/img_list.txt'):
    fd = open(txt_path, 'r')
    imgs = []

    for line in fd:
        line = line.rstrip()
        words = line.split(' ')
        imgs.append((words[0], int(words[1])))
    return imgs
