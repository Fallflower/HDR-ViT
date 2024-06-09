import argparse
import gzip
import numpy as np
import pandas as pd
from progressbar import *
from torchvision import transforms
from torch.utils.data import dataset, dataloader

class LoadDataset(dataset.Dataset):
    def __init__(self, data_file, label_file, num_images, data_dir='data/', transform=None):
        with gzip.open(os.path.join(data_dir, data_file), 'rb') as df:
            self.x = np.frombuffer(df.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
        with gzip.open(os.path.join(data_dir, label_file), 'rb') as lf:
            self.y = np.frombuffer(lf.read(), np.uint8, offset=8)
        self.length = len(self.y)
        self.transform = transform

    def __getitem__(self, i):
        img, target = self.x[i].transpose(1, 2, 0).copy(), self.y[i]
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return self.length


def get_test_trans():
    return transforms.Compose([
        transforms.ToTensor(),

    ])


def get_train_trans():
    return transforms.Compose([
        transforms.ToTensor(),

    ])


def get_dataloader(mode, data_file, label_file, opt: argparse.Namespace):
    if mode == 'train':
        return dataloader.DataLoader(
            dataset=LoadDataset(
                data_file=data_file,
                label_file=label_file,
                num_images=opt.num_images,
                transform=get_train_trans()
            ),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.tr_dl_num_worker
        )
    elif mode == 'test':
        return dataloader.DataLoader(
            dataset=LoadDataset(
                data_file=data_file,
                label_file=label_file,
                num_images=opt.num_images,
                transform=get_test_trans()
            ),
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.te_dl_num_worker
        )
    else:
        raise ValueError('Unknown mode: %s' % mode)
