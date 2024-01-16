import os
import pickle
import glob
import random

import numpy as np
import torch.utils.data as data
from PIL import Image
import pandas as pd
import torchvision.transforms as T


class_list1=['Baby','Kid','Teenager','20-30s','40-50s','Senior','Negroid','Caucasian','Mongoloid','unmasked','masked','mid-light','light','mid-dark','dark','Fear','Disgust','Surprise','Anger','Sadness','Neutral','Happiness','Female','Male']

class Data166(data.Dataset):

    def __init__(self, split='train', partition=0, root='datasets', data_path='all_2.csv', transform=None, target_transform=None):

        df= pd.read_csv(data_path)
        self.transform = transform  # data transforms during training
        self.target_transform = target_transform  # data transforms during testing

        self.root_path = root  # path to datasets

        self.attr_id = df.loc[:, df.columns!='#Image'].columns.tolist()  # attribute names
        self.attr_num = len(self.attr_id)  # number of attributes
        self.img_id = df['#Image'].tolist()
        df_label = df.loc[:, df.columns!='#Image']
        self.label = [np.asarray(df_label.iloc[index].tolist()) for index in range(len(df_label))]



    

    def __getitem__(self, index):
        imgname, gt_label = self.img_id[index], self.label[index]
        imgpath = os.path.join(self.root_path, imgname)
        img = Image.open(imgpath)

        if self.transform is not None:
            img = self.transform(img)

        gt_label = gt_label.astype(np.float32)

        if self.target_transform is not None:
            gt_label = self.transform(gt_label)

        return img, gt_label, imgname

    def __len__(self):
        return len(self.img_id)

def get_transform(args):
    height = args.height
    width = args.width
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = [
        T.Resize((height, width))
    ]
	
    train_transform += [
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
    ]

    train_transform += [
        T.ToTensor(),
        # normalize,
    ]
    train_transform = T.Compose(train_transform)

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        # normalize
    ])
    return train_transform, valid_transform

def get_info():
    return class_list1, len(class_list1)