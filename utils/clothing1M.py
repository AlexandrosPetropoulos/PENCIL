import os
import os.path
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle
from PIL import Image
from skimage import io
import random
from sklearn.model_selection import train_test_split


class clothing1MDataset(Dataset):

    base_folder = 'data/clothing1M'
    url = "Google drive"
    filename = "clothing1M"
    tgz_md5 = '97eceeb196236b16543738112f37df78'

    #files
    train_file_list = []

    val_file_list = []

    test_file_list = []

    #files
    clean_label_kv = os.path.join(base_folder,r'annotations/clean_label_kv.txt')
    clean_test_key_list = os.path.join(base_folder,r'annotations/clean_test_key_list.txt')
    clean_train_key_list = os.path.join(base_folder,r'annotations/clean_train_key_list.txt')
    clean_val_key_list = os.path.join(base_folder,r'annotations/clean_val_key_list.txt')
    noisy_label_kv = os.path.join(base_folder,r'annotations/noisy_label_kv.txt')
    noisy_label_kv_260K = os.path.join(base_folder,r'annotations/noisy_label_kv_260K.txt')
    noisy_train_key_list = os.path.join(base_folder,r'annotations/noisy_train_key_list.txt')

    
    transform1 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((256,256)),
        transforms.RandomCrop(224,4),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.492, 0.482, 0.446), std = (0.247, 0.244, 0.262)),
    ])
    transform2 = transforms.Compose([
        transforms.Resize((224,224)),#256
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.492, 0.482, 0.446), std = (0.247, 0.244, 0.262)),
    ])
    
    
    

    def __init__(self, root, train=0,
                 transform=None, target_transform=None, noise = None, rate = 0.0):

        self.root = os.getcwd()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.count = 0

        #read the appropriate files
        #read ID and file name
        self.train_df = pd.read_csv(self.noisy_label_kv_260K, header = None, names = ['image_file_name','class_name'], delim_whitespace=True)

        #val set
        df_clean_label_all = pd.read_csv(self.clean_label_kv, header = None, names = ['image_file_name','class_name'], delim_whitespace=True)
        df_clean_label_test_files = pd.read_csv(self.clean_val_key_list, header = None, names = ['image_file_name'], delim_whitespace=True)

        df_all = df_clean_label_all.merge(df_clean_label_test_files.drop_duplicates(), on=['image_file_name'], how='left', indicator=True)
        self.val_df = df_clean_label_all[df_all['_merge']== 'both'].reset_index(drop=True)

        del df_clean_label_all, df_clean_label_test_files, df_all

        #test set
        df_clean_label_all = pd.read_csv(self.clean_label_kv, header = None, names = ['image_file_name','class_name'], delim_whitespace=True)
        df_clean_label_test_files = pd.read_csv(self.clean_test_key_list, header = None, names = ['image_file_name'], delim_whitespace=True)

        df_all = df_clean_label_all.merge(df_clean_label_test_files.drop_duplicates(), on=['image_file_name'], how='left', indicator=True)
        self.test_df = df_clean_label_all[df_all['_merge']== 'both'].reset_index(drop=True)

        del df_clean_label_all, df_clean_label_test_files, df_all



    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        if self.train == 0:
            img = Image.open(os.path.join(self.root,self.base_folder,self.train_df.image_file_name[index])).convert("RGB")
            target = self.train_df.class_name[index]
        elif self.train == 1:
            img = Image.open(os.path.join(self.root,self.base_folder,self.test_df.image_file_name[index])).convert("RGB")
            target = self.test_df.class_name[index]
        else:
            img = Image.open(os.path.join(self.root,self.base_folder,self.val_df.image_file_name[index])).convert("RGB")
            target = self.val_df.class_name[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train == 0:
            return img, target, index
        else:
            return img, target

    def __len__(self):
        if self.train == 0:
            return self.train_df.shape[0]
        elif self.train == 1:
            return self.test_df.shape[0]
        else:
            return self.val_df.shape[0]
    
