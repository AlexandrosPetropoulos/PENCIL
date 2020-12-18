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


class Cub200_2011Dataset(Dataset):

    base_folder = 'data/cub200_2011'
    url = "www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
    filename = "CUB_200_2011.tgz"
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    train_list = [  ]

    val_list = [  ]

    test_list = [  ]
     
    
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
        file = os.path.join(self.root, self.base_folder, 'images.txt')
        temp_df1 = pd.read_csv(file, header = None, names = ['image_id','image_file_name'], delim_whitespace=True)

        #read class-targets
        file = os.path.join(self.root, self.base_folder, 'image_class_labels.txt')
        temp_df2 = pd.read_csv(file, header = None, names = ['image_id','class_name'] ,delim_whitespace=True)
        #make ID start from 0
        temp_df2['image_id'] = temp_df2['image_id'].apply(lambda x: x-1)
        #make classes have range 0-199
        temp_df2['class_name'] = temp_df2['class_name'].apply(lambda x: x-1)

        #read the recommended file
        file = os.path.join(self.root, self.base_folder, 'train_test_split.txt')
        temp_df3 = pd.read_csv(file, header = None, names = ['image_id','is_training_image'] ,delim_whitespace=True)

        #create dataframe with image_id, image_file_name, class_name, is_training_image
        pd_dataset = pd.DataFrame([temp_df2['image_id'],temp_df1['image_file_name'],temp_df2['class_name'],temp_df3['is_training_image']]).transpose()
        del temp_df1, temp_df2, temp_df3

        #create dataframe for train, val and test set
        train_df =  pd_dataset[pd_dataset.is_training_image == 1]
        del train_df['is_training_image']

        self.test_df = pd_dataset[pd_dataset.is_training_image == 0]
        del self.test_df['is_training_image']
        del pd_dataset

        # fix index
        self.test_df = self.test_df.reset_index(drop=True)

        # create val set keeping 10% of train(stratified)
        train_array = train_df.to_numpy()
        del train_df
        X_train, X_test, y_train, y_test = train_test_split(train_array[:,0:train_array.shape[1]-1], train_array[:,-1],stratify=train_array[:,-1], test_size=0.1)
        self.train_df = pd.DataFrame(np.c_[X_train,y_train], columns = ['image_id','image_file_name','class_name'])
        self.val_df = pd.DataFrame(np.c_[X_test,y_test], columns = ['image_id','image_file_name','class_name'])
        del X_train, X_test, y_train, y_test, train_array


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train == 0:
            img = Image.open(os.path.join(self.root,self.base_folder,'images',self.train_df.image_file_name[index])).convert("RGB")
            target = self.train_df.class_name[index]
        elif self.train == 1:
            img = Image.open(os.path.join(self.root,self.base_folder,'images',self.test_df.image_file_name[index])).convert("RGB")
            target = self.test_df.class_name[index]
        else:
            img = Image.open(os.path.join(self.root,self.base_folder,'images',self.val_df.image_file_name[index])).convert("RGB")
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
    
