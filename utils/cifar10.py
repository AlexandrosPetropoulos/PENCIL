import os
import os.path
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle
from PIL import Image



class Cifar10Dataset(Dataset):

    base_folder = 'data/cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    # val_dataset is from data_batch_5

    val_list = [
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    

    transform1 = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32,4),
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.492, 0.482, 0.446), std = (0.247, 0.244, 0.262)),
    ])
    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = (0.492, 0.482, 0.446), std = (0.247, 0.244, 0.262)),
    ])
    


    def __init__(self, root, train=0,
                 transform=None, target_transform=None, noise = None, rate = 0.0):

        self.root = os.getcwd() # set path to current working directory (cwd)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.count = 0
        # a folder in cwd must exist with the dataset /data/cifar10/
        self.base_folder = os.path.join(os.path.expanduser('~'),os.getcwd(),'data/cifar10/')

        # now load the picked numpy arrays
        if self.train == 0:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                entry = pickle.load(fo, encoding='latin1')

                self.train_data.append(entry['data'])
                
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['fine_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data[0:45000]
            self.train_labels = self.train_labels[0:45000]

            # ###########################
            #adding noise to labels
            if(noise == 'symmetric'):
                for label in range(len(self.train_labels)):
                    if np.random.random()< rate:
                        self.train_labels[label] = np.random.randint(0,10)
                        self.count += 1
            elif(noise == 'asymmetric'):
                for label in range(len(self.train_labels)):
                    if np.random.random() < rate:
                        if self.train_labels[label] == 9:
                            self.train_labels[label] = 1
                        elif self.train_labels[label] == 2:
                            self.train_labels[label] = 0
                        elif self.train_labels[label] == 4:
                            self.train_labels[label] = 7
                        elif self.train_labels[label] == 3:
                            self.train_labels[label] = 5
                        elif self.train_labels[label] == 5:
                            self.train_labels[label] = 3
                        self.count += 1
            # ###########################
            print(f"{self.count} labels changed")
            self.train_data = self.train_data.reshape((45000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        elif self.train == 1:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['fine_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.val_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.val_data = entry['data']
            if 'labels' in entry:
                self.val_labels = entry['labels']
            else:
                self.val_labels = entry['fine_labels']
            fo.close()
            self.val_data = self.val_data[5000:10001]
            self.val_labels = self.val_labels[5000:10001]
            self.val_data = self.val_data.reshape((5000, 3, 32, 32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))  # convert to HWC


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train == 0:
            img, target = self.train_data[index], self.train_labels[index]
        elif self.train == 1:
            img, target = self.test_data[index], self.test_labels[index]
        else:
            img, target = self.val_data[index], self.val_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

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
            return len(self.train_data)
        elif self.train == 1:
            return len(self.test_data)
        else:
            return len(self.val_data)

