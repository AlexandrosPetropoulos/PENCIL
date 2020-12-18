import os
import os.path
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle
from PIL import Image



class Cifar100Dataset(Dataset):

    base_folder = 'data/cifar100'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', 'a88cafc152244af753f735de768cd67f']
    ]

    val_list = [
        ['train', 'a88cafc152244af753f735de768cd67f'],
    ]

    test_list = [
        ['test', '83941d587109b95175f43acc81a1287e'],
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

    # this list used for asymmetric noise
    list_of_similar_classes = [[4,30,55,72,95,4], [1,32,67,73,91,1], [54,62,70,82,92,54], [9,10,16,28,61,9], [0,51,53,57,83,0], [22,39,40,86,87,22], [5,20,25,84,94,5], [6,7,14,18,24,6], [3,42,43,88,97,3],
    [12,17,37,68,76,12], [23,33,49,60,71,23], [15,19,21,31,38,15], [34,63,64,66,75,34], [26,45,77,79,99,26], [2,11,35,46,98,2], [27,29,44,78,93,27], [36,50,65,74,80,36], [47,52,56,59,96,47], [8,13,48,58,90,8], [41,69,81,85,89,41]]

    def __init__(self, root, train=0,
                 transform=None, target_transform=None, noise = None, rate = 0.0):

        self.root = os.getcwd()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.count = 0

        # now load the picked numpy arrays
        if self.train == 0:
            self.train_data = []
            self.train_labels = []
            self.train_coarse_labels = [] # used for asymmetric noise
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

                self.train_coarse_labels = entry['coarse_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)

            self.train_data = self.train_data[0:45000]
            self.train_labels = self.train_labels[0:45000]
            ###########################
            #adding noise
            if(noise == 'symmetric'):
                for label in range(len(self.train_labels)):
                    if np.random.random()< rate:
                        self.train_labels[label] = np.random.randint(0,10)
                        #print(label)
                        self.count += 1
            elif(noise == 'asymmetric'):
                for label in range(len(self.train_labels)):
                    if np.random.random() < rate:
                        coarse_index = self.train_coarse_labels[label]
                        current_index_of_class = self.list_of_similar_classes[coarse_index].index(self.train_labels[label])
                        self.train_labels[label] = self.list_of_similar_classes[coarse_index][current_index_of_class + 1]
                        self.count += 1
            ###########################
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
            self.val_data = self.val_data[45000:50001]
            self.val_labels = self.val_labels[45000:50001]
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


