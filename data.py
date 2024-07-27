import torch
import torchvision
import torch.nn as nn
import numpy as np
import csv
from PIL import Image
import pytorch_lightning

from torchvision import transforms
from torch.utils.data import DataLoader

file="fer2013.csv"

train_x,train_y=[],[]
public_test_x,public_test_y=[],[]
private_test_x,private_test_y=[],[]

with open(file,'r') as csvin:
    data=csv.reader(csvin)
    for row in data:
        
        if row[-1]=="Training":
            # put all pixels into a list
            temp=[]
            for pixel in row[-2].split( ):
                temp.append(int(pixel))
            train_x.append(temp)
            train_y.append(int(row[0]))
        
        if row[-1]=="PublicTest":
            # put all pixels into a list
            temp=[]
            for pixel in row[-2].split( ):
                temp.append(int(pixel))
            public_test_x.append(temp)
            public_test_y.append(int(row[0]))
            
        if row[-1]=="PrivateTest":
            # put all pixels into a list
            temp=[]
            for pixel in row[-2].split( ):
                temp.append(int(pixel))
            private_test_x.append(temp)
            private_test_y.append(int(row[0]))
            
# convert data into numpy arrays
train_x,train_y=np.asarray(train_x), np.asarray(train_y)
public_test_x,public_test_y=np.asarray(public_test_x), np.asarray(public_test_y)
private_test_x,private_test_y=np.asarray(private_test_x), np.asarray(private_test_y)

# reshape into 48*48
train_x=train_x.reshape(-1,48,48)
public_test_x=public_test_x.reshape(-1,48,48)
private_test_x=private_test_x.reshape(-1,48,48)

# print data shape
# print(f"Train data  : {train_x.shape} | Target: {train_y.shape}")
# print(f"Public test : {public_test_x.shape} | Target: {public_test_y.shape}")
# print(f"Private test: {private_test_x.shape} | Target: {private_test_y.shape}")


from torch.utils.data import Dataset

class FER2013(Dataset):
    def __init__(self,split="train",transforms=None,device="mps"):
        super().__init__()
        
        self.transforms=transforms
        self.split=split
        self.device=device
        
        if split=='train':
            self.train_x=train_x
            self.train_y=train_y
        elif split=="public":
            self.public_test_x=public_test_x
            self.public_test_y=public_test_y
        elif split=="private":
            self.private_test_x=private_test_x
            self.private_test_y=private_test_y
    
    def __getitem__(self,idx):
        # get data and label at idx
        if self.split=='train':
            img,label=self.train_x[idx], self.train_y[idx]
        elif self.split=='public':
            img,label=self.public_test_x[idx], self.public_test_y[idx]
        elif self.split=='private':
            img,label=self.private_test_x[idx], self.private_test_y[idx]
        
        if self.transforms is not None:
            img=Image.fromarray(img.astype(np.uint8))
            img=self.transforms(img)
        
        # move to correct device
        img=img.to(self.device)
        label=torch.tensor(label,device=self.device)
        
        return img,label
    
    def __len__(self):
        if self.split=='train':
            return len(self.train_y)
        elif self.split=='public':
            return len(self.public_test_y)
        elif self.split=='private':
            return len(self.private_test_y)
        

# set up transforms and data loaders for training
from torchvision import transforms
from torch.utils.data import DataLoader

cut_size=44

# for each train image: random crop + horizontal flip + to tensor
transform_train=transforms.Compose([transforms.RandomCrop(cut_size),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor()])

# for each test image: random crop into 10 subimages, prediction = average over these images
transform_test=transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) 
                                                 for crop in crops]))
])

# get data loaders
batch_size=128

train_data=FER2013(split="train",transforms=transform_train,device="mps")
train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True)

val_data=FER2013(split="public",transforms=transform_test,device="mps")
val_loader=DataLoader(val_data,batch_size=batch_size,shuffle=False)

test_data=FER2013(split="private",transforms=transform_test,device="mps")
test_loader=DataLoader(test_data,batch_size=batch_size,shuffle=False)

# check a batch of data
# batch,label=next(iter(train_loader))
# print(f"Train batch: {batch.shape} | Label: {label.shape}")
# print(f"Device: {batch.device}")