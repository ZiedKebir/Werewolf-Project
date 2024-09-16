# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 20:15:37 2024

@author: ziedk
"""

#Load the DataSet 

import pymongo
import os
from PIL import Image
import numpy as np
import albumentations as A
import cv2
import matplotlib.pyplot as plt
from pymongo import MongoClient
import gridfs
from PIL import Image
import io
import random



def resize_image(image_array:np.ndarray):
    """
    Parameters
    ----------
    image_array : np.ndarray
        DESCRIPTION.
    Takes an image as an array and resizes it

    Returns
    -------
    nd.ndarray with shape: (225,225,3)
    """

    #Convert the ndarray to am image
    img = Image.fromarray(image_array)
    #resize the image
    resized_img = img.resize((225,225))
    #Returns a PIL.Image.Image Type*
    return np.array(resized_img)


# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['Werewolf']
fs = gridfs.GridFS(db)

#Get all the ids of the items stored in mongodb
images_id_mongodb = [i['_id'] for i in db.fs.files.find()]
random.shuffle(images_id_mongodb) #Shuffle the dataset
#get the byte object corresponding to each ID
images_byte_mongodb = [fs.find_one({'_id':i}) for i in images_id_mongodb]
#create a list containing the vector version of an image
list_images_arrays=list()


#Parses through byte images convert them to Pil images and then resizes them and changes them to array
#The resulting arrays are stored in a list. If there is an error with an image it is deleted from the tensor that will be used to train the model
index = 0
count = 0
for i in images_byte_mongodb:
    index+=1
    try:
        image = i.read()
        Pil_Image = Image.open(io.BytesIO(image))
        array = resize_image(np.array(Pil_Image))
        list_images_arrays.append(array)
    except:
        count+=1
        del images_id_mongodb[index]
        del images_byte_mongodb[index]


Training_tensor = torch.Tensor(list_images_arrays)




import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')




transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainset

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')