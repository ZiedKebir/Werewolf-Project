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
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

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

# get the image labels 
image_roles = [i.Role for i in images_byte_mongodb]
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


#Code the image_roles 
hot_encoding = 0
hot_encoding_dict = dict()
for i in list(set(image_roles)):
    hot_encoding_dict[i]=hot_encoding
    hot_encoding+=1
    
image_roles_hot_encoding= [hot_encoding_dict[i] for i in image_roles]



# Convert the data to tensors
tensor_x =  torch.Tensor(list_images_arrays)
tensor_x  = tensor_x.permute(0,3,1,2)
tensor_y = torch.Tensor(image_roles_hot_encoding)
tensor_y  = tensor_y.permute(0,3,1,2)
tensor_y = tensor_y.long()

#Create a data loader
my_dataset = TensorDataset(tensor_x,tensor_y)
dataloader = DataLoader(my_dataset, batch_size = 300, shuffle = True )


"""
Building the CNN
"""
# Building CNN
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

class Net(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.features_extractor = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1),
            nn.ELU(),
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten())
        self.classifier = nn.Sequential(nn.Linear(64*112*112,num_classes),
                                        nn.Softmax(dim=-1))
    
    def forward(self,x):
        x = self.features_extractor(x)
        x = self.classifier(x)
        return x
    
"""
test = tensor_x[[(tensor_y==i).nonzero().squeeze()[0].item() for i in range(0,13)]] #test include on image of each class
test_y = tensor_y [[(tensor_y==i).nonzero().squeeze()[0].item() for i in range(0,13)]]
test_y = test_y.long()
test.size()
test = test.permute(0,3,1,2) #Change the indexation of the tensor so that it is recognized by the class object Net [Size,Channels,nbr_rows,nbr_columns]

f = net.forward(test)
"""


net = Net(13)
criterion = nn.CrossEntropyLoss() #used for a multiclassification problem
optimizer = optim.Adam(net.parameters(),lr=0.001)

for epoch in range(35):
    for images , labels in dataloader:
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output,labels)
        loss.backward()
        optimizer.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 30, loss.item()))        
            


        
