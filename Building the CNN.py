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
import matplotlib.pyplot as plt
from pymongo import MongoClient
import gridfs
import io
import random
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


import torch
print(torch.version.cuda)


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
tensor_x  = tensor_x.permute(0,2,1,3) # #Change the indexation of the tensor so that it is recognized by the class object Net [Size,Channels,nbr_rows,nbr_columns]
tensor_y = torch.Tensor(image_roles_hot_encoding)
tensor_y.size()
#tensor_y  = tensor_y.permute(0,3,1,2)
tensor_y = tensor_y.long()
#Create a data loader
my_dataset = TensorDataset(tensor_x,tensor_y)
dataloader = DataLoader(my_dataset, batch_size = 8, shuffle = True, pin_memory=True )



"""
Building the CNN
"""
# Building CNN
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

# Define the model class (same as your original code)
class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*28*28, num_classes),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        x = self.features_extractor(x)
        x = self.classifier(x)
        return x

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate the model and move it to the appropriate device (CPU or GPU)
net = Net(13).to(device)
#net = Net(13)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For multiclass classification
optimizer = optim.Adam(net.parameters(), lr=0.001)

from torch.utils.tensorboard import SummaryWriter

# Create a TensorBoard writer
writer = SummaryWriter('runs/Large_CNN_3_Layers')
num_epochs = 20

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.permute(0,2,1,3)
        #print(images.size())

        images = images.to(device)
        labels = labels.to(device)
        #print("Label Size",labels.size())
        optimizer.zero_grad()
        output = net(images)
        #print("Output Size",output.size())
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # Update metrics
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total

    # Log metrics to TensorBoard
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')


writer.close()  # Don't forget to close the writer



labels.size()


output.size()

torch.cuda.empty_cache()


# Save model and optimizer states
def save_checkpoint(model, optimizer, epoch, file_path="model_checkpoint.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, file_path)
    
    