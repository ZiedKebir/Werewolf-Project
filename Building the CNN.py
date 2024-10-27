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

"""
Load the training dataset 
"""
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


set(image_roles)

# Convert the data to tensors
tensor_x =  torch.Tensor(list_images_arrays)
tensor_x  = tensor_x.permute(0,2,1,3) # #Change the indexation of the tensor so that it is recognized by the class object Net [Size,Channels,nbr_rows,nbr_columns]
tensor_y = torch.Tensor(image_roles_hot_encoding)
tensor_y.size()
#tensor_y  = tensor_y.permute(0,3,1,2)
tensor_y = tensor_y.long()
#Create a data loader
my_dataset = TensorDataset(tensor_x,tensor_y)
dataloader = DataLoader(my_dataset, batch_size = 128, shuffle = True, pin_memory=True )


"""my_dataset_red = TensorDataset(tensor_x[0:128],tensor_y[0:128])
dataloader_red = DataLoader(my_dataset_red, batch_size=128)"""

"""
x = 236
tensor_np = tensor_x[x].numpy()

image = Image.fromarray((tensor_np).astype(np.uint8))  # Assuming tensor values are normalized
image.show()

tensor_y[x]
image_roles[x]
"""

"""
Load the Testing/Validation dataset 
"""

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db_test = client['Werewolf_Test_validation']
fs = gridfs.GridFS(db_test)

c = 0

for i in db_test['fs.files'].find():
    print(i['_id'])


#Get all the ids of the items stored in mongodb
images_id_mongodb = [i['_id'] for i in db_test.fs.files.find()]
random.shuffle(images_id_mongodb) #Shuffle the dataset

#get the byte object corresponding to each ID
images_byte_mongodb = [fs.find_one({'_id':i}) for i in images_id_mongodb]


# get the image labels 
image_roles = [i.Role for i in images_byte_mongodb]
#create a list containing the vector version of an image
list_images_arrays=list()





images_byte_mongodb[3].read()

#Parses through byte images convert them to Pil images and then resizes them and changes them to array
#The resulting arrays are stored in a list. If there is an error with an image it is deleted from the tensor that will be used to train the model
index = 0
count = 0
for i in images_byte_mongodb:
    index+=1
    image = i.read()
    Pil_Image = Image.open(io.BytesIO(image))
    print(Pil_Image)
    array = resize_image(np.array(Pil_Image))
    list_images_arrays.append(array)



#Code the image_roles 
hot_encoding = 0
hot_encoding_dict = dict()
for i in list(set(image_roles)):
    hot_encoding_dict[i]=hot_encoding
    hot_encoding+=1
    
image_roles_hot_encoding= [hot_encoding_dict[i] for i in image_roles]




#Check that all the images have a size equal to (225,225,3) and change the images with 1 channel to a three channel image
count = 0 
corrected_list_images_array = list()
for i in list_images_arrays:
    if i.shape != (225,225,3):
        image = Image.fromarray(i).convert("RGB")
        corrected_list_images_array.append(PIL_to_array(image))
    else:
        corrected_list_images_array.append(i)
        

#check whether all the images in the corrected list have the right size
count=0
for i in corrected_list_images_array:
    if i.shape != (225,225,3):
        count+=1
print(count)
        







"""
# Show an image that is an array
image = Image.fromarray((list_images_arrays[1]).astype(np.uint8))  # Assuming tensor values are normalized
image.show()
"""




# Convert the data to tensors
tensor_x_test =  torch.Tensor(corrected_list_images_array)
tensor_x_test  = tensor_x_test.permute(0,2,1,3) # #Change the indexation of the tensor so that it is recognized by the class object Net [Size,Channels,nbr_rows,nbr_columns]
tensor_y_test = torch.Tensor(image_roles_hot_encoding)
tensor_y_test.size()
#tensor_y  = tensor_y.permute(0,3,1,2)
tensor_y_test = tensor_y_test.long()
#Create a data loader
my_dataset_test = TensorDataset(tensor_x_test,tensor_y_test)
dataloader_Test = DataLoader(my_dataset_test, batch_size = 64, shuffle = True, pin_memory=True )



tensor_x_test.size()
tensor_y_test.size()
    



"""
Building the CNN
"""
# Building CNN
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from tqdm import tqdm

# Define the model class (same as your original code)
class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2),
            nn.ELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1,stride=2),
            nn.ELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1,stride=2),
            nn.ELU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes), 
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        x = self.features_extractor(x)
        x = self.classifier(x)
        return x

net = Net(13)


first_batch = next(iter(dataloader))
first_batch[0].shape
first_batch = first_batch[0].permute(0,3,1,2)

x = net.features_extractor(first_batch)
x.size()

net.classifier(x)






# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#device = torch.device("cpu")

print(f"Using device: {device}")

# Instantiate the model and move it to the appropriate device (CPU or GPU)
#net = Net(13).to(device) #13 is the number of predictions in this case we have 13 classes
net = Net(13)


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For multiclass classification
optimizer = optim.Adam(net.parameters(), lr=0.000001)
#optimizer=torch.optim.Adamax(net.parameters(), lr=0.00001)
#optimizer = torch.optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
#optimizer = torch.optim.RMSprop(net.parameters(), lr=0.0001)




from torch.utils.tensorboard import SummaryWriter


def load_checkpoint(model, optimizer, file_path="model_checkpoint.pth"):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

# Create a TensorBoard writer
# Line of code to launch Tensorboard ---> tensorboard --logdir=runs 
writer = SummaryWriter('runs/run_2/Large_CNN_3_Layers_Test_Val')
num_epochs = 1

#net,optimizer,start_epoch = load_checkpoint(net,optimizer,file_path="model_checkpoint.pth")





def check_accuracy(loader, model):
    """
    Checks the accuracy of the model on the given dataset loader.

    Parameters:
        loader: DataLoader
            The DataLoader for the dataset to check accuracy on.
        model: nn.Module
            The neural network model.
    """
    #if loader.dataset.train:
    #    print("Checking accuracy on training data")
    #else:
    #    print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        for x, y in loader:
            x = x.permute(0, 3, 1, 2)
            # Forward pass: compute the model output
            scores = model(x)
            _, predictions = scores.max(1)  # Get the index of the max log-probability
            num_correct += (predictions == y).sum()  # Count correct predictions
            num_samples += predictions.size(0)  # Count total samples

        # Calculate accuracy
        accuracy = float(num_correct) / float(num_samples) * 100
        print(f"Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")
    
    model.train()  # Set the model back to training mode



for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    for batch_index, (data, targets) in enumerate(tqdm(dataloader)):
        # Move data and targets to the device (GPU/CPU)
        data  = data.permute(0, 3, 1, 2)
        # Forward pass: compute the model output
        scores = net(data)
        loss = criterion(scores, targets)

        # Backward pass: compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # Optimization step: update the model parameters
        optimizer.step()
    
    print("Epoch[{}/{}] ,Train Accuracy: {:.1f}%", 
          epoch+1,num_epochs,check_accuracy(dataloader, net))
    
    with torch.no_grad():
        net.eval()
        correct = 0
        total = 0
        all_val_loss = list()
        for images, labels in dataloader_Test:
            outputs = net(images)
            total+=labels.size(0)
            #calculated predictions
            predicted = torch.argmax(outputs,dim=1)
            print("print predicted values",predicted)
            #calculated actual values
            correct += (predicted == labels).sum().item()
            #calculate the loss 
            all_val_loss.append(criterion(outputs,labels).item())
        #calculate val-loss
        mean_val_loss = sum(all_val_loss)/len(all_val_loss)
        #calculate val-accuracy 
        mean_val_acc = 100 * (correct/total)
    print(
        'Epoch [{}/{}], Loss:{:.4f}, val-loss: {:.4f}, Val-acc: {:.1f}%',
          epoch+1, num_epochs, loss.item(), mean_val_loss, mean_val_acc
          )
            
            
    


def check_accuracy(loader, model):
    """
    Checks the accuracy of the model on the given dataset loader.

    Parameters:
        loader: DataLoader
            The DataLoader for the dataset to check accuracy on.
        model: nn.Module
            The neural network model.
    """
    #if loader.dataset.train:
    #    print("Checking accuracy on training data")
    #else:
    #    print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation
        for x, y in loader:
            x = x.permute(0, 3, 1, 2)
            # Forward pass: compute the model output
            scores = model(x)
            _, predictions = scores.max(1)  # Get the index of the max log-probability
            num_correct += (predictions == y).sum()  # Count correct predictions
            num_samples += predictions.size(0)  # Count total samples

        # Calculate accuracy
        accuracy = float(num_correct) / float(num_samples) * 100
        print(f"Got {num_correct}/{num_samples} with accuracy {accuracy:.2f}%")
    
    model.train()  # Set the model back to training mode

# Final accuracy check on training and test sets
check_accuracy(dataloader, net)




"""
for epoch in range(0, num_epochs):
    count_batches = 0 
    running_loss = 0.0
    correct = 0
    total = 0

    # Set model to training mode
    net.train()

    for images, labels in dataloader_red:
        images = images.permute(0, 3, 1, 2)  # Adjust shape if needed
        #images = images.to(device)
        #labels = labels.to(device)

        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # Update metrics
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        count_batches += 1 
        print(f"Batch {count_batches}/376 completed")
    
    print(f'Epoch [{epoch+1}/{num_epochs}] Completed')

    avg_loss = running_loss / len(dataloader_red)
    accuracy = 100 * correct / total

    # Log metrics to TensorBoard
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    # Validation phase
    net.eval()  # Set model to evaluation mode
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():  # No gradient calculation for validation
        for val_images, val_labels in dataloader_Test:
            val_images = val_images.permute(0, 3, 1, 2) 
            #val_images = val_images.to(device)
            #val_labels = val_labels.to(device)

            val_output = net(val_images)
            loss = criterion(val_output, val_labels)

            val_loss += loss.item()
            _, predicted = torch.max(val_output.data, 1)
            total_val += val_labels.size(0)
            correct_val += (predicted == val_labels).sum().item()

    avg_val_loss = val_loss / len(dataloader_Test)
    val_accuracy = 100 * correct_val / total_val

    # Log validation metrics to TensorBoard
    writer.add_scalar('Loss/val', avg_val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_accuracy, epoch)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.2f}%, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

# Close the TensorBoard writer
writer.close()
"""




torch.cuda.reset_peak_memory_stats()
labels.size()


output.size()

torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Save model and optimizer states
def save_checkpoint(model, optimizer, epoch, file_path="model_checkpoint.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, file_path)
    

save_checkpoint(net, optimizer, epoch, file_path="model_checkpoint.pth")
