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

os.chdir("C:/Users/ziedk/OneDrive/Bureau/Data Science Projects/Werewolf-Project")

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['Werewolf']
fs = gridfs.GridFS(db)



def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)


def data_augmentation(image_name:str):
    """
    Takes an image and modifies it randomly
    Returns
    -------
    An Image

    """
    
    image = cv2.imread('Images/'+image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = A.Compose(
        [A.CLAHE(),
         A.RandomRotate90(),
         A.Transpose(),
         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50,
                            rotate_limit=45, p=.75),
         A.Blur(blur_limit=3),
         A.OpticalDistortion(),
         A.GridDistortion(),
         A.HueSaturationValue()])
    
    augmented_image = transform(image=image)['image']
    #visualize(augmented_image)
    return augmented_image


"""

import numpy as np

with open('Images/Chasseur.jpg', 'rb') as f:
    contents = f.read()
img_array = np.array(contents)

print(img_array)
image = cv2.imread('Images/'+image_name)
image = cv2.cvtColor(contents, cv2.COLOR_BGR2RGB)

image

#Open the image in read-only format.
with open('Images/Chasseur.jpg', 'rb') as f:
    contents = f.read()



import matplotlib.image as img
image = img.imread('Images/Chasseur.jpg')

fs.put(image, filename='To Delete')



len(image)

contents

os.listdir("Images")

"""

def store_image_and_augment(Image_Name:str):
    Role_Name = Image_Name.split('.')[0]
    #Open the image in read-only format.
    with open('Images/'+Image_Name, 'rb') as f:
        contents = f.read()

    #Now store/put the image via GridFs object.
    fs.put(contents, filename=Image_Name,Role=Role_Name)

    
    for i in range(1,100):
        augmented_image = data_augmentation(Image_Name)
        
        # Assuming img_array is your NumPy array (image data)
        # Convert the NumPy array back to an image
        img = Image.fromarray(augmented_image)

        # Create a BytesIO buffer to store the image in bytes
        byte_io = io.BytesIO()

        # Save the image to the buffer in a specific format (e.g., JPEG, PNG)
        img.save(byte_io, format='JPEG')

        # Get the byte data
        img_bytes = byte_io.getvalue()

        fs.put(img_bytes, filename=Image_Name+str(i),Role=Role_Name)
    return None



def store_all_images_and_modif():
    for Image_Name in os.listdir("Images"):
        print(Image_Name)
        store_image_and_augment(Image_Name)

store_all_images_and_modif()

#Get All the files stored in MongoDb
for grid_out in fs.find():
    print(f"Filename: {grid_out.filename}, File ID: {grid_out._id}")
    
    

    
    
def byte_to_image(mongodb_file_name:str):
    file = fs.find_one({'filename': mongodb_file_name})
    image = file.read()
    Pil_Image = Image.open(io.BytesIO(image))
    return Pil_Image


file = fs.find_one({'Role': 'mongodb_file_name'})

## Get More Images 
#Using scikit-learn
from sklearn.datasets import load_digits
digits = load_digits()
images = digits.images
labels = digits.target

image_2D = images[0]

image_3D = np.expand_dims(image_2D, axis=-1)

image_3D

#Using Tensorflow
import tensorflow as tf
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()


#Select 2000 Images randomly 


train_images[0].shape


array([[[ 0.],
        [ 0.],
        [ 5.],
        [13.],
        [ 9.],
        [ 1.],
        [ 0.],
        [ 0.]],

       [[ 0.],
        [ 0.],
        [13.],
        [15.],
        [10.],
        [15.],
        [ 5.],
        [ 0.]],

       [[ 0.],
        [ 3.],
        [15.],
        [ 2.],
        [ 0.],
        [11.],
        [ 8.],
        [ 0.]],

       [[ 0.],
        [ 4.],
        [12.],
        [ 0.],
        [ 0.],
        [ 8.],
        [ 8.],
        [ 0.]],

       [[ 0.],
        [ 5.],
        [ 8.],
        [ 0.],
        [ 0.],
        [ 9.],
        [ 8.],
        [ 0.]],

       [[ 0.],
        [ 4.],
        [11.],
        [ 0.],
        [ 1.],
        [12.],
        [ 7.],
        [ 0.]],

       [[ 0.],
        [ 2.],
        [14.],
        [ 5.],
        [10.],
        [12.],
        [ 0.],
        [ 0.]],

       [[ 0.],
        [ 0.],
        [ 6.],
        [13.],
        [10.],
        [ 0.],
        [ 0.],
        [ 0.]]])

