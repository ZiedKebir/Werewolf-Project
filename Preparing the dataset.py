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

def PIL_to_array(Pil_image):
    """
    Parameters
    ----------
    Pil_image : PIL.Image.Image
        An Image
    Returns
    -------
        An Image with the type np.ndarray
    """
    return np.array(Pil_image)


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


def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)


def data_augmentation(image_name:str):
    """
    Parameters
    ----------
    image_name : str
        Contains the name of the image as stored in the source folder. 
        The name should also include the extension .jpg
    Returns
    -------
    augmented_image : np.ndarray
        Applies changes like rotaions, bluring... to the image in order to augment the size of 
        our dataset

    """    
    image = Image.open('Images/'+image_name)
    #Convert the array to a Pil image type
    image = PIL_to_array(image)
    #Resize the image before transforming it
    image = resize_image(image)

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
image = data_augmentation('Chasseur.jpg')
visualize(image)
image.shape



cv2img = cv2.imread('Images/'+'Sorciere.jpg')


visualize(cv2img)


type(cv2img)

image = Image.open('Images/'+'Sorciere.jpg')
visualize(image)

np.array(image)

type(image)

type(np.array(image))

image


from PIL import Image 
 

cv2img = cv2.imread('Images/'+'Test.jpg')


resize_image(cv2img)

"""

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


file = fs.find_one({'Role':'Ange' })
image = file.read()
image
Pil_Image = Image.open(io.BytesIO(image))
np.asarray(Pil_Image).shape

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
import keras
import tensorflow as tf
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
assert train_images.shape == (370,370, 370, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)


x_train_resized = tf.image.resize(x_train[0], (370, 370))
x_test_resized = tf.image.resize(x_test, (370, 370))



t = tf.image.resize(image, (370, 370))

visualize(t)


#Select 2000 Images randomly 

img = Image.fromarray(train_images[0])

res_img = tf.image.resize(img, (225, 225))


visualize(res_img)

img_2 = Image.fromarray(res_img)



import tensorflow_datasets as tfds
import tensorflow as tf

# Load dataset
dataset, info = tfds.load('cifar10', with_info=True, as_supervised=True)

# Resize images to 370x370
def preprocess_image(image, label):
    image = tf.image.resize(image, [370, 370])
    return image, label

dataset = train_images.map(preprocess_image)

a,b= preprocess_image(train_images[0],'Cast')
a

visualize(train_images[0])

