import pymongo
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
import gridfs
from PIL import Image, ImageFilter
import io
import random
import albumentations as A
import tensorflow as tf
import keras
os.chdir("C:/Users/ziedk/OneDrive/Bureau/Data Science Projects/Werewolf-Project")

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['Werewolf']
fs = gridfs.GridFS(db)

#Delete Files
result = db.fs.files.delete_many({"Role": "Test"})  # Replace 'collection_name' with your actual collection name
print(f"Deleted {result.deleted_count} documents.")

myquery = {'Role':'Chasseur'}

count = 0
for i in db.fs.files.find():
    count +=1 

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


def store_image_and_augment(Image_Name:str):
    Role_Name = Image_Name.split('.')[0]
    #Open the image in read-only format.
    with open('Images/'+Image_Name, 'rb') as f:
        contents = f.read()

    #Now store/put the image via GridFs object.
    fs.put(contents, filename=Image_Name,Role=Role_Name)

    
    for i in range(1,1000):
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
    count = 0
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

"""
file = fs.find_one({'Role':'Ange' })
image = file.read()
image
Pil_Image = Image.open(io.BytesIO(image))
np.asarray(Pil_Image).shape
"""
## Get More Images 
#Using scikit-learn
from sklearn.datasets import load_digits
digits = load_digits()
images = digits.images
labels = digits.target


#Using Tensorflow
import keras
import tensorflow as tf
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

def store_non_card_single_image(img:np.array):
        img = resize_image(img)
        img = Image.fromarray(img)

        # Create a BytesIO buffer to store the image in bytes
        byte_io = io.BytesIO()

        # Save the image to the buffer in a specific format (e.g., JPEG, PNG)
        img.save(byte_io, format='JPEG')

        # Get the byte data
        img_bytes = byte_io.getvalue()

        fs.put(img_bytes, filename='non card',Role='Other')
        return None
import time
def store_non_card_all_images(images_array:np.array(np.array)):
    for i in images_array:
        store_non_card_single_image(i)
        time.sleep(2)
        return None

def Rand(start, end, num):
    res = []
    for j in range(num):
        res.append(random.randint(start, end)) 
    return res

non_card_images_to_store_index = Rand(0,50000,300)
non_card_images_to_store = train_images[non_card_images_to_store_index]

for i in non_card_images_to_store:
    store_non_card_single_image(i)

    
"""
Testing and validation dataset
"""

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['Werewolf_Test_validation']
fs = gridfs.GridFS(db)


#### Card Images 

#Load and store the card images and for each image perform some rotation and change in the saturation
def store_image_and_modify(Image_Name:str):
    Role_Name = Image_Name.split('.')[0]
    #Open the image in read-only format.
    with open('Images/'+Image_Name, 'rb') as f:
        contents = f.read()

    #Now store/put the image via GridFs object.
    fs.put(contents, filename=Image_Name,Role=Role_Name)

    for i in range(1,200):
        augmented_image = data_rotation_saturation_modif(Image_Name)
        
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


def data_rotation_saturation_modif(image_name):
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
    
    #Rotate Image
    rotation_angle = random.uniform(-180, 180)  # Random angle between -180 and 180 degrees
    rotated_image = image.rotate(rotation_angle)    
    # Apply blurring
    blurred_rotated_image = rotated_image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0,10)))  # Change radius for stronger blur
    
    #Convert the array to a Pil image type
    image = PIL_to_array(blurred_rotated_image)
    #Resize the image before transforming it
    image = resize_image(image)
    
    #visualize(augmented_image)
    return image
    


def store_all_images_and_modif():
    for Image_Name in os.listdir("Images"):
        print(Image_Name)
        store_image_and_modify(Image_Name)
        

store_all_images_and_modif()



### Stare non card images
from datasets import load_dataset, Image

dataset = load_dataset("beans", split="train")
image = dataset[250]["image"]
PIL_to_array(image).shape

image.show()

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['Werewolf_Test_validation']
fs = gridfs.GridFS(db)


#Store first_batch_of_images: 
dataset_1 = load_dataset("beans", split="train")
#(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
#dataset_2 = test_images


image = Image.fromarray((resize_image(test_images[0])).astype(np.uint8))  # Assuming tensor values are normalized
image.show()


test_images[0].shape

def test_validation_non_card_images():
    for i in range(0,150):
        array_image_1 = PIL_to_array(dataset[i]["image"])
        resized_image_1 = resize_image(array_image_1)
        #resized_image_2 = resize_image(test_images[i])
        # Store the images in mongodb- database Werewolf_Test_validation
        store_non_card_all_images(resized_image_1)
        #store_non_card_all_images(resized_image_2)
    return None


test_validation_non_card_images()


array_image_1 = PIL_to_array(dataset[0]["image"])
array_image_1.shape
resized_image_1 = resize_image(array_image_1)
resized_image_1.shape


resized_image_2 = resize_image(test_images[0])
resized_image_2.shape




