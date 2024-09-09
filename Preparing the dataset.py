import pymongo
import os

os.chdir("C:/Users/ziedk/OneDrive/Bureau/Data Science Projects/Werewolf-Project")

# open connection at port 27017
client = pymongo.MongoClient('localhost', 27017)
# create db tutorial
mydb = client["Werewolf"]
# create collection example
collection = mydb["Images"]


# Open the folder containing the card images
from PIL import Image
import requests

im = Image.open("Images/Chasseur.jpg")



import albumentations as A
import cv2
import matplotlib.pyplot as plt

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
    
    #Open the image in read-only format.
    with open('Images/'+Image_Name, 'rb') as f:
        contents = f.read()

    #Now store/put the image via GridFs object.
    fs.put(contents, filename=Image_Name)

    
    for i in range(1,11):
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

        fs.put(img_bytes, filename=Image_Name+str(i))
    return None


def store_all_images_and_modif():
    for Image_Name in os.listdir("Images"):
        print(Image_Name)
        store_image_and_augment(Image_Name)



store_all_images_and_modif()

        
        
        

        
        
augmented_image = data_augmentation("Chasseur.jpg")

with open('Images/Chasseur.jpg', 'rb') as f:
    contents = f.read()


import io
type(contents)

Pil_image = Image.fromarray(augmented_image)

Pil_image.save(byte_io, format='JPEG')

# Create a BytesIO buffer to store the image in bytes
byte_io = io.BytesIO()
img_bytes = byte_io.getvalue()



# Assuming img_array is your NumPy array (image data)
# Convert the NumPy array back to an image
img = Image.fromarray(augmented_image)

# Create a BytesIO buffer to store the image in bytes
byte_io = io.BytesIO()

# Save the image to the buffer in a specific format (e.g., JPEG, PNG)
img.save(byte_io, format='JPEG')

# Get the byte data
img_bytes = byte_io.getvalue()

# Now img_bytes contains the image data in bytes
print(type(img_bytes))  # Should print <class 'bytes'>




fs.put(Pil_image, filename="PIL")



for i in os.listdir("Images"):
    store_images(i)




from pymongo import MongoClient
import gridfs
from PIL import Image

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['image_database']
fs = gridfs.GridFS(db)

def store_image(image_path):
    with open(image_path, 'rb') as f:
        data = f.read()
        file_id = fs.put(data, filename=image_path)
        print(f"Image stored with ID: {file_id}")

# Store an image
store_image('Images/Chasseur.jpg')


def retrieve_image(file_id, output_path):
    out_file = fs.get(file_id).read()
    with open(output_path, 'wb') as f:
        f.write(out_file)
    print(f"Image retrieved and saved to: {output_path}")

# Retrieve and save an image
file_id = '66defa1d6660bc52e9672171'  # Replace with the actual file ID
retrieve_image(file_id, 'Test.jpg')



for grid_out in fs.find():
    print(f"Filename: {grid_out.filename}, File ID: {grid_out._id}")
    
    
    
file_id = '66def65a6660bc52e9672169'  # Replace with the actual ID you got from the print statement above
try:
    out_file = fs.get(file_id).read()
    with open('retrieved_image.jpg', 'wb') as f:
        f.write(out_file)
    print("Image retrieved successfully")
except gridfs.errors.NoFile:
    print("File not found!")    
    





file = fs.find_one({'filename': 'file'})
image = file.read()

print(image)



from PIL import Image
import numpy as np

# Open the image file
with open('Images/Chasseur.jpg', 'rb') as f:
    img = Image.open(f)
    img.load()  # Ensure the image is loaded before closing the file

# Convert the image to a NumPy array
img_array = np.array(img)

print(img_array)



img = Image.fromarray(img_array)

img