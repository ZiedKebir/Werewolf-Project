# Project Description

This project uses a Convolutional Neural Network (CNN) to classify images of cards from the Werewolf game. The model is designed to recognize different cards (roles) based on images using supervised learning techniques. The main goal of the project is to set up a complete pipeline for data processing, model training, and performance evaluation.

# Key Features
* ### Data Preprocessing:

  - Loading images stored in MongoDB using GridFS.
  - Converting images into NumPy arrays, resizing them to 225x225 pixels, and handling errors during image processing.
  - Encoding card role labels using one-hot encoding.

* ### Convolutional Neural Network (CNN) Model:

  - CNN architecture with multiple convolutional layers, pooling layers, and fully connected layers.
  - Using ReLU activation function and CrossEntropyLoss for multi-class classification.
  
* ### Training and Evaluation:

  - Model training with a training dataset and evaluation with a validation/test set.
  - Tracking model performance in terms of accuracy and loss using TensorBoard.

* ### Using PyTorch:

  - Training on GPU (if available) with optimization via the Adam algorithm.
  - Saving model checkpoints for easy resumption of training.

* ### Additional Features
  - Data Loading from MongoDB: Images are stored in a MongoDB database using GridFS, allowing efficient management of large datasets.
  - TensorBoard for Visualization: The project uses TensorBoard to track metrics during training (loss and accuracy).


# Prerequisites
 - PyTorch
 - MongoDB with GridFS for image storage
 - TensorBoard for visualizing results
 - PIL and NumPy for image preprocessing

# Author

Zied KEBIR
