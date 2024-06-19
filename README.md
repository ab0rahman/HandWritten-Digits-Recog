
# Handwritten Digit Recognition

# Working:
The code is designed to train a neural network to recognize handwritten digits using the MNIST dataset and then use the trained model to predict digits from new images in a specified directory. The process includes loading the dataset, building and training a model, saving and loading the model, preprocessing new images, and making predictions.

## Installation

Start by importing the required libraries.

`OpenCV`
`matplotlib`
`tensorflow`

To run this project, you will need to add the following.

```bash
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

```
**OpenCV:** cv2 is a module provided by OpenCV (Open Source Computer Vision Library), a highly optimized library with a focus on real-time applications and computer vision.

**matplotlib:** Matplotlib is a library for creating interactive visualizations in Python. It is widely used in data science, machine learning, and scientific research to visualize data, create plots, and generate graphs.

**tensorflow:** TensorFlow is an library developed by the for numerical computation and machine learning. It provides a flexible platform for building and deploying machine learning models, especially deep learning models. 
    

## Load and preprocess the MNIST dataset

The MNIST database of handwritten digits is one of the most popular image recognition datasets. It contains 60k examples for training and 10k examples for testing
It is included in the `tf.keras.datasets` module.

```bash
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
```

`x_train` and `x_test` are arrays containing the image data.

`y_train` and `y_test` are arrays containing the corresponding labels.

The pixel values of the images are originally in the range [0, 255]. 
`tf.keras.utils.normalize()`  Normalizing these values to the range [0, 1] can help improve the performance and training speed of the neural network.
Normalization involves scaling the pixel values by dividing by 255.

## Define the model architecture

```bash
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

Sequential model `(tf.keras.models.Sequential)`:
Sequential is a linear stack of layers. It allows building a model layer by layer, where each layer has exactly one input tensor and one output tensor.

In this case, the model is a simple stack of layers, with the input passing through each layer sequentially to the output.

#### Flatten layer `(tf.keras.layers.Flatten(input_shape=(28, 28)))`:
Flatten layer transforms the image format from a 2D array (28x28 pixels) to a 1D array (28x28 = 784 pixels).
This layer has no parameters to learn; it just reformats the data.

#### Dense layers `(tf.keras.layers.Dense(128, activation='relu'))`:
Dense means a fully connected layer where every neuron is connected to every neuron in the previous layer.

The first dense layer has 128 neurons. It uses a **Rectified Linear Unit (ReLU)**  activation function that introduces non-linearity into the model and allows it to learn complex patterns in the data.

The second Dense layer also has 128 neurons with ReLU activation. These layers are intermediate or "hidden" layers that help the model learn data representations.

#### Output layer `(tf.keras.layers.Dense(10, activation='softmax'))`:
The final Dense layer has 10 neurons, which corresponds to the 10 classes (digits 0-9) in the MNIST dataset.
It uses a *softmax activation function* that outputs a probability distribution into 10 classes. Each neuron in this layer represents the probability that the input image belongs to a certain class of digits.

<img src="https://raw.githubusercontent.com/ab0rahman/HandWritten-Digits-Recog/f569941e7343ed71eff104c6db8fc94b862476a6/digits/neutral-network-diagram.svg" width="500" height="auto">

### Summary:
### Input layer: 
The merge layer transforms a 2D input image into a 1D array.

### Hidden layers:
Two dense layers, each with 128 neurons, using ReLU activation to learn complex patterns in the data.
### Output layer:
Dense layer with 10 neurons and softmax activation for output probability for each digit class.

## Compile the model

```bash
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```


#### Optimizer (optimizer='adam'):

Optimizers are algorithms that adjust the weights and learning rate during training to reduce the loss function. 'adam' stands for **Adaptive Moment Estimation.**
Adam is an advanced optimizer that combines the benefits of two other popular optimizers: AdaGrad and RMSProp.

#### Loss function `(loss='sparse_categorical_crossentropy'):`

The loss function measures how well the model performs on the training data and guides the optimization process by calculating the loss gradient with respect to the model weights.
'sparse_categorical_crossentropy' is used when the labels are integers (as in the MNIST dataset), where each integer represents a different class.

#### Metrics `(metrics=['accuracy'])`:
Metrics are used to track training and evaluation steps. **"accuracy"** is a common metric used for classification problems.
During training and evaluation, the accuracy of the model on the training data will be calculated and displayed.

## Train the model

Involves feeding the training data (x_train and y_train) into the neural network model for a specified number of epochs. 
```bash
model.fit(x_train, y_train, epochs=5)
```

**x_train:** This is the input training data, typically a numpy array or a TensorFlow tensor, containing the images. In this case, x_train is a normalized array of shape (60000, 28, 28) for 60,000 images each of size 28x28 pixels.

**y_train:** This is the target training data, representing the labels or categories to which each training example belongs. For the MNIST dataset, y_train is an array of integers ranging from **0** to **9**, indicating the digit in each corresponding image.

**epochs=5:** An epoch is one complete pass through the entire training dataset. Setting epochs=5 means that the model will iterate over the entire x_train and y_train datasets 5 times during training.

#### Steps involved During Training:
- Forward Pass
- Loss Calculation
- Backward Pass (Backpropagation)
- Metrics Calculation
- Epoch Progress

## Save the model
The save method serializes the model architecture, weights, and configuration into a single HDF5 file.
```bash
model.save('handwritten.model')
```
`handwritten.model` This is the filename or path where the trained model will be saved.

The model will be saved in the Hierarchical Data Format (HDF5) file format, which is efficient for storing large numerical data.

## Load the model

Loading a saved model allows you to reuse a previously trained neural network without having to train it again from scratch.

```bash
model = tf.keras.models.load_model('handwritten.model')
```


### Function to preprocess the image and make a prediction.
**Read Image:** `cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)` reads the image from image_path in grayscale.

```bash
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
```
**Image Validation:** Checks if the image is valid, raising a ***FileNotFoundError*** if not.
```bash
if img is None:
  raise FileNotFoundError(f"Image {image_path} not found.")
```
**Resize Image:** Resizes the image to a standard size of 28x28 pixels using `cv2.resize(img, (28, 28))`.
```bash
img = cv2.resize(img, (28, 28))
```
**Invert and normalize the image:** Inverting an image means reversing the intensity values of its pixels In grayscale images:

A pixel value of `0`(black) becomes `255` (white).

A pixel value of `255` (white) becomes `0`(black).

#### Normalization 
Scales the pixel values of the image to a range of `[0, 1]`. This standardization helps in improving the convergence of the neural network during training and can also speed up the training process. By normalizing pixel values to a range between 0 and 1, the model becomes more stable and learns better representations of the data.
```bash
img = np.invert(img)
img = img / 255.0
```

**Reshape for prediction:** Convolutional Neural Networks (CNNs) expect input tensors in the form `(batch_size, height, width, channels)`.

*Batch Size* Reshaping with a batch size of 1 (img.reshape(1, 28, 28)) ensures that the model receives a single image in a format consistent with how it was trained.
```bash
img = img.reshape(1, 28, 28)
```

**Make prediction:** The predict method applies the trained neural network model to the input image, producing an output that consists of probability scores for each possible class representing the likelihood of the image belonging to a specific digit class (0-9).
```bash
prediction = model.predict(img)
predicted_digit = np.argmax(prediction)
```
*argmax* function to find the index of the highest probability in the prediction array.

**Display the image and prediction:**  prints a message indicating the predicted digit based on the model's output.

`plt.imshow()` displays an image using Matplotlib.

`cmap=plt.cm.binary` specifies that the image should be displayed in grayscale (binary colormap)
```bash
print(f"The digit might be {predicted_digit}")
plt.imshow(img[0], cmap=plt.cm.binary)
plt.show()
```
**Exception:** 
If any exception occurs within the try block, Python catches it and executes the corresponding except block.
```bash
except Exception as e:
    print(f"Error processing image {image_path}: {e}")
```

## Process images from the digits directory
`import os` Imports the Python os module, which provides functions for interacting with the operating system.

`image_number = 1` Initializes a variable image_number to 1, starting the loop from the first image `(digit1.jpg)`.

`while` loop continues iterating as long as there exists a file named `digits/digit{image_number}.jpg` in the directory.

`os.path.isfile()` checks if the specified path `(digits/digit{image_number}.jpg)` is a file that exists.

`process_and_predict_image()` to handle image preprocessing (resizing, inverting, normalizing) and making predictions using a pre-trained machine learning model.

`image_number += 1` Increments image_number by 1 in each iteration, moving to the next image

```bash
image_number = 1
while os.path.isfile(f"digits/digit{image_number}.jpg"):
    process_and_predict_image(f"digits/digit{image_number}.jpg")
    image_number += 1
```


# Collaborators
  Abdur Rahman | [GitHub](https://github.com/ab0rahman) | [Email](mailto:letsmail.him@gmail.com) | 
