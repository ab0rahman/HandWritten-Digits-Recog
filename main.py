import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load and preprocess the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Define the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Save the model
model.save('handwritten.model')

# Load the model
model = tf.keras.models.load_model('handwritten.model')

def process_and_predict_image(image_path):
    """Function to preprocess the image and make a prediction."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image {image_path} not found.")
        
        # Resize the image to 28x28
        img = cv2.resize(img, (28, 28))
        
        # Invert and normalize the image
        img = np.invert(img)
        img = img / 255.0
        
        # Reshape for prediction
        img = img.reshape(1, 28, 28)
        
        # Make prediction
        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)
        
        # Display the image and prediction
        print(f"The digit might be {predicted_digit}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

# Process images from the digits directory
image_number = 1
while os.path.isfile(f"digits/digit{image_number}.jpg"):
    process_and_predict_image(f"digits/digit{image_number}.jpg")
    image_number += 1
