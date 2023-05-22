import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Defining data paths
dataset_path = "VictoriaTrabWorm#9563/chest_xray"

train_folder = os.path.join(dataset_path, "train")
test_folder = os.path.join(dataset_path, "test")
val_folder = os.path.join(dataset_path, "val")

# Function to resize images
def resize_image(image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, (desired_width, desired_height))
    return resized_image

# Loading the training dataset
train_images = []
train_labels = []
train_categories = ["NORMAL", "PNEUMONIA"]
for category in train_categories:
    category_folder = os.path.join(train_folder, category)
    file_names = os.listdir(category_folder)
    for file_name in file_names:
        image_path = os.path.join(category_folder, file_name)
        image = cv2.imread(image_path)
        train_images.append(image)
        train_labels.append(category)

# Loading the testing dataset
test_images = []
test_labels = []
test_categories = ["NORMAL", "PNEUMONIA"]
for category in test_categories:
    category_folder = os.path.join(test_folder, category)
    file_names = os.listdir(category_folder)
    for file_name in file_names:
        image_path = os.path.join(category_folder, file_name)
        image = cv2.imread(image_path)
        test_images.append(image)
        test_labels.append(category)

# Loading the validation dataset
val_images = []
val_labels = []
val_categories = ["NORMAL", "PNEUMONIA"]
for category in val_categories:
    category_folder = os.path.join(val_folder, category)
    file_names = os.listdir(category_folder)
    for file_name in file_names:
        image_path = os.path.join(category_folder, file_name)
        image = cv2.imread(image_path)
        val_images.append(image)
        val_labels.append(category)

label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.transform(test_labels)
val_labels = label_encoder.transform(val_labels)

# Defining the desired width and height for the resized images
desired_width = 224
desired_height = 224

# Function to resize images
def resize_images(images):
    resized_images = []
    for image in images:
        resized_image = cv2.resize(image, (desired_width, desired_height))
        resized_images.append(resized_image)
    return resized_images

# Resizing the images
train_images = resize_images(train_images)
test_images = resize_images(test_images)
val_images = resize_images(val_images)

# Converting the lists to NumPy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)
test_images = np.array(test_images)
test_labels = np.array(test_labels)
val_images = np.array(val_images)
val_labels = np.array(val_labels)


# Building and training the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Defining the model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(desired_width, desired_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(train_images, train_labels, epochs=5, validation_data=(val_images, val_labels))