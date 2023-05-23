# Self-assigned project: Chest X-Ray Image Classification

This project aims to classify chest X-ray images into two categories: Pneumonia and Normal. The dataset consists of 5,863 X-ray images obtained from pediatric patients. The goal is to build and evaluate  a model that can identify pneumonia cases from chest X-ray images obtained from a dataset from kaggle.

## The purpose
The purpose of this project is to develop an image classification system using machine learning techniques to aid in the diagnosis of pneumonia. By training a model on the provided dataset, I aim to achieve accurate classification performance and gain insights into the interpretability of the models.

## Steps
Data Loading: Loading the dataset of chest X-ray images from the provided file structure. The dataset is organized into train, test, and validation folders, each containing subfolders for the different categories (e.g., "NORMAL" and "PNEUMONIA").
Data Preprocessing: Preprocessing the images by resizing them to a desired width and height. Converting the images to NumPy arrays and performing label encoding on the target labels (categories). 
Model Definition: Defining the model architecture for image classification. Using a Convolutional Neural Network (CNN) approach.
Model Training: Compiling the model by specifying the optimizer, loss function, and evaluation metrics. Training the model using the training dataset and validating it using the validation dataset.
Model Evaluation: After training the model, I made predictions on the test set and calculated the classification metrics (accuracy, precision, recall, and F1 score) using the predictions and true labels. Generating a classification report to summarize the performance of the model.
Results: Saving the accuracy plot to a file for future reference. Similarly, plotting the training and validation loss and saving it to a separate file. These plots provide insights into the model's learning progress and potential overfitting.

## Dependencies
The project has been run through Coder Python 1.77.3 via UCloud. 
Libraries: OpenCV, scikit-learn, TensorFlow (see requirements.txt)

## Getting Started
Clone this repository: git clone https://github.com/VictoriaTrabW/visual-project-VictoriaTrabW
Install the required dependencies: pip install -r requirements.txt
The Chest X-Ray Images dataset can be accessed through kaggle at: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
I placed my dataset in my member files on UCloud under “Member Files: VictoriaTrabWorm#9563”

## Results
The project will provide insights into the performance of a trained model for classifying chest X-Ray images. The evaluation metrics such as the history of valuation loss and classification report will be available in the “out” folder.

Note: I had trouble setting up a virtual environment because it created a lot of file changes, which git could not handle. So unfortunately there is no venv.

## Reflections
Based on the classification report, it seems that there is an issue with the performance of the model in classifying the "NORMAL" category. The precision, recall, and F1-score for the "NORMAL" category are all 0.00, indicating that the model did not correctly identify any instances of the "NORMAL" class.
On the other hand, the model performs relatively well in classifying the "PNEUMONIA" category, with a precision of 0.62, recall of 1.00, and F1-score of 0.77. This suggests that the model can effectively identify instances of pneumonia from the chest X-ray images.
The overall accuracy of the model on the test set is 0.62, meaning that it correctly predicted the class for approximately 62% of the samples.
Going forward it is important to investigate why the model is struggling to classify the "NORMAL" category and consider potential improvements, such as adjusting the model architecture, increasing the training data, or using data augmentation techniques.
In summary, the model shows good performance in detecting pneumonia cases but needs further improvement in accurately classifying normal cases.
