# Facial Expression Recognition Project

## Overview
Facial expression recognition (FER) is a fascinating project in the field of machine learning and computer vision. The goal is to develop a system that can automatically identify human emotions from facial expressions captured in images or videos. Convolutional Neural Networks (CNNs) are a popular choice for this task due to their ability to effectively learn spatial hierarchies and patterns in images.

## Dataset
Here I worked with a "Human Face Emotions" dataset from Kaggle. This dataset contains expressions of humans such as happiness, sadness, anger.

Link: https://www.kaggle.com/datasets/sanidhyak/human-face-emotions

## Technologies Used
- Python
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- imgaug

## Project Structure
- `data/`: Contains the dataset used for training and testing.
- `notebooks/`: Jupyter notebooks for data exploration, model development, and evaluation.
- `src/`: Python scripts for preprocessing, model training, and inference.
- `models/`: Pre-trained models or saved models.
- `results/`: Results and evaluation metrics.

## Model Architecture
The facial expression recognition model is based on a Convolutional Neural Network (CNN) architecture.

### Pretrained model
A pretrained model refers to a model that has been trained on a large dataset for a specific task, such as image classification, object detection etc. In this project we used **VGG16** model as a feature extractor.

## Results
- The project achieves promising results in terms of accuracy and generalization performance.
- Detailed evaluation metrics and performance analysis are provided in the directory.
