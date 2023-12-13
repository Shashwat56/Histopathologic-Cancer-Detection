# Histopathologic-Cancer-Detection
Implemented a model to identify metastatic tissue in histopathologic scans of lymph node sections


Problem Statement: In this dataset, you are provided with a large
number of small pathology images to classify. Files are named with an
image id. The train_labels.csv file provides the ground truth for the
images in the train folder. You are predicting the labels for the images in
the test folder. A positive label means that there is at least one pixel of
tumor tissue in the center 32x32px area of a patch. Tumor tissue in the
outer region of the patch does not influence the label. This outer region is
provided to enable fully-convolutional models that do not use
zero-padding, to ensure consistent behavior when applied to a
whole-slide image.

Abstract:
The early detection of cancer is crucial for its effective treatment.
Histopathologic examination is a gold standard for diagnosing cancer,
but it requires highly trained pathologists and is time-consuming. With the
advances in machine learning and computer vision, automated cancer
detection systems have shown great potential to assist pathologists and
improve diagnostic accuracy. In this project, we present a histopathologic
cancer detection system using image segmentation and machine
learning.

About the dataset:
The train data we have here contains 220,025 images and the test set
contains 57,468 images. The original PCam dataset contains duplicate
images due to its probabilistic sampling, however, the version presented
on Kaggle does not contain duplicates. A positive label indicates that the
center 32x32px region of a patch contains at least one pixel of tumor
tissue. Tumor tissue in the outer region of the patch does not influence the
label.

Introduction
Cancer is a leading cause of death worldwide, and early detection is
crucial for its successful treatment. Histopathologic examination is a gold
standard for diagnosing cancer, but it requires highly trained pathologists
and is time-consuming. The use of machine learning and computer vision
has shown great potential to assist pathologists and improve diagnostic
accuracy. In this project, we aim to develop a histopathologic cancer
detection system using image segmentation and machine learning.

Data Pre-Processing and Analysis
Importing the required libraries to use TensorFlow and Keras for building
a convolutional neural network (CNN) model for image classification.
This code sets up the directory structure for the training and validation
sets. Defining the source directories (traindir and valdir) and the
destination directories for the training and validation sets. The
traincancer, trainnoncancer, valcancer, and valnoncancer directories are
subdirectories of traindir and valdir, respectively. 

Dimensionality Reduction using LDA
The code applies Linear Discriminant Analysis (LDA) to the train and
validation image datasets. LDA is a dimensionality reduction technique
that finds a projection of the data that maximizes the separation between
classes.
In this case, LDA is applied to the flattened image data, and the resulting
transformed data has a reduced number of dimensions (in this case, just
one dimension). The code first defines an image data generator for the
datasets, and then loads the images in batches and applies LDA to each
batch. The transformed data is concatenated into a single array for each
dataset, and the explained variance ratio of the LDA is printed.
This code defines a function normalise_process that takes an image and
its corresponding label as input. The image is first cast to a float32 tensor
and normalized by dividing each pixel value by 255 (the maximum pixel
value for an 8-bit image). The function then returns the normalized image
and its label.
Then we define a convolutional neural network model using Keras for
image classification on a binary classification task.
The model architecture consists of three convolutional layers, each
followed by a batch normalization layer and a max pooling layer. The
convolutional layers use 32, 64, and 128 filters respectively, with a 3x3
kernel size and ReLU activation function. The max pooling layers use a
2x2 pool size with stride 2.
The output of the final max pooling layer is flattened and passed to two
fully connected layers with 128 and 64 units respectively, each followed by
a dropout layer with a dropout rate of 0.1. Finally, a dense layer with one
unit and sigmoid activation function is used to output the binary
classification result.

Image Segmentation: U-Net Architecture
The provided code implements a function called "segment_image" which
uses a pre-trained U-Net model to segment histopathologic cancer
images. The function takes four parameters: the path to the model file,
the path to the input image file, the desired shape of the input image, and
a threshold value for the binary mask.

Finally we got an accuracy of 91.99%.

