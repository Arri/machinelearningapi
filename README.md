*Version: 1.1* <br />
*First Version: Jan 2020 by Arasch Lagies* <br />
*Last Update: 07/29/2021 by Arasch Lagies* <br />

# FPGA Based CNN Accelerator Model Development

This repository contains in its new version the following folders:
1. Documentation
2. ML_Library


## 1. Documentation
This folder contains newer and older documentation material, presentations and charts.

## 2. data
This contains training and testing data. 
Base-training is done using the MNIST dataset.

## 3. utils
These are utilily functions for testing and evaluating model-library and the generated models.

## 4.  _ML_Library
This machine learning library was developed at   and allows the sequential construction of CNN models. The model layers are written from scratch using numpy for most calculations. The model design is kept similar to the TensorFlow2 (Keras) Sequential API.

For model construction available are folling layers:
- Conv2D()
- MaxPool2D()
- Flatten()
- Dense()
- Normalize()

This library is useful as a reference for the model and HW development (e.g. for the development of a FPGA-based CNN accelerator), as it allows to see all details of the model and how the data is being processed.

The main fuction for model design, training and evaluation is the Python script *train_cnn.py*.
The main fuction for inference is the Python script *predict.py*.
The script *minist_Test.py* is a test function to see how the trained model works on new hand-generated data.

The main functions for this library are in the folder *src/*.
