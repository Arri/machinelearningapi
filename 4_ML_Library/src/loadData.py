############################################################################
# Load training and testing data with their labels for supervised learning
#
# Author: Arasch Lagies
# First Version: 4/5/2020
# Last Update: 8/28/2020
#
# Call:
############################################################################
import os
import pickle
from utils import *


TRAINDATAPATH = "./data/train/"
TESTDATAPATH  = "./data/test/"
TRAINX        = "fashionMNIST_train-images-idx3-ubyte.gz"   #"train-images-idx3-ubyte.gz"   #"train-images-idx3-ubyte_reshaped32x32.pkl"        #"train-images-idx3-ubyte.gz"
TRAINY        = "fashionMNIST_train-labels-idx1-ubyte.gz"   #"train-labels-idx1-ubyte.gz"
TESTX         = "fashionMNIST_t10k-images-idx3-ubyte.gz"    #"t10k-images-idx3-ubyte.gz"   #"t10k-images-idx3-ubyte_reshaped32x32.pkl"         #"t10k-images-idx3-ubyte.gz"
TESTY         = "fashionMNIST_t10k-labels-idx1-ubyte.gz"    #"t10k-labels-idx1-ubyte.gz"
TRAINDATASIZE = 50000
TESTDATASIZE  = 10000
IMAWIDTH      = 28   #32    #28    #32
IMAGEDEPTH    = 1

class loadData:
    def __init__(self, trainpath=TRAINDATAPATH, testpath=TESTDATAPATH, trainData=TRAINX, trainLabels=TRAINY, testData=TESTX, testLabels=TESTY,
                        trainSize=TRAINDATASIZE, testSize=TESTDATASIZE, imageWidth=IMAWIDTH, imageDepth=IMAGEDEPTH):
        self.trainXpath = os.path.join(trainpath, trainData)
        self.trainYpath = os.path.join(trainpath, trainLabels)
        self.testXpath  = os.path.join(testpath, testData)
        self.testYpath  = os.path.join(testpath, testLabels)
        self.trainSize  = trainSize
        self.testSize   = testSize
        self.imageWidth = imageWidth
        self.imageDepth = imageDepth  

    def pickleTraindat(self):
        """ Load pickled Training data from file and put all in memory """
        # Load the data...
        self.trainX = pickle.load(open(self.trainXpath, 'rb'))
        self.trainY = extract_labels(self.trainYpath, self.trainSize).reshape(self.trainSize, 1)
        # Calculate the dimensions of the training frames...
        self.imageHeight =  self.trainX.shape[1] // self.imageWidth   
        print(f"[INFO] Train data img_depth = {self.imageDepth}, img_height = {self.imageHeight}, img_width = {self.imageWidth}.")
        return self.trainX, self.trainY, self.imageHeight

    def gzTraindat(self):
        """ Load Training data that was compressed in a gz file and put all in memory """
        # Load the data...
        self.trainX = extract_data(self.trainXpath, self.trainSize, self.imageWidth)
        self.trainY = extract_labels(self.trainYpath, self.trainSize).reshape(self.trainSize, 1)
        # Calculate the dimensions of the training & testing frames...
        self.imageHeight =  self.trainX.shape[1] // self.imageWidth
        print(f"[INFO] Train data img_depth = {self.imageDepth}, img_height = {self.imageHeight}, img_width = {self.imageWidth}.")
        return self.trainX, self.trainY, self.imageHeight

    def pickleTestdat(self):
        """ Load pickled Testing data from file and put all in memory """
        # Load data...
        self.testX = pickle.load(open(self.testXpath, 'rb'))
        self.testY = extract_labels(self.testYpath, self.testSize).reshape(self.testSize,1)
        # Calculate the dimensions of the testing frames...
        self.imageHeight = self.testX.shape[1] // self.imageWidth
        print(f"[INFO] Test data img_depth = {self.imageDepth}, img_height = {self.imageHeight}, img_width = {self.imageWidth}.")
        return self.testX, self.testY, self.imageHeight

    def gzTestdat(self):
        """ Load Testing data that was compressed in a gz file and put all in memory """
        # Load data...
        self.testX = extract_data(self.testXpath, self.testSize, self.imageWidth)
        self.testY = extract_labels(self.testYpath, self.testSize).reshape(self.testSize,1)
        # Calculate the dimensions of the testing frames...
        self.imageHeight = self.testX.shape[1] // self.imageWidth
        print(f"[INFO] Test data img_depth = {self.imageDepth}, img_height = {self.imageHeight}, img_width = {self.imageWidth}.")
        return self.testX, self.testY, self.imageHeight

    def getFashionMNIST(self):
        """ Get the Fashion MNIST data from the TensorFlow Keras dataset and put all in memory """
        import tensorflow as tf        # This required that the TesorFlow libraries are installed...
        import numpy as np

        # Load the fashion MNIST pre-shuffled train and test data
        (self.trainX, self.trainY), (self.testX, self.testY) = tf.keras.datasets.fashion_mnist.load_data()
        trainNum, trainHeight, trainWidth = self.trainX.shape
        testNum, testHeight, testWidth = self.testX.shape
        print(f"[INFO] Keras fashion MNIST shape of training data: {(trainNum, trainHeight, trainWidth)}, shape of testing data: {(testNum, testHeight, testWidth)}")
        # Reshape the data for further processing. Required is here (num_of_frames, height x width 1-dimensional)...
        self.trainX = np.reshape(self.trainX, (trainNum, trainHeight*trainWidth))
        self.trainY = np.reshape(self.trainY, (trainNum, 1))
        self.testX = np.reshape(self.testX, (testNum, testHeight*testWidth))
        self.testY = np.reshape(self.testY, (testNum, 1))
        print(f"[INFO] The fashion MNIST data has after the reshape procedure a for train data shape {self.trainX.shape} and for test data shape {self.testX.shape}.")

        return self.trainX, self.trainY, trainNum, trainHeight, self.testX, self.testY, testNum, testHeight

    def readImages(self):
        """ 
            Load custom Training and Testing data from a file system and put all in memory.
            The different classes of this collection are saved in separate folders and the folder names are the class-names.
            The frames of this collection are then resized all to the same quadratic format.
            Then the reshaped frames are again reshaped to 1-D arrays:
            -- In later steps this 1-D format helps to add to the 1 dimension at the end the number corresponding to the class...
        """
        print("[INFO] Function not completed...")
        pass
