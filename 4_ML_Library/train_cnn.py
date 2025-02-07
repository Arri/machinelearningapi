#############################################################################################
# Description: Sequential library similar to Tensorflow 2.0 Keras Sequential-API for model design, 
# training, evaluation and deployment.
# Models are accessed and saved in pickle (binary) format.
#
# Author Arasch Lagies
# 
# Date: Dec 2, 2019
# Last Update: 08/28/2020
#
# Call: python train_cnn.py -s <save_filename.pkl>
#
# This call saves the trained model into the pickle file "save_filename.pkl". The model and 
# its trained weights and biases are also saved in bianry form and in a csv file where they are
# human readable.
#############################################################################################

import sys
# Add src path...
sys.path.append('./src')
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sequential import *
from loadData import *

SAVEMODEL = "test_result.pkl"
SAVEMODELPATH = "./modelSave"
TRAINDATASIZE = 50000
TESTDATASIZE = 10000
IMAWIDTH   = 28   #32    #28    #32
IMAGEDEPTH = 1
TRAINDATAPATH = "../2_data/"
TESTDATAPATH  = "../2_data/"
TRAINX = "train-images-idx3-ubyte.gz"  #"fashion"  #"fashionMNIST_train-images-idx3-ubyte.gz"   #"train-images-idx3-ubyte.gz"   #"train-images-idx3-ubyte_reshaped32x32.pkl"        #"train-images-idx3-ubyte.gz"
TRAINY = "train-labels-idx1-ubyte.gz"  #"fashion"  #"fashionMNIST_train-labels-idx1-ubyte.gz"   #"train-labels-idx1-ubyte.gz"
TESTX  = "t10k-images-idx3-ubyte.gz"   #"fashion"  #"fashionMNIST_t10k-images-idx3-ubyte.gz"    # "t10k-images-idx3-ubyte.gz"   #"t10k-images-idx3-ubyte_reshaped32x32.pkl"         #"t10k-images-idx3-ubyte.gz"
TESTY  = "t10k-labels-idx1-ubyte.gz"   #"fashion"  #"fashionMNIST_t10k-labels-idx1-ubyte.gz"    #"t10k-labels-idx1-ubyte.gz"
BATCHSIZE  = 32
NUMEPOCHS  = 4
NORMCHOICE = 'bySTD'  #"None"  #'by255'    #'bySTD'
NORMLAYER  = "None"   #'by255' #'byStd'    #'by255'
TRAIN      = True
PROCESSPARALLEL = False        # Flag to enable multiprocessing for the training....


ap = argparse.ArgumentParser(description='Train a convolutional neural network.')

ap.add_argument("-s", "--save_model", type=str, default=SAVEMODEL,        #metavar = 'Save Path', 
                help='name of file to save parameters in. Default = {}'.format(SAVEMODEL))
ap.add_argument("-dx", "--trainx", type=str, default=TRAINX,
                help="Training data file. Default = {}".format(TRAINX))
ap.add_argument("-dy", "--trainy", type=str, default=TRAINY,
                help="Training data labels. Default = {}".format(TRAINY))
ap.add_argument("--testx", type=str, default=TESTX,
                help="Testing data file. Default = {}".format(TESTX))
ap.add_argument("--testy", type=str, default=TESTY,
                help="Testing label file. Default = {}".format(TESTY))
ap.add_argument("--size", type=int, default=TRAINDATASIZE,
                help="TRAINDATASIZE or amount of images for training that are stored in the pickel file. Default = {}".format(TRAINDATASIZE))
ap.add_argument("--testsize", type=int, default=TESTDATASIZE,
                help='TESTDATASIZE or amount of images for testing that are stored in a pickle file. Default = {}'.format(TESTDATASIZE))
ap.add_argument("-d", "--imgWidth", type=int, default=IMAWIDTH,
                help="Image width -- the hight is alculated using the given width and channel depth. Only one int value. Default = {}".format(IMAWIDTH))
ap.add_argument("--imgDepth", type=int, default=IMAGEDEPTH,
                help="Image depth. Default = {}".format(IMAGEDEPTH))
ap.add_argument("--batch", type=int, default=BATCHSIZE,
                help="Batch size. Default = {}".format(BATCHSIZE))
ap.add_argument("--epochs", type=int, default=NUMEPOCHS,
                help="Number of epochs. Default = {}".format(NUMEPOCHS))
ap.add_argument("--norm", type=str, default=NORMCHOICE,
                help="Choice for which normalization to use on the input image. Default = {}".format(NORMCHOICE))
ap.add_argument("--normlayer", type=str, default=NORMLAYER,
                help="Choice for normalization performed in the norm layer. Available are 'None', 'by255' and 'byStd'. Default = {}".format(NORMLAYER))
ap.add_argument("--train", type=str, default=TRAIN,
                help="Flag to run a full training (yes) or to only run an inference. Default is '{}'".format(TRAIN))
args = ap.parse_args()


def main(trainData = args.trainx, trainLabels=args.trainy, save_model=args.save_model, testData=args.testx, testLabels=args.testy, 
                        data_size=args.size, test_size=args.testsize, img_width=args.imgWidth, img_depth=args.imgDepth, innorm=args.norm, normlayer=args.normlayer,
                        num_epochs=args.epochs, batch=args.batch, parallelProcess=PROCESSPARALLEL):
    """
    main() performs following steps:
    1. Collect training data...
    2. Build a model
    3. Compile the model
    3. Train the model
    4. Save the model
    5. Re-load the model
    6. Load testing data
    7. Evaluate model
    """
    ## Collect the training data...
    data = loadData(trainpath=TRAINDATAPATH, testpath=TESTDATAPATH, trainData=trainData, trainLabels=trainLabels, testData=testData, testLabels=testLabels,
                        trainSize=data_size, testSize=test_size, imageWidth=img_width, imageDepth=img_depth)
    if trainData.endswith(".pkl"):
        trainX, trainY, img_height = data.pickleTraindat()
    elif trainData.endswith(".gz"):
        trainX, trainY, img_height = data.gzTraindat()
    elif trainData.lower() == "fashion":
        trainX, trainY, data_size, img_height, *_ = data.getFashionMNIST()
    else:
        raise ValueError(f"[ERROR] The given test X input file has a not supported format....")

    print(f"[INFO] Shape of the input training data is: {trainX.shape} and the shape of the input labels is: {trainY.shape}.")

    # Normalize the data...
    trainX = normalize(trainX, innorm)
 
    # Show a random example set from the trainings set...  
    nrows, ncols = 3, 3
    display(trainX, nrows, ncols, data_size, img_height, img_width, trainY, title_ = "Sample of the Input Data")
    
    ## Build the Model Sequentially ...
    # Note: The first layer needs to specify the dimensions of the input frame...
    # Model definition
    model = Sequential(toTrain=args.train, inputNorm=innorm, loggingPath=SAVEMODELPATH, model_name=save_model) 
    ##=============================================================================================================================================================
    ##=================================== This is the part to comment out to run only the evaluation and saving of model as csv and as numpy binary================
    if args.train:
        print("===========================================================")
        print("============= Starting the Training Procedure =============")
        print("===========================================================")
        model.Add(model.layers.Conv2D(input_shape=(img_height,img_width,img_depth), kernel_size=(3,3), filters=8, stride=(1,1), padding='none', activation='relu'))
        #model.Add(model.layers.Normalize(norm=normlayer))
        model.Add(model.layers.Conv2D(kernel_size=(3,3), filters=8, stride=(1,1), padding='none', activation='relu'))  
        model.Add(model.layers.Conv2D(kernel_size=(3,3), filters=8, stride=(1,1), padding='none', activation='relu')) 
        model.Add(model.layers.Conv2D(kernel_size=(3,3), filters=8, stride=(1,1), padding='none', activation='relu'))
        model.Add(model.layers.MaxPool2D(pool_size=(2,2), strides=(1,1), padding='none'))
        model.Add(model.layers.Flatten())
        # model.Add(model.layers.Dense(units_out=128, activation='relu'))
        # model.Add(model.layers.Dense(units_out=3, activation='relu'))
        model.Add(model.layers.Dense(units_out=10, activation='softmax'))

        ## Compile the model...
        model.Compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'],
                        lr = None, beta1 = None, beta2 = None)

        ## Train the model...
        model.Fit(x_train = trainX, y_train = trainY, img_dim = img_height, img_depth = img_depth, num_epochs = num_epochs, batch = batch, num_classes = 10, parallel=parallelProcess)

        ## Save the model (Pickle file)
        model.Save_Model(os.path.join(SAVEMODELPATH, save_model))
    ##=============================================================================================================================================================
    
    ## Load the model
    model.Load_Model(os.path.join(SAVEMODELPATH, save_model))

    ## Save the model in binary form...
    model.Save2Binary(os.path.join(SAVEMODELPATH, save_model))

    ## Save the model in plain text form...
    model.Save2Text(os.path.join(SAVEMODELPATH, save_model))

    ## Evaluate the model using testing data
    print("===========================================================")
    print("================== Evaluating the Model ===================")
    print("===========================================================")
    # a. Get test data..
    if testData.endswith('.gz'):
        testX, testY, img_height = data.gzTestdat()
    elif testData.endswith('.pkl'):
        testX, testY, img_height = data.pickleTestdat()
    elif testData.lower() == "fashion":
        *_, testX, testY, test_size, img_height = data.getFashionMNIST()
    else:
        print(f"[ERROR] The given test X input file has a not supported format....")
        exit(0)

    # b. Normalize the test data...
    testX = normalize(testX, innorm)

    # c. Evaluete the test data
    model.Evaluate(testX, testY, img_depth = img_depth, img_dim = img_height)

if __name__=="__main__":
    main()

