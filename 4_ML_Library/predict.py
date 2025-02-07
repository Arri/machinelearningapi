##########################################################################################################
# Description: Inference engine with the option of inserting frames and skipping layers 
#               that where trained and processed with other methods (e.g. a CNN accelerator)
#               Note: only layers that follow each other can be replaced by externally generated frames
#                   e.g. to replace two Conv2D layers, the two Conv2D must be directly subsequent layers
#
# Author: Arasch Lagies
#
# Date: 2/19/2020
# Latest Update: 2/19/2020
#
#    - e.g. to repace 'Conv2D' layers during the inference use the commandline argument --replace Conv2D
#       e.g. Call: python inference.py --replace Conv2D
# Note: not using the --replace flag uses the trained parameters and performes the inference on the computer 
#
# If running inference on PC Call:  > python predict.py yourTrainedModle.pkl --replace None --frame 
# If running inference with replacing Conv2D layer using accelerator result frames Call:
#                                   > python predict.py yourTrainedModel.pkl --accframe acceleratorOutputFrames.txt   
############################################################################################################
# Add src path...
import sys
sys.path.append('./src')
import os
import numpy as np
import csv
import argparse

from sequential import *
from inference import *
from utils import *
import logging
# from API.processFPGA import *

REPLACE           = "none"   #"scaleTest"   #"Conv2D"                     # If this is set to 'Conv2D' then the input will be taken and processed as accelerator genrated frames...
                                                                # If this is set to 'scaleTest' then the scaled truncated to integer weights and biases will be used for the Conv2D process...

MODELPATH         = './modelSave/small_3xConv2D_3x3_.pkl'    #"/home/user1/Documents/CNN/cnn/new5_t10k_noInNorm_1ConvWrap16_3x3_NormBy255_Fl_DenSofM_2Epochs.pkl"    # degfault pikl file
ACCFRAMES         = "AcceleratorOutput.txt"   #"Original_MNIST0_yLABEL_7_.txt"    #"AcceleratorOutput.txt"
FRAME             = 'Original_MNIST0_yLABEL_7_.txt'
NORMCHOICE        = 'None'   #'bySTD'
TESTFRAME         = './TestFrames'
ACCELERATORFRAMES = './fpga_accelerator'
REPLACEBITS       = 8       # Bitdepth for the accelerator binary output data...

ap = argparse.ArgumentParser(description="Prediction with and without accelerator input.")
ap.add_argument("inModel", metavar='Save Path', default=MODELPATH,
                help="Path and name of the trained model. Default = {}".format(MODELPATH))
ap.add_argument("-r", "--replace", type=str, default=REPLACE,
                help="Layer name to be replaced. Available is Conv2D. Default = None, which means that no layer is inserted.")
# ap.add_argument("-a", "--accframe", type=str, default=ACCFRAMES,
#                 help="Raw input frame. This frame is taken for processing if" + \
#                 "If '--replace' flag with argument given. Assuming for accelerator frames 8-bit binary (7bits data, 1bit sign), Big Endian, sign bit most left..")
ap.add_argument("-f", "--frame", type=str, default=FRAME,
                help="Raw input frame. Default is {}".format(FRAME))
ap.add_argument("--norm", type=str, default=NORMCHOICE,
                help="Choice for which normalization to use on the input image. Default = {}".format(NORMCHOICE))
ap.add_argument("--bits", type=int, default=REPLACEBITS,
                help="Bit depth of the accelerator genereted output frames")

args = ap.parse_args()



def main(inModel=args.inModel, replace=args.replace, testframe=args.frame, bits=args.bits, norm=args.norm):   #accName=args.accframe, testframe=args.frame, bits=args.bits, norm=args.norm):
    """
        0. Read in the original model.
        1. Check if a layer of the original trained model needs to be replaced.
        2. If no replacement, then perform the standard forward propagatiom using the full set of trained parameters
        3. If replacement and the name of a replacement layer is given:
           - take the file with the binary replacement layer frames (produced by the accelerator), 
             convert the replacement layer binary data to integer numbers,
             scale the data beck to the original range,
           - in the inference forward pass, when 'Layer'==replacement layer name, then frame=list of replacement layer frames
    """
    scale = 0    # Scaling factor...
    # Read the original model from given pickle file...
    model = Sequential()
    params = model.Load_Model(save_path = inModel)

    # Set up an instance of the  inference engine
    infer = inference(params=params)
    #------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if (replace.lower() == 'conv2d'):                   # A Conv2D replacement layer is provided (onlyone supported for now)
        print(f"[INFO] Running inference using the Accelerator generated frames and skipping the Conv2D layer.")
        print(f"[INFO] Reading the accelerator output frames from the file {testframe}.")
        ## Read the accelerator processed frames...
        # 1. The accelerator provides processed frames as positive 8bit binary values
        # Reading the accelerator output:
        accframe = []       # List of input frames...
        framePath = os.path.join(ACCELERATORFRAMES, testframe)
        print(f"[INFO] Convertindg back from binary to integer dec.")
        with open(framePath, newline='') as csvf:
            reader = csv.reader(csvf, delimiter=',')
            for line in reader:
                if not line==[' ']:
                    accframe.extend(np.array([int(l,2) for l in line]))     # Converting the frame pixels from binary back to dec integers...

        # 2. Get the scaling factor...
        for layer in params:
            if layer["Layer"][:6] == 'Conv2D':
                # Determine the expected output dimensions of the Conv2D layer...
                depth, rows, cols = layer["Forward Frame"].shape
                frame = np.array(accframe)
                frame = np.reshape(frame, (depth, rows, cols))                             # Reshape frames to the original dimensions
                if ("Scale Factor" in layer):
                    scale = frame.max()/255.    #layer['Scale Factor']/255.
                    print(f"[INFO] Found a Conv2D layer and a Scale Factor of {scale}.")
                    break
                else:
                    print(f"[ERROR] Found a Conv2D layer in the trained model {inModel}, but no scaling factor...")
                    exit(0)
        else:
            print(f"[ERROR] Cound not find any Conv2D layer in the provided model {inModel}")
            exit(0)

        print(f"[INFO] The shape of the provided frame(s) matrix is {frame.shape}")
        print(f"[INFO] The max value is {np.max(frame)} and min value is {np.min(frame)}")
        # Display the input frames...
        display(frame, nrows=1, ncols=1, data_size=1, title_ = "Input Frame")
        print()
        print(f"[INFO] The trained Weights and Biases of the Conv2D layer were scaled by {scale} to normalize them to {layer['Bit Depth']} binary values.")
        print(f"[INFO] Converting back to original vlaues (neglegting the precision loss due to integer conversion).")
        frame = np.true_divide(frame, scale)                # Scaling the pixels of the acclerator frames back...

        print("[INFO] Accelerator output frames scaled back by Weight & Biases scaling factor.")
        print(f"[INFO] Back-scaled frames max is {np.max(frame)} and the min value is {np.min(frame)}.")

        # Predict...:
        pred, prob = infer.forward(frame, replace='Conv2D')     

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------
    elif (replace == "scaleTest"):
        ## Test procedure:
        # The scaled and truncated to integer values of the weights and biases of the Conv2D layer are being used for the convolution...
        # So with w' = int(A * w) and b' = int(A * b) one gets z' = w' * f + b' ~= A * (w * f + b) = A * z    with a loss in precision due to the int conversion...
        # 1. Read the input TEST frame, wich is in integer 8-bit form:
        conv = []
        framePath = os.path.join(TESTFRAME, testframe)
        print(f"[INFO] Running inference test using a test frame and the scaled & integer converted Weights and Biases.")
        print(f"[INFO] Reading frame {framePath}.")
        with open(framePath, newline='') as csvf:
            reader = csv.reader(csvf, delimiter=',')
            for line in reader:
                conv.append([int(float(i)) for i in line])          # Reading the input frame as integer 8-bit pixel values...
        # Convert the list of lists into a numpy ndarray...
        convFrame = np.array([np.array(xi) for xi in conv])
        print(f"[INFO] Received frame of shape {convFrame.shape}")

        # Plot the input frame before inference...
        display(convFrame, nrows=1, ncols=1, data_size=1, title_ = "Input Frame")

        frame = np.expand_dims(convFrame, axis=0)
        print(f"[INFO] Converted the shape of the input frame to {frame.shape}.")

        # Predict...:
        pred, prob = infer.forward(frame, replace='scaleTest')

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------      
    else:                               # No replacement layer is provided --> use forward prop with all layers...
        print(f"[INFO] The supplied model requires input frames of the shape {params[0].get('Frame Shape')}")
        conv = []
        framePath = os.path.join(TESTFRAME, testframe)
        print(f"[INFO] Reading frame {framePath}.")
        with open(framePath, newline='') as csvf:
            reader = csv.reader(csvf, delimiter=',')
            for line in reader:
                conv.append([float(l) for l in line])
        # Convert the list of lists into numpy ndarray...
        convFrame = np.array([np.array(xi) for xi in conv])
        
        # Plot the input frame before inference...
        display(convFrame, nrows=1, ncols=1, data_size=1, title_ = "Input Frame")
        
        print(f"[INFO] Received input frame of shape {convFrame.shape}.")
        frame = np.expand_dims(convFrame, axis=0)
        print(f"[INFO] Converted the input frame to {frame.shape}")
        # Normalize the data...
        print(f"[INFO] Range before scaling: max = {frame.max()}, min = {frame.min()}.")
        if norm == 'byStd':
            frame -= int(np.mean(frame))
            frame /= int(np.std(frame))
        elif norm == ' by255':
            frame = np.divide(frame, 255.0)    
        print(f"[INFO] Range after scaling: max = {frame.max()}, min = {frame.min()}.")

        # Predict...:
        pred, prob = infer.forward(frame)


    print("----------------------------------------------------")
    print(f"Prediction = {pred}, Probability = {(prob*100):.4f}%")
    print("----------------------------------------------------")

    return 1



if __name__=="__main__":
    main()