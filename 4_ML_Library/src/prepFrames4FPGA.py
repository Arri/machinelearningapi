#################################################################
# Description: 
# Resize MNIST images from (28,28,1) to (32,32,1)




import os
import numpy as np
import gzip
import pickle
import matplotlib.pyplot as plt
import argparse
import csv
from utils import *


INPUTFRAMES = "t10k-images-idx3-ubyte.gz"     #"train-images-idx3-ubyte.gz"        #"t10k-images-idx3-ubyte.gz"     #"train-images-idx3-ubyte.gz"
INPUTLABELS = "t10k-labels-idx1-ubyte.gz" #"train-labels-idx1-ubyte.gz"        #"t10k-labels-idx1-ubyte.gz"

TESTFOLDER  = "./TestFrames"
# CONVERTTO   = 'TwoComplement'
bintest     = "MNIST"          # Front part of the test files
DATASIZE    = 10000       #50000       #10000       #50000
ORIGDIM     = 28
TARGETDIM   = 32

RESIZED     = "t10k-images-idx3-ubyte_reshaped" + str(TARGETDIM) + "X" + str(TARGETDIM) + ".pkl"    #"train-images-idx3-ubyte_reshaped32x32.pkl"   #"t10k-images-idx3-ubyte_reshaped32x32.pkl"        #"train-images-idx3-ubyte_reshaped32x32.pkl"
COLLECT     = 9     # Number of images to collect. Should be square of am integer...

ap = argparse.ArgumentParser(description="Prepare train and test images for the accelerator.")
ap.add_argument("--frames", required=False, default=INPUTFRAMES,
                    help='Pickled file containing input frames. Default = {}'.format(INPUTFRAMES))
ap.add_argument("--labels", required=False, default=INPUTLABELS,
                    help='Pickled file containing input labels. Default = {}'.format(INPUTLABELS))
ap.add_argument('--saveTo', required=False, default=RESIZED,
                    help='File to save resized frames to. Default = {}'.format(RESIZED))
# ap.add_argument('--convert', type=str, default=CONVERTTO,
#                     help='Convert frame pixels to "TwoComplement" or "Binary". Default = {}'.format(CONVERTTO))

args = ap.parse_args()


class prepImgs:
    def __init__( self, frames=args.frames, labels=args.labels, saveTo=args.saveTo, 
                    data_size=DATASIZE, orig_dim=ORIGDIM, target_dim=TARGETDIM, test_folder=TESTFOLDER, collect=COLLECT ):
        self.inFrames    = frames
        self.inLabels    = labels
        self.saveTo      = saveTo
        # self.convert     = convert  
        self.data_size   = data_size 
        self.orig_dim    = orig_dim     
        self.target_dim  = target_dim
        self.test_folder = test_folder
        self.X           = np.zeros((self.data_size, 1, self.target_dim, self.target_dim))
        self.collection  = []    # Collection of randomly chosen indices of frames...
        self.collect     = collect    # Number of images to collect -- should be square of an integer...

    def readOrig(self):
        self.Xorig = extract_data(self.inFrames, self.data_size, self.orig_dim)                 # Reading frames...
        self.Yorig = extract_labels(self.inLabels, self.data_size).reshape(self.data_size, 1)   # Reading labels...
        print(f"[INFO] The original shape of X is ... = {self.Xorig.shape}")
        

    def resizeImgs(self):
        ## Resize the 28x28 input matrices to 32x32 by adding 2 rows/columns on top/bottom and left/right...:
        # Reshape the pickeled data to the original 28x28...
        self.Xorig = self.Xorig.reshape(self.data_size, 1, self.orig_dim, self.orig_dim)
        # Resize the image UP using numpy's pad function...
        del_dim = int((self.target_dim - self.orig_dim) / 2)
        for i in range(self.data_size):
            self.X[i,0] = np.pad(self.Xorig[i,0], (del_dim, del_dim), 'edge')     # Reshaping to target dim...

        print(f"[INFO] The final shape is ... : {self.X.shape}")
        print(f"[INFO] Trying to save this array as a new pickle file in {self.saveTo}  ...")
        with open(self.saveTo, 'wb') as file:
            pickle.dump(self.X, file)
        # Free memory
        del self.Xorig
        del self.Yorig


    def converTO8Bit(self):
        # Load the pickle file...
        self.X_dash =  pickle.load(open(self.saveTo, 'rb'))
        self.y_dash =  extract_labels(self.inLabels, self.data_size).reshape(self.data_size, 1)
        print(f"[INFO] Loaded the pickle file {self.saveTo}. It contains matrix of size {self.X_dash.shape}")
        # Check if result folder exists and create it if not...
        if not os.path.exists(self.test_folder):
            print(f"[INFO] Creating the folder {self.test_folder}...")
            os.makedirs(self.test_folder)        

        # Pick random frames and save them in 8-bit binary form into a text file...   
        for i in range(self.collect):
            # Pick a random file...
            while True:
                pick_one = np.random.choice(self.data_size, 1)
                if not pick_one in self.collection:
                    self.collection.extend(pick_one)
                    # print("is not in collection")
                    break
            print(f"---- {pick_one}  ===== {self.collection}")
            
            # Save this frame after the previous made sure we don't have that frame already...
            ## Save the frame with original values...
            fileName = os.path.join(self.test_folder, "Original_" + bintest + str(i) + "_yLABEL_" + str(self.y_dash[pick_one][0][0]) + "_.txt")
            with open(fileName, 'w+', newline='') as f:
                f_writer = csv.writer(f, delimiter=",")
                for row in self.X_dash[pick_one].reshape(self.target_dim, self.target_dim):
                    f_writer.writerow(row)

            fileName = os.path.join(self.test_folder, bintest + "_8bit" + str(i) + "_yLABEL_" + str(self.y_dash[pick_one][0][0]) + "_.txt")
            # print("=====================================================")
            with open(fileName, 'w+', newline='') as f:
                f_writer = csv.writer(f, delimiter=",")            
                for row in self.X_dash[pick_one].reshape(self.target_dim, self.target_dim):
                    onerow = []
                    for col in row:
                        onerow.append(f"{int(col):08b}")
                    f_writer.writerow(onerow)

    def display(self):
        # Show a random example set from the trainings set...  
        rowscols = int(np.sqrt(self.collect))
        fig, ax = plt.subplots(nrows=rowscols, ncols=rowscols, figsize=[6,8])
        for i, axi in enumerate(ax.flat):
            pick_one = self.collection[i]
            print(f"pick_one = {pick_one}, self.y_dash = {self.y_dash[pick_one]}")
            sample = self.X_dash[pick_one].reshape(32, 32)
            axi.imshow(sample, cmap='gray', alpha=0.25)
            rowid = i
            colid = i % rowscols
            axi.tick_params(axis='both', which='major', labelsize=9)
            axi.tick_params(axis='both', which='minor', labelsize=8)
            axi.set_title(f"Label = {self.y_dash[pick_one][0]}")
        plt.show()


def run():
    prep = prepImgs()
    prep.readOrig()
    prep.resizeImgs()
    prep.converTO8Bit()
    prep.display()


if __name__=="__main__":
    run()