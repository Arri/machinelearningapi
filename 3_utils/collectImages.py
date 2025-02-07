############################################################################################
# This function inherits the class library in camera.py to collect training and testing
# images using an attached web-cam.
# This function performs altready during the image capture the train/test split.
#  
# Author: Arasch U Lagies
# First Version: 05/26/2020
# Last Update: 05/26/2020
#
# Default call: python collecImages.py
#
#############################################################################################
import argparse
from camera import stream
import cv2
import os
import time
import re
import random
import shutil

CWIDTH          = 200   #640        # Set the with for the captured images
CHEIGHT         = 200   #480        # Set the height for the captured images
CFPS            = 30                # Set the framerate of the web-cam
FOLDER          = './pictures/'     # Name of the folder to dump images into
TRAINFOLDER     = 'training'           # Name of folder used to collect in the training images
TESTFOLDER      = 'testing'            # Name of folder used to collect in the testing images
VALIDATION      = 'validation'
TRAINSPLIT      = 0.8               # Percentage of all images captured to be put into the train folder
                                    # (1 - TRAINSPLIT) is what is put into the test folder...
VALIDATIONSPLIT = 0.2               # Validation set split. Percentage taken from the training set...

PICTURES = 100                      # Number of images to collect per run - these pictures belong all to one class

class capture(stream):
    def __init__(self, classe = None, width = CWIDTH, height = CHEIGHT, cfps = CFPS, folder = FOLDER, 
                        trainsplit=TRAINSPLIT, valsplit=VALIDATIONSPLIT):
        # Collect chosen camera settings
        self.width      = width
        self.height     = height
        self.cfps       = cfps
        self.folder     = folder
        self.trainsplit = trainsplit
        self.valsplit   = valsplit
        self.classe     = classe

        super().__init__()

        # Check if directory exists, if not create it...
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        # Check if target folder is empty...
        self.counter = 0        # Start file counter at 0...
        if not len(os.listdir(self.folder)) == 0:
            # Find the file with the highest count...
            # for filename in os.listdir(self.folder):
            for _, _, filenames in os.walk(self.folder):
                for filename in [f for f in filenames if f.endswith('.jpg')]:
                    num=re.findall(r'\d+', filename)
                    number = int(num[0])
                    if number > self.counter:
                        self.counter = number
            self.counter += 1

    def savePic(self):
        """ 
            This function used OpenCV library to capture images.
            The image capture is done in the inherited class stream() from the file camera.py.
        """ 
        self.get()
        cv2.imwrite(os.path.join(self.folder, 'capture_' + str(self.classe) + '_' + str(self.counter) + '.jpg'), self.frame)
        self.counter += 1

    def release(self):
        """ Release the camera and close all windows """
        self.shut()

    def train_test_split(self):
        """ 
            This function splits the collected images into training and testing sets and puts the images into folders named with the 
            classification name
        """
        self.trainpath = os.path.join(FOLDER, TRAINFOLDER)
        self.testpath  = os.path.join(FOLDER, TESTFOLDER)
        self.valpath   = os.payh.join(FOLDER, VALIDATION)
        # Check if train and test folders already exist in FOLDER. If not create them new...
        if not os.path.exists(self.trainpath):
            os.mkdir(self.trainpath)
        if not os.path.exists(self.testpath):
            os.mkdir(self.testpath)
        if not os.path.exists(self.valpath):
            os.mkdir(self.valpath)
        # Check if the class exists as a folder in training in testing directories...
        self.train_classdir = os.path.join(self.trainpath, str(self.classe))
        self.test_classdir  = os.path.join(self.testpath, str(self.classe))
        self.val_classdir  = os.path.join(self.valpath, str(self.classe))
        if not os.path.exists(self.train_classdir):
            os.mkdir(self.train_classdir)
        if not os.path.exists(self.test_classdir): 
            os.mkdir(self.test_classdir)
        if not os.path.exists(self.val_classdir): 
            os.mkdir(self.val_classdir)
        # Move images to test and train folsers...
        for filename in os.listdir(self.folder):
            if filename.endswith('.jpg'):
                # Generate a random number that determines if image goes into training or into testing folder ...
                rand = random.uniform(0, 1)
                if rand <= self.trainsplit:
                    randval = random.uniform(0, 1)
                    if randval < self.valsplit:
                        shutil.move(os.join(self.folder, filename), self.val_classdir)
                    else:
                        shutil.move(os.path.join(self.folder, filename), self.train_classdir)
                else:
                    shutil.move(os.path.join(self.folder, filename), self.test_classdir)



if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('classe', metavar = 'Class Name', 
                help='Please provide name of file to save parameters in.')
    ap.add_argument("-w", "--width", required=False,
       default=CWIDTH, type=int, help="With of captured image...")
    ap.add_argument("-e", "--height", required=False,
       default=CHEIGHT, type=int, help="Height of captured image...")
    ap.add_argument("-f", "--folder", required=False,
       default=FOLDER, type=str, help="Folder where pictures are to be saved...")
    ap.add_argument("-p", "--pics", required=False,
       default=PICTURES, type=int, help="Amount of pictures to take...")
    ap.add_argument("-s", "--fps", required=False,
       default=CFPS, type=float, help="Camera frames per second...")
    ap.add_argument("-t", "--trainsplit", required=False,
       default=TRAINSPLIT, type=float, help="Train split from all images as a number between 0 and 1...")

    args       = vars(ap.parse_args())
    classe     = args["classe"]
    width      = args["width"]
    height     = args["height"]
    folder     = args["folder"]
    pics       = args["pics"]
    cfps       = args["fps"]
    trainsplit = args["trainsplit"]

    cap = capture(classe, width, height, cfps, folder, trainsplit)

    # Capture images... 
    for i in range(pics):
        print('[INFO] Picture {} from {} pictures'.format(i,pics))
        cap.savePic()
        time.sleep(1)
    
    # and release camera...
    cap.release()

    # ditribute the images into training and testing folders...
    cap.train_test_split()

