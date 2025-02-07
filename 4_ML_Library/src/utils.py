#####################################################################################################
# Description: Utility methods for a Convolutional Neural Network
#
# Author: Arasch U Lagies
# Version: V.2.
# First Version: June 12th, 2018 (based on https://github.com/Alescontrela/Numpy-CNN)
# Last Update: 5/26/2020
#
#####################################################################################################
import numpy as np
import gzip
import matplotlib.pyplot as plt

#####################################################
################## Utility Methods ##################
#####################################################
        
def extract_data(filename, num_images, IMAGE_WIDTH):
    '''
    Extract images by reading the file bytestream. Reshape the read values into a 3D matrix of dimensions [m, h, w], where m 
    is the number of training examples.
    '''
    print('[INFO] Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, IMAGE_WIDTH*IMAGE_WIDTH)
        return data

def extract_labels(filename, num_images):
    '''
    Extract label into vector of integer values of dimensions [m, 1], where m is the number of images.
    '''
    print('[INFO] Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def display(X, nrows, ncols, data_size, img_height=32, img_width=32, y_dash=None, title_=None):
    # Show a random example set from the trainings set...  
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=[6,8])
    for i, axi in enumerate(ax.flat):
        pick_one = np.random.choice(data_size, 1)
        sample = X[pick_one].reshape(img_height, img_width)
        axi.imshow(sample, cmap='gray', alpha=0.25)
        axi.tick_params(axis='both', which='major', labelsize=9)
        axi.tick_params(axis='both', which='minor', labelsize=8)
        if y_dash is not None:
            axi.set_title(f"Label = {y_dash[pick_one][0]}")
    if title_ is not None:
        fig.suptitle(title_, fontsize=16)    
    plt.show()

def twosCom_binDec(bin, digit):
    while len(bin)<digit :
        bin = '0'+bin
    if bin[0] == '0':
        return int(bin, 2)
    else:
        return -1 * (int(''.join('1' if x == '0' else '0' for x in bin), 2) + 1)

def normalize(X, innorm):
    print(f"[INFO] Range before {innorm} scaling: max = {X.max()}, min = {X.min()}.")
    if innorm.lower() == 'bystd':
        X_ = X - np.mean(X)
        X_ /= np.std(X)
    elif innorm.lower() == 'by255':
        X_ = np.divide(X, 255.0)    
    print(f"[INFO] Range after {innorm} scaling: max = {X_.max()}, min = {X_.min()}.")
    return X_