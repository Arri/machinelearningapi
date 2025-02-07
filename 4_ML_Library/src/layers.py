#################################################################
# Layers collection 
# Author: Arasch Lagies
#
# First version: 12/26/2019
# Last Update: 08/28/2020
#
# Forwrd inference layer processing functions
##################################################################
import os
import numpy as np
from activation import *
import time

##-----------------------------------------------------------------------------------------------
def convolution(frame, kernel, bias, stride, padding=None):
    """
    This functiom convolves 'kernel' over 'frame' using the stride 'stride'.
    Then apply the chosen activation function on the resulting matrix
    """
    (n_f, n_c_f, fx, fy) = kernel.shape                                # kernel dimensions assuming square shaped kernel

    # Add padding to the frame...
    npad = ( (0,0), (int(fx/2), int(fx/2)), (int(fx/2), int(fy/2)) )   # Only rows and columns of frames are padded...
    if (padding == None or padding.lower() == 'none'):
        pass
    elif (padding.lower() == 'zeros'):
        frame = np.pad(frame, pad_width=npad, mode='constant', constant_values=0)
    elif (padding.lower() == 'edge'):
        frame = np.pad(frame, pad_width=npad, mode='edge')
    elif (padding.lower() == 'wrap'):
        frame = np.pad(frame, pad_width=npad, mode='wrap')
    else:
        raise ValueError("[ERROR] The given Conv2D padding was not recognized...")

    n_c, in_dim_x, in_dim_y = frame.shape       # frame dimensions assuming square shaped frame

    sx = stride[0]                              # stride stepsize in x direction
    sy = stride[1]                              # stride stepsize in y direction

    out_dim_x = int((in_dim_x - fx)/sx) + 1     # calculate x-output dimension
    out_dim_y = int((in_dim_y - fy)/sy) + 1     # calculate y-output dimension

    assert n_c == n_c_f, "Dimensions of kernel must match dimensions of the input frame"

    out = np.zeros((n_f, out_dim_x, out_dim_y))
        
    # convolve the kernel over every part of the frame, adding the bias at each step.
    for curr_f in range(n_f):
        curr_y = out_y = 0
        while curr_y + fy <= in_dim_y:
            curr_x = out_x = 0
            while curr_x + fx <= in_dim_x:
                out[curr_f, out_y, out_x] = np.sum(kernel[curr_f] * frame[:,curr_y:curr_y+fy, curr_x:curr_x+fx]) + bias[curr_f]
                curr_x += sx
                out_x += 1
            curr_y += sy
            out_y += 1
    return out

##-----------------------------------------------------------------------------------------------
def maxpool(frame, f=(2,2), s=(2,2), padding='none'):
    """
    Downsample 'frame' using kernel size (fx, fy) and stride (sx, sy)
    """
    (fx, fy) = f
    (sx, sy) = s
    
    # Add padding to the frame...
    npad = ( (0,0), (int(fx/2), int(fy/2)), (int(fx/2), int(fy/2)) )    # Only rows and columns of frames are padded...
    if (padding == None or padding.lower() == 'none'):
        pass
    elif (padding.lower() == 'zeros'):
        frame = np.pad(frame, pad_width=npad, mode='constant', constant_values=0)
    elif (padding.lower() == 'edge'):
        frame = np.pad(frame, pad_width=npad, mode='edge')
    elif (padding.lower() ==  'wrap'):
        frame = np.pad(frame, pad_width=npad, mode='wrap')
    else:
        print("[ERROR] The given MaxPool padding was not recognized...")
        exit(0) 
    
    n_c, h_prev, w_prev = frame.shape

    h = int((h_prev - fx)/sx) + 1
    w = int((w_prev - fy)/sy) + 1

    downsampled = np.zeros((n_c, h, w))

    # Downsampling with MaxPooling...
    for i in range(n_c):
        # slide maxpool window over each part of the frame and assign the max value at each step to the output
        curr_y = out_y = 0
        while curr_y + fx <= h_prev:
            curr_x = out_x = 0
            while curr_x + fy <= w_prev:
                downsampled[i, out_y, out_x] = np.max(frame[i, curr_y:curr_y+fy, curr_x:curr_x+fx])
                curr_x += sx
                out_x += 1
            curr_y += sy
            out_y += 1
    return downsampled

##-----------------------------------------------------------------------------------------------
def flatten(frame):
    """
    Get the frame input sizes and use those values to flatten the matrix to a vector
    """
    (nf2, dim2x, dim2y) = frame.shape
    return frame.reshape((nf2 * dim2x * dim2y, 1))                    # flatten layer

##----------------------------------------------------------------------------------------------- 
def dense(frame, waights, biases):
    """
    Apply dot product between the input frame and the wights matrix. Then add the biases per neuron.
    Finally apply the chosen activation function on the result matrix
    """
    out = waights.dot(frame) + biases                                 # Dense layer calculation
    return out

##-----------------------------------------------------------------------------------------------
def normalize(frame, norm, meanFrame, stdFrame, maxFrame):
    """
        Function allowing to normalize frame(s)
    """
    # Normalize the data...
    if norm == 'byStd':
        frame -= meanFrame
        frame /= stdFrame
    elif norm == 'by255':
        frame = np.true_divide(frame, 255)    
    return frame



