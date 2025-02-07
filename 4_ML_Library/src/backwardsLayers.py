#####################################################################
# Backwads calculation of layers
# Author: Arasch Lagies
# Based on: Alejandro Escontrela ( https://github.com/Alescontrela/Numpy-CNN )
# First Version: 30/12/2019
# Last Update: 02/04/2020
####################################################################

import numpy as np

##-----------------------------------------------------------------------------------------------
def nanargmax(arr):
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs

##-----------------------------------------------------------------------------------------------
def convolutionBackward(dconv_prev, conv_in, filt, s_, padding=None):
    """
    Backpropagation through a convolutional layer.
    """
    # For now assuming stride is in x and y same
    s = s_[0]
    (n_f, n_c, f, _) = filt.shape
    (_, orig_dim, _) = conv_in.shape
    ## Initialize derivatives
    dout = np.zeros(conv_in.shape)
    dfilt = np.zeros(filt.shape)
    dbias = np.zeros((n_f, 1))
    for curr_f in range(n_f):
        # loop through all filters
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                # loss gradient of filter (used to update the filter)
                dfilt[curr_f] += dconv_prev[curr_f, out_y, out_x] * conv_in[:, curr_y:curr_y+f, curr_x:curr_x+f]
                # loss gradient of the input to the convolution operation (conv1 in the case of this network)
                dout[:, curr_y:curr_y+f, curr_x:curr_x+f] += dconv_prev[curr_f, out_y, out_x] * filt[curr_f]
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        # loss gradient of the bias
        dbias[curr_f] = np.sum(dconv_prev[curr_f])

    return dout, dfilt, dbias

##-----------------------------------------------------------------------------------------------
def maxpoolBackward(dpool, orig, kernel, stride, padding=None):
    """
    Backpropagation through a maxpooling layer. The gradients are passed through the indices if greatest value in the 
    original maxpooling during the forward step.

    """
    # For now assuming filter and frame are qudratic
    f = kernel[0]
    s = stride[0]
    (n_c, orig_dim, _) = orig.shape

    dout = np.zeros(orig.shape)

    for curr_c in range(n_c):
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                # obtain index of largest value in input for current window
                (a, b) = nanargmax(orig[curr_c, curr_y:curr_y+f, curr_x:curr_x+f])
                dout[curr_c, curr_y+a, curr_x+b] = dpool[curr_c, out_y, out_x]

                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1

    return dout

##-----------------------------------------------------------------------------------------------
def denseBackward(dout, prevImg, weight, bias):
    dw = dout.dot(prevImg.T)
    db = np.sum(dout, axis=1).reshape(bias.shape)
    dImg = weight.T.dot(dout)

    return dImg, dw, db

##-----------------------------------------------------------------------------------------------
def flattenBackward(dout, prevImg):
    return dout.reshape(prevImg.shape)

##-----------------------------------------------------------------------------------------------
def normalizeBackward(frame, norm, meanFrame, stdFrame, maxFrame):
    """
        Function allowing to normalize frame(s)
    """
    # Normalize the data...
    if norm == 'byStd':
        frame *= stdFrame
        frame += meanFrame     
    elif norm == 'by255':
        frame = np.multiply(frame, maxFrame)    

    return frame