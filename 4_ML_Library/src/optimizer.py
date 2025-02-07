#################################################################################################
# Optimizer for the training procedure
# (https://cs231n.github.io/optimization-1/):
# Optimization is the process of finding the set of parameters W that minimize the loss function
# 
# Author: Arasch Lagies
# First Version: 12/18/2019
# Last update: 08/28/2020
#
# This function library is called by sequential.py
#################################################################################################
import os
import numpy as np
# For multiprocessing per batch...
from multiprocessing import cpu_count, Pool
from functools import partial
# ................................
from inference import *
from loss import *


def adamGD(batch, num_classes, lr, dim, n_c, beta1, beta2, epsilon, params, cost, parallel):
    """ADAM optimiser
        1. update the parameters through Adam gradient decent
        Note: 
        params[0] contains waights/filters of the first layer. 
        On location 0 of the list is "frame" and the dimensions of the input frame
        In case of Conv2D:
            params[1]['Layer'] is the layer name for layer (here Conv2D), 
            params[1]['Kernels'] is a tuple containing the layer parameters
            params[1]['Biases'] are the biases of the Conv2D layer
            params[1]['Stride] is a tuple with the stride values for the Conv2D layer
            paramd[1]['Padding'] spcifies for Conv2D and MaxPool the padding 
            params[1]['Activation'] is the name of the activation function after the Conv2D layer
        """
    X = batch[:,0:-1]       # get batch inputs
    X = X.reshape(len(batch), n_c, dim, dim)
    Y = batch[:,-1]         # get batch lebels

    cost_ = 0
    batch_size = len(batch)

    max_number_processes = batch_size          # Try to use as many cores for the conv computation as kernels - Hovewer the Pool function will only use as many cores as available

    # Initialize gradients and momentum, RMS params
    dfw = []
    db  = []
    v   = []
    bv  = []
    s   = []
    bs  = []

    # Instantiate inference
    infer = inference(params)
    for num, lay in enumerate(params):
        if lay.get("Layer")[:6] == "Conv2D" or lay.get("Layer")[:5] == "Dense":
            dfw.append(np.zeros(lay["Weights"].shape))
            db.append(np.zeros(lay["Biases"].shape))
            v.append(np.zeros(lay["Weights"].shape))
            bv.append(np.zeros(lay["Biases"].shape))
            s.append(np.zeros(lay["Weights"].shape))
            bs.append(np.zeros(lay["Biases"].shape))
        else:                                                                        # empty locations in the list, just so the addressing further down is simpler
            dfw.append(np.zeros(10))                                                 # Using here for now a 10 as the max list length for the patameters.. 
                                                                                     # These are just placeholders and not further used
            db.append(np.zeros(10))
            v.append(np.zeros(10))
            bv.append(np.zeros(10))
            s.append(np.zeros(10))
            bs.append(np.zeros(10))


    if parallel:        # Training using multiprocessing...
        # Collecting input arguments for multiprocessing Pool...
        func_arguments = partial( infWorker, X=X, Y=Y, num_classes=num_classes, infer=infer )
        loss_  = []
        grads_ = []
        # The iterables for the multiprocessing Pool...
        iterables = [i for i in range(batch_size)]
        with Pool(processes=max_number_processes) as pool:
            result = pool.map( func_arguments, iterables )
        for n, r in enumerate(result):
            # print(f"result = {r}")
            loss_.append(r[0])
            grads_.append(r[1])
        for i in range(batch_size):
            loss  = loss_[i]
            grads = grads_[i]
            for layerNum, df_, db_ in grads:
                #layerNum = j                                                            # Required, as location 0 is occupied by the input frame
                # Sum up weights/filters and biases
                dfw[layerNum] += df_
                db[layerNum]  += db_
            cost_ += loss
    else:               # Single process training...
        for i in range(batch_size):
            x = X[i]
            y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1)                   # convert label to one-hot

            ## Collect gradients for training 
            # Forward pass
            infer.forward(x)
            # Calculate the loss
            loss = infer.lossCalc(y)
            # Backward pass
            grads = infer.backward()

            for layerNum, df_, db_ in grads:
                #layerNum = j                                                            # Required, as location 0 is occupied by the input frame
                # Sum up weights/filters and biases
                dfw[layerNum] += df_
                db[layerNum]  += db_
            cost_ += loss

    # ADAM parameter update...
    for i, dfb in enumerate(zip(dfw, db)):
        # Calculate momentum and RMSProp for each layer's weight/filter and bias...:
        if params[i]["Layer"][:6] == "Conv2D" or params[i]["Layer"][:5] == "Dense": # Add weights/filters and biases only for 'Conv2D' and 'Dense' 
            v[i] = beta1*v[i] + (1-beta1)*dfb[0]/batch_size                         # Momentum update
            s[i] = beta1*s[i] + (1-beta1)*(dfb[0]/batch_size)**2                    # RMSProp update
            params[i]["Weights"] -= lr * v[i]/np.sqrt(s[i] + epsilon)                  # Combine momentum and RMSProp to perform update with Adam

            bv[i] = beta1*bv[i] + (1-beta1)*dfb[1]/batch_size
            bs[i] = beta2*bs[i] + (1-beta2)*(dfb[1]/batch_size)**2
            params[i]["Biases"] -= lr * bv[i]/np.sqrt(bs[i] + epsilon)

    cost_ = cost_/batch_size
    cost.append(cost_)

    return params, cost

# Worker function for multiprocessing ...
def infWorker(i, X, Y, num_classes, infer):
    x = X[i]
    y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1)                   # convert label to one-hot

    ## Collect gradients for training 
    # Forward pass
    infer.forward(x)
    # Calculate the loss
    loss = infer.lossCalc(y)
    # Backward pass
    grads = infer.backward()
    ret = [loss,grads]
    return ret

