###########################################################################
# Activation functions
# Author: Arasch Lagies
# Based on: Alejandro Escontrela
#
# First Version: 12/27/2019
# Latest Update: 12/27/2019
###########################################################################
import numpy as np

def relu(frame):
    frame[frame<=0] = 0
    return frame

def softmax(frame):
    out = np.exp(frame)
    summe = np.sum(out)
    if (not summe==0):
        return out/np.sum(out)
    else:
        return 0
