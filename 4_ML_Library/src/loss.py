#####################################################################
# Loss Functions
# Author: Arasch Lagies
# Based on: 
#     Alejandro Escontrela ( https://github.com/Alescontrela/Numpy-CNN ) 
#     https://towardsdatascience.com/common-loss-functions-in-machine-learning-46af0ffc4d23
#
# First Version: 12/27/2019
# Last Update:02/04/2019
######################################################################
import numpy as np


def categoricalCrossEntropy(probs, label):
    return -np.sum(label * np.log(probs))

# Cross Etropy Loss / Negative Log Likelihood
def cross_entropy(probs, label, epsilon=1e-10):
    probs = np.clip(probs, epsilon, 1. - epsilon)
    N = probs.shape[0]
    ce_loss = -np.sum(np.sum(label * np.log(probs + 1e-5)))/N
    return ce_loss

# Mean absolute Error:
def mae(probs, label):
    differences = probs - label
    absolute_differences = np.absolute(differences)
    mean_absolute_differences = absolute_differences.mean()
    return mean_absolute_differences

# Mean Square Error:
def rmse(probs, label):
    differences = probs - label
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)
    return rmse_val