###############################################################################
# PyTest code for automated testing of routines of the Machine Learning API
# Testing of the API library file sequential.py
#
# Author: Arasch Lagies
# First Version: 4/8/2020
# Last Update: 4/8/2020
#
# Call:
##############################################################################
import pytest
from train_cnn import *
import os

# Using a fixture to load in test data...

def test_train_cnn():
    # Test if the input frame has the right shape:
    # Train data: (n, width*height) with n=number of frames and image depth=1
    assert len(trainX.shape) == 2 