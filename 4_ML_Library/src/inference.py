###########################################################
# DNN training routine
# Author: Arasch Lagies
# 
# First Version: 12/26/2019
# Latest Update: 08/28/2020
#
# Forward and backward propagation calls for model training
###########################################################
import os
from layers import *
from loss import *
from backwardsLayers import *
from utils import *
from math import *


class inference:
    def __init__(self, params):
        """
        The 'inference' class contains the forward and backward calls of the model designed by the user.
        'forward' + 'backward' are used during training.
        'forward' only is used during inference.
        """
        self.params = params

    def forward(self, frame, replace="No Replace"):
        """
        The Forward function is used during training for the forward pass of the batch data through the network.
        During prediction / inference this function is used with pre-trained weights and biases for predictions.
        'params' is a list of dictionaries, each containing all data spwcific to the layer, inlucing settings, trained pareters, input and output frames.'.
        The input parameter 'replace' specifies which layer should be replaced with frames processed by an external Accelerator.
        !!! So far only the 'Conv2D' layer can be replaced. By doing so also the Forward Frame layer is skipped !!!
        """
        for n, layer in enumerate(self.params):
            if layer.get("Layer") == "Input Frame":
                if replace == "No Replace":
                    self.params[n]["Forward Frame"] = frame
            #----------------------------------------------------------------------------------------------------------------------------------------
            elif layer.get("Layer")[:6] == "Conv2D":
                if replace == "Conv2D":
                    pass
                elif replace == "scaleTest":
                    # Using the scaled Weights and Biases (as decimal integer values ) as provided for the Accelerator for Conv2D operation
                    frame = convolution(frame, layer["Scaled Weights"], layer["Scaled Biases"], layer["Stride"], layer["Padding"])    
                    # Apply the activation function
                    if layer["Activation"].lower() == "relu":
                        frame = relu(frame)
                    print(f"[INFO] The Frames generated with the scaled Weights and Biases has a max of {frame.max()} and a min of {frame.min()}")
                    # Scale output back using the scaling factor of weights and biases used for accelerator scaling...
                    scale = frame.max()/255.   #layer["Scale Factor"])
                    frame = np.true_divide( frame, scale )     
                    print(f"[INFO] After dividing all pixels by the scaling factor {scale}, the max value is {frame.max()} and the min is {frame.min()}.")
                else:
                    frame = convolution(frame, layer["Weights"], layer["Biases"], layer["Stride"], layer["Padding"])    
                    # Apply the activation function
                    if layer["Activation"].lower() == "relu":
                        frame = relu(frame)
                    # print(f"[INFO] Using trained Weights and Biases in floating point the max value is {frame.max()} and the min is {frame.min()}.")

                # display(frame, nrows=round(sqrt(layer["Kernels"])), ncols=round(sqrt(layer["Kernels"])), data_size=layer["Kernels"], title_ = "Convolution Output Frames")
                self.params[n]["Forward Frame"] = frame
            #----------------------------------------------------------------------------------------------------------------------------------------
            elif layer.get("Layer")[:9] == "Normalize":
                if(np.min(frame) == np.max(frame) == 0):
                    meanFrame = 0
                    stdFrame  = 1
                    maxFrame  = 0
                else:
                    meanFrame = np.mean(frame)
                    stdFrame  = np.std(frame)
                    maxFrame  = np.max(frame)
                frame = normalize(frame, layer["Norm Choice"], meanFrame, stdFrame, maxFrame )
                self.params[n]["Forward Frame"] = frame
                self.params[n]["Mean of Frame"] = meanFrame
                self.params[n]["Std of Frame"]  = stdFrame
                self.params[n]["Max of Frame"]  = stdFrame
            #----------------------------------------------------------------------------------------------------------------------------------------
            elif layer.get("Layer")[:9] == "MaxPool2D":
                frame = maxpool(frame, layer["Pool Size"], layer["Stride"], layer["Padding"])   
                self.params[n]["Forward Frame"] = frame  
            #----------------------------------------------------------------------------------------------------------------------------------------                                                           
            elif layer.get("Layer")[:7] == "Flatten":
                frame = flatten(frame)
                self.params[n]["Forward Frame"] = frame
            #----------------------------------------------------------------------------------------------------------------------------------------
            elif layer.get("Layer")[:5] == "Dense":
                frame = dense(frame, layer["Weights"], layer["Biases"])                                             
                # Apply the actination function
                if layer["Activation"].lower() == "relu":
                    frame = relu(frame)
                elif layer["Activation"].lower() == "softmax":
                    frame = softmax(frame)
                self.params[n]["Forward Frame"] = frame
            #----------------------------------------------------------------------------------------------------------------------------------------
        self.probs = frame
        return np.argmax(self.probs), np.max(self.probs)
        

    def lossCalc(self, label):
        """
        This inference module calls desired loss calculation function from the loss.py function library.
        """
        self.label = label
        if self.probs.any() <= 0:
            input("Got a <=0 and waiting for continuation")
        self.loss = categoricalCrossEntropy(self.probs, self.label)
        return self.loss


    def backward(self):
        """
        Backpropagation used during training to train weights and biases of the trainable layers.
        The backpropagation algorithm adjusts weights and biases using gradient descent and the Optimizer algorithm (so far only ADAM) to 
        determine the adequate stepzise for the optimization process.
        """
        self.grads = tuple()
        dout = self.probs - self.label                                          # Derivative of the loss w.r.t. final layer output

        dwlast = dout.dot(self.params[-1]['Biases'].T)                          # Loss gradient of final dense layer weights
        dblast = np.sum(dout, axis = 1).reshape(self.params[-1]["Biases"].shape)

        prev_grad = dout
        # Backpropagate through Activation function (here ReLu)
        if self.params[-1]["Activation"].lower() == "relu":
            pre[dlayer["Forward Frame"]<=0] = 0                                 # Backpropagate through ReLu

        for num, dlayer in reversed(list(enumerate(self.params))):              # Iterate backward through the list of processed layers
            if num==0:                                                          # Don't process the input frame
                break
            #----------------------------------------------------------------------------------------------------------------------------------------
            if dlayer.get("Layer")[:5] == "Dense":
                # Backpropagate through Activation function (here ReLu)
                if dlayer["Activation"].lower() == "relu":
                    dImg[dlayer["Forward Frame"]<=0] = 0                        # Backpropagate through ReLu

                # Lossgradient of dense layer outputs
                dImg, dw, db = denseBackward(prev_grad, self.params[num-1]["Forward Frame"], dlayer["Weights"], dlayer["Biases"])
                prev_grad = dImg

                self.grads += ((num, dw, db),)
            #----------------------------------------------------------------------------------------------------------------------------------------
            elif dlayer.get("Layer")[:7] == "Flatten":
                # Reshape into the dimensions of the next layer (her e.g. a MaxPool layer)
                dImg = flattenBackward(prev_grad, self.params[num-1]["Forward Frame"])
                prev_grad = dImg
            #----------------------------------------------------------------------------------------------------------------------------------------
            elif dlayer.get("Layer")[:9] == "MaxPool2D":
                # Backprop through the max-pooling layer (only neurons with the highest activation in the window get updated)
                prev_grad = prev_grad.reshape(dlayer["Forward Frame"].shape)    # Reshape the previous layer to the dimensions of the original MaxPooling layer
                                                                                # (The previous layer was flattened in the Flatten step)

                dImg = maxpoolBackward(prev_grad, self.params[num-1]["Forward Frame"], dlayer["Pool Size"], dlayer["Stride"])  
                prev_grad = dImg
            #----------------------------------------------------------------------------------------------------------------------------------------
            elif dlayer.get("Layer")[:9] == "Normalize":
                # Backpropagation undoing the normalization layer
                dImg = normalizeBackward(prev_grad, dlayer["Norm Choice"], dlayer["Mean of Frame"], dlayer["Std of Frame"], dlayer["Max of Frame"])
            #----------------------------------------------------------------------------------------------------------------------------------------
            elif dlayer.get("Layer")[:6] == "Conv2D":
                # Backpropagate through activation layer (ReLu)
                if dlayer["Activation"].lower() == 'relu':
                    dImg[dlayer["Forward Frame"]<=0] = 0                        # Backpropagate through ReLu layer

                # Backpropagate previous gradient through convolutional layer...
                dImg, df, db = convolutionBackward(prev_grad, self.params[num-1]["Forward Frame"], dlayer["Weights"], dlayer["Stride"])      
                prev_grad = dImg

                self.grads += ((num, df, db),)
            #----------------------------------------------------------------------------------------------------------------------------------------

        return self.grads



