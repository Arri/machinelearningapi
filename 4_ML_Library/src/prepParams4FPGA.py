################################################################################
# Read from pickeled model file or take the provided madel parameters 
# and scale the CNN weights and biases, convert them to 7bit binary + 1bit sign
# or to two's complement and save the binary data to a txt file.
#
# Author: Arasch Lagies
#
# Created: 2/20/2020
# Last Update: 2/20/2020
#
# Call:
################################################################################
import os
import pickle
import numpy as np
import csv

CONVERTTO     = 'TwoComplement'
BITDEPTH      = 8  


class prep4FPGA:
    def __init__(self, params=None, convert_to=CONVERTTO, bits=BITDEPTH ):
        self.convert_to    = convert_to
        self.bits          = bits
        self.bmax          = 0       # Max value for the given bits...
        for i in range(self.bits-1): # Minus 1 as weigths and biases need to handle neg numbers (one bit for signage)
            self.bmax += 2 ** i

        self.cnns          = []     # List of dictionaries containing layer name and weights and biases...
        self.scaledCNN     = []     # List of dictionaries containing layer name and SCALED weights and biases...
        self.bin_CNN       = []     # List of dictionaries containing layer name and SCALED weights and biases 
                                    # as 8bit Two's Complement or as 7bit + 1 signage bit...

    def ReadWB(self, params=None):
        """
            Read weights and biases to the Conv2D layers and save in a list of dictionaries....
        """
        if params==None:
            print(f"[ERROR] Please provide a Conv2D layer with kernel Weights and Biases...")
            exit(0)      
        layer=params 

        if layer.get("Layer")[:6] == "Conv2D":
            print(f"[INFO] Received layer {layer['Layer']} with weights shape {layer['Weights'].shape} and bias shape {layer['Biases'].shape}") 
            self.cnns.extend([{key:layer[key] for key in ["Layer", "Weights", "Biases"]}])

    def ScaleWeightsBiases(self):
        """
            To skale the filter and bias values, all values need to be first normalized
            using the ABSOLUTE max of all Weight and Bias values ...

            NOTE: Each layer has its own normalization (betweem its weights+biases). The normalization factor is saved in the
                  layer's dictionary at the key 'Norm Factor'
                  The scale values are same for weights and biases of a layer...
            Then scale all weights and biases of a layer using its individual Norm Factor
        """
        # Find the largest ABSOLUTE number...
        maxVal = -self.bmax           # start value...
        for layer in self.cnns:          
            # Same max for scaling for both Weights and Biases...
            maxf = np.max(np.absolute(layer["Weights"]))
            if maxf > maxVal:
                maxVal = maxf
            maxb = np.max(np.absolute(layer["Biases"]))
            if maxb > maxVal:
                maxVal = maxb
            
            # Scale values using the self.maxVal 
            # by full bit-depth in case of two's complement or
            # by bit-depth/2 in case of binary + 1 signage bit...
            if self.convert_to == "TwoComplement":
                self.scale = self.bmax / maxVal
                layer["Weights"] = self.scale * layer["Weights"]    # Scale Weigths and biases and reduce to integer values...
                layer["Weights"] = layer["Weights"].astype(int)

                layer["Biases"]  = self.scale * layer["Biases"]
                layer["Biases"]  = layer["Biases"].astype(int)
            else:      # Saving values as 7bit binary + 1bit signage...
                self.convert_to = '7+1bit'
                self.scale = self.bmax / maxVal
                layer["Weights"] = self.scale * layer["Weights"]
                layer["Weights"] = layer["Weights"].astype(int)

                layer["Biases"]  = self.scale * layer["Biases"] 
                layer["Biases"]  = layer["Biases"].astype(int)

            collect = {'Layer': layer['Layer'], 'Weights': layer['Weights'], 'Biases': layer['Biases'], 'Norm Factor': self.scale}
            self.scaledCNN.extend([collect])
        return collect["Weights"], collect["Biases"], self.scale

    def ConvertNums(self):
        """
            Convert the Weights and Biases to binary numbers as 2s-Complement or as 7bit+1 signage...
        """
        for layer in self.scaledCNN:
            if self.convert_to == "TwoComplement":
                orig_shape = layer['Weights'].shape
                flat = layer['Weights'].flatten()
                binaryW = np.array([np.binary_repr(v, width=8) for v in flat])
                binaryW = binaryW.reshape(orig_shape)

                orig_shape = layer["Biases"].shape
                flat = layer["Biases"].flatten()
                binaryB = np.array([np.binary_repr(v, width=8) for v in flat])
                binaryB = binaryB.reshape(orig_shape)
            else:
                self.convert_to = '7+1bit'
                orig_shape = layer['Weights'].shape
                flat = layer['Weights'].flatten()
                binaryW = np.array(["{0:08b}".format(v if v>=0 else np.absolute(v)+128) for v in flat])
                binaryW = binaryW.reshape(orig_shape)

                orig_shape = layer["Biases"].shape
                flat = layer["Biases"].flatten()
                binaryB = np.array(["{0:08b}".format(v if v>=0 else np.absolute(v)+128) for v in flat])
                binaryB = binaryB.reshape(orig_shape)

            collect = {'Layer': layer['Layer'], 'Weights': binaryW, 'Biases': binaryB, 'Norm Factor': self.scale, 'Representation': self.convert_to}
            self.bin_CNN.extend([collect])
            print(f"[INFO] Converted layer {layer['Layer']} to the representation {self.convert_to} with bit-deth {self.bits} and a scaling factor {self.scale}.")
            # print(self.bin_CNN)
        return collect["Weights"], collect["Biases"], self.scale

def run():

    prep = prep4FPGA()
    prep.ReadWB()
    prep.ScaleWeightsBiases()
    prep.ConvertNums()
    


if __name__=="__main__":
    run()