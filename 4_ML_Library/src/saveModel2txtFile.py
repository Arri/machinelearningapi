###########################################################################################################################
# Formatted and human readable saving of all training parameter and results to a text file ...
#
# Author: Arasch Lagies
# First Version: 3/2/2020
# Latest Update: 3/2/2020
#
###########################################################################################################################
import os
import csv


def saveFormatted( save_path='None', params='None', cost='None', scaleTo='None', convertTo='None' ):
    if (save_path=='None' or params=='None'):              # No input provided to save to a txt file...
        return

    to_save = [params, cost]
    txt_path = os.path.splitext(save_path)[0] + '.csv'
    with open(txt_path, "w", newline='') as csvFile:
        writer = csv.writer(csvFile)
        ## Save the list of dictionaries
        for litem in to_save:
            for subitem in litem:
                weightSC = []
                biasSC   = []
                weightB  = []
                biasB    = []
                try:
                    for key, val in subitem.items():
                        # To sort matrices in columns...
                        if (subitem["Layer"] == "Input Frame" ):
                            if (key == "Layer"):
                                writer.writerow([key, val ])
                            else:
                                writer.writerow([ '', key, val ])
                        elif (subitem["Layer"][:6] == "Conv2D"):
                            if (key ==  "Layer"):
                                writer.writerow([ key, val ])
                            elif (key == "Weights"):
                                writer.writerow(['', key, "Shape = {}".format(val.shape) ])   
                                row = [v for v in val]
                                writer.writerow(['']+['']+['Trained Weights orig.']+row)
                                if ('Bit Depth' in subitem):
                                    writer.writerow(['', '', 'Bit Depth', scaleTo])
                                if ('Scale Factor' in subitem):                                       # in case a bit value was chosen to which to scale wrights and biases of Conv2D layers
                                    writer.writerow([ '', '', 'Scaling factor for {}-bit conversion'.format(scaleTo), subitem["Scale Factor"] ])
                                if ('Scaled Weights' in subitem): 
                                    binw = [v for v in subitem["Scaled Weights"]]
                                    writer.writerow(['']+[''] + ['{}-bit + 1-bit sign scaled weights: '.format(scaleTo-1)] + binw)
                                if ('Binary Weights' in subitem):
                                    binb = [v for v in subitem["Binary Weights"]]
                                    writer.writerow(['']+[''] + [convertTo] + binb)
                            elif (key == "Biases"):
                                writer.writerow([ '', key, 'Trained Biases orig.', val, "Shape = {}".format(val.shape) ])
                                if ("Scaled Biases" in subitem):
                                        writer.writerow([ '', '', "{}-bit scaled biases: ".format(scaleTo), subitem["Scaled Biases"] ])
                                if ("Binary Biases" in subitem): 
                                        writer.writerow(['', '', convertTo, subitem["Binary Biases"]])  
                            elif (key == "Forward Frame"):
                                writer.writerow([ '', key, val, "Shape = {}".format(val.shape) ])
                            elif (key == "Kernels"):
                                writer.writerow([ '', key, val ])        
                            else:
                                if not key=='Scale Factor' and not key=='Scaled Weights' and not key=='Binary Weights' and not key=="Scaled Biases" and not key=="Binary Biases":
                                    writer.writerow([ '', key, val ])
                        elif (subitem["Layer"][:9] == "Normalize"):
                            if (key == "Layer"):
                                writer.writerow([ key, val ])
                            else:
                                writer.writerow(['', key, val])
                        elif (subitem["Layer"][:9] == "MaxPool2D"):
                            if (key =="Layer"):
                                writer.writerow([ key, val ])
                            else:
                                writer.writerow(['', key, val])
                        elif (subitem["Layer"][:7] == "Flatten"):
                            if (key == "Layer"):
                                writer.writerow([ key, val ])
                            else:
                                writer.writerow(['', key, val])
                        elif (subitem["Layer"][:5] == "Dense"):
                            if (key == "Layer"):
                                writer.writerow([ key, val ])
                            elif (key == "Weights"):
                                writer.writerow([ '', key, "Shape = {}".format(val.shape) ])   
                                row = [v for v in val]
                                writer.writerow(['']+['']+['Trained Weights']+row)
                            elif (key == "Biases"):
                                writer.writerow([ '', key, 'Trained Biases', val, "Shape = {}".format(val.shape) ])
                            else:
                                writer.writerow([ '', key, val ])
                except:                                                                     # The costs are not in the same dict format as the other layers...
                    if subitem == "Cost History":
                        writer.writerow([ subitem, "Length = {} cycles".format(len(cost)) ])
                    else:
                        writer.writerow([ subitem ])

    return

