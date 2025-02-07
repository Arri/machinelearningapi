###############
# Convert data into csv files
# Arasch Lagies
# 1/14/2021

import os
import csv
import gzip
import numpy as np
import pandas as pd

xDatafile = './test/t10k-images-idx3-ubyte.gz'
yDatafile = './test/t10k-labels-idx1-ubyte.gz'

DataSize  = 10000
Take      = 10
Width     = 28
Height    = 28
Depth     = 1

class convert:
    def __init__(self, xData=xDatafile, yData=yDatafile, dsize = DataSize, width=Width, height=Height, depth=Depth):
        self.xData = xData
        self.yData = yData
        self.dsize = dsize
        self.width = width
        self.height = height
        self.depth = depth

    def gzTestdat(self):
        """ Load Testing data that was compressed in a gz file and put all in memory """
        # Load data...
        self.testX = self.extract_data(self.xData, self.dsize, self.width)
        self.testY = self.extract_labels(self.yData, self.dsize).reshape(self.dsize,1)
        # Calculate the dimensions of the testing frames...
        self.imageHeight = self.testX.shape[1] // self.width
        print(f"[INFO] Test data img_depth = {self.depth}, img_height = {self.imageHeight}, img_width = {self.width}.")
        return self.testX, self.testY, self.imageHeight

    def saveCSV(self, amount=10, filename="mnist10.csv"):
        # print(self.testX[0].reshape(self.height,self.width))
        # exit(0)
        with open(filename, 'a') as f:
            # csvw = csv.writer(f)
            for img in range(amount):
                mg = np.asarray(self.testX[img].reshape(self.height,self.width))
                np.savetxt(f, self.testY[img].astype(int), fmt='%i', delimiter="")
                np.savetxt(f, mg.astype(int), fmt='%i', delimiter=",")
                # mg = np.asarray(self.testX[img].reshape(self.height,self.width))
                # csvw.writerow(self.testY[img])
                # csvw.writerow(mg)
                # csvw.writerows('')
            # mg = np.asarray(self.testX[0].reshape(self.height,self.width))
            # np.savetxt(filename, mg, delimiter=",")
        # with open(filename, 'w+') as f:
        #     # csvw = csv.writer(f)
        #     for img in range(amount):
        #         # csvw.writerow(self.testY[img])
        #         # for row in self.testX[img]:
        #         # csvw.writerow(self.testX[img])

    def extract_data(self, filename, num_images, IMAGE_WIDTH):
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

    def extract_labels(self, filename, num_images):
        '''
        Extract label into vector of integer values of dimensions [m, 1], where m is the number of images.
        '''
        print('[INFO] Extracting', filename)
        with gzip.open(filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * num_images)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels

if __name__=="__main__":
    c = convert()
    c.gzTestdat()
    c.saveCSV()
