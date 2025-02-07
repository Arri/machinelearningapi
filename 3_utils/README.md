# Utility Functions for Collection of Training and Testing Material

## Required Packages:
To install required packages in Anaconda virtual invironment execute in the utilities folder:

>  conda install --file requirements.txt

In standard pip virtual environment do:

> pip install --upgrade -r requirements.txt

## Discription:

**collectImages.py** is used to collect training und testing images with an attached web-cam.
To start type:

> python collectImages.py **myclass**

You can optionally specify in the call of the fuction:
- (Required) the class name for the pictures to be captured. This name is used as part of the the file name. 
- (Optional) the desired frame width and height - input arguments *--width* and *--height*
- (Optional) the name of the folder to collect the data in (default is ./pictures/) - input argument *--folder*
- (Optional) the amount of pictures to collect per picture - input argument *--pics*

e.g.

> python collectImages.py **myclass** --width **xx** --height **yy** --folder **foldername**
.... Further details TBD