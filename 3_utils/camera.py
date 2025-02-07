########################################################################
# Utility tool to capture images using a webcam for CNN model training.
# Capture video stream from camera using OpenCV.
#
# Author: Arasch U Lagies
# First Version: 05/27/2020
# Last Update: 05/27/2020
#
# This class needs to be instanced by another fuction.
# To capture images the function collectImages.py needs to be executed,
# which inherits this class library...
########################################################################
import argparse
import cv2
import time

CWIDTH  = 640
CHEIGHT = 480
CFPS    = 30
WEBCAM = 0

class stream:
    def __init__(self, width = CWIDTH, height = CHEIGHT, fps = CFPS, webcam = WEBCAM):
        self.width = width
        self.height = height
        self.fps = fps
        self.webcam = webcam

        self.camera = cv2.VideoCapture(self.webcam)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        print("[INFO] Camera warming up...")
        time.sleep(5)
        print("Done...")

    def get(self):
        (self.grabbed, self.frame) = self.camera.read()

    def show(self):
        cv2.imshow("Image", self.frame)

    def shut(self):
        fps_meas = self.camera.get(cv2.CAP_PROP_FPS)
        print("[INFO] Camera speed was {} fps".format(fps_meas))
        self.camera.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--width", required=False,
       default=CWIDTH, type=int, help="Width of video frames...")
    ap.add_argument("-e", "--height", required=False,
       default=CHEIGHT, type=int, help="Height of video frames...")
    ap.add_argument("-f", "--fps", required=False,
       default=CFPS, type=int, help="Fps of the video stream...")
    ap.add_argument("-c", "--camera", required=False,
       default=CFPS, type=int, help="Choose camera...")

    args = vars(ap.parse_args())
    width = args["width"]
    height = args["height"]
    fps = args["fps"]
    webcam = args["camera"]

    cam = stream(width, height, fps, webcam)

    while(1):
        cam.get()
        cam.show()
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cam.shut()