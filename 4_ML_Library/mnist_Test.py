	
"""
MNIST Image frame for manual testing
Author: Arasch Lagies

"""
import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pygame
import tensorflow as tf
import numpy as np
import platform as pf
import matplotlib.pyplot as plt

from API.inference import *
from API.sequential import *

# Define some colors
BLACK = (0, 0, 0)
GRAY  = (120, 120, 120)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
 
# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 10
HEIGHT = 10
ROWS = 28
COLS = 28
 
# This sets the margin between each cell
MARGIN = 2

# Create a 2 dimensional array. A two dimensional
# array is simply a list of lists.
grid = []
for row in range(ROWS):
    # Add an empty array that will hold each cell
    # in this row
    grid.append([])
    for column in range(COLS):
        grid[row].append(0)  # Append a cell
 
# Initialize pygame
pygame.init()
 
# Set the HEIGHT and WIDTH of the screen
rowsize = ROWS * HEIGHT + MARGIN*ROWS
colsize = COLS * WIDTH + MARGIN*COLS
WINDOW_SIZE = [rowsize, colsize]
screen = pygame.display.set_mode(WINDOW_SIZE)
 
# Set title of screen
pygame.display.set_caption("MNIST Model Testing")
 
# Loop until the user clicks the close button.
done = False

# Start drawing at left mouse click and stop at right mouse click
START_DRAWING = False
 
# Used to manage how fast the screen updates
clock = pygame.time.Clock()
 
# -------- Main Program Loop -----------
while not done:
    for event in pygame.event.get():  # User did something
        # Check if left mouse button is pressed:
        if event.type == pygame.MOUSEBUTTONDOWN:
            START_DRAWING = True
        elif event.type == pygame.MOUSEBUTTONUP:
            START_DRAWING = False

        if event.type == pygame.QUIT:  # If user clicked close
            done = True  # Flag that we are done so we exit this loop
        elif event.type == pygame.MOUSEMOTION and START_DRAWING:   #MOUSEBUTTONDOWN:
            # User clicks the mouse. Get the position
            pos = pygame.mouse.get_pos()
            # Change the x/y screen coordinates to grid coordinates
            column = pos[0] // (WIDTH + MARGIN)
            row = pos[1] // (HEIGHT + MARGIN)
            # Set that location to one
            grid[row][column] = 1
            # Make the surrounding pixels gray
            for r in [-1,0,1]:
                for c in [-1,0,1]:
                    if grid[row+r][column+c] == 0:
                        grid[row+r][column+c] = 0.8
            # print("Click ", pos, "Grid coordinates: ", row, column)
 
    # Set the screen background
    screen.fill(BLACK)
 
    # Draw the grid
    for row in range(ROWS):
        for column in range(COLS):
            color = WHITE
            if grid[row][column] == 1:
                color = BLACK
            if grid[row][column] == 0.8:
                color = GRAY
            pygame.draw.rect(screen,
                             color,
                             [(MARGIN + WIDTH) * column + MARGIN,
                              (MARGIN + HEIGHT) * row + MARGIN,
                              WIDTH,
                              HEIGHT])
 
    # Limit to 60 frames per second
    clock.tick(60)
 
    # Go ahead and update the screen with what we've drawn.
    pygame.display.flip()

# Load the TF model...
print()
print("=============================================================================================")
print("Loading the TensorFlow2 model...")
inModel = './TF2_reference/modelSave/model_20-06-2020_07-17-54_PM_.h5'
print(f"[INFO] Loading model file {inModel}.")
model = tf.keras.models.load_model(inModel)
# print(model.summary())
test_images = np.asarray(grid)
plt.imshow(grid)
plt.show()
test_images = test_images.reshape(1, 28, 28, 1)
detections = model.predict(test_images)
print("")
print(f"==> TensorFlow 2.1 detected a {np.argmax(detections)} and is {np.amax(detections)*100:.2f}% sure.")
print("=============================================================================================")
# Predict...:
# Read the original model from given pickle file...
print("Loading the Axiado ML Library...")
inModel = './modelSave/ThreeCNN_MaxP_nopad_Dense128_3ep.pkl'
model = Sequential()
params = model.Load_Model(save_path = inModel)

# Set up an instance of the  inference engine
infer = inference(params=params)
frame = np.asarray(grid)
frame = frame.reshape(1, 28, 28)
pred, prob = infer.forward(frame)
print("")
print(f"==> Axiado ML Library detected a {pred} and is {(prob*100):.2f}% sure.")
print("=============================================================================================")
# Be IDLE friendly. If you forget this line, the program will 'hang'
# on exit.
pygame.quit()