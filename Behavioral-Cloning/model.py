import tensorflow as tf
import numpy as np
import cv2
import csv

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import time


#load logging data
lines = []
with open('./Data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
		#print(line)

# buffers to store the original images and steering angles		
images = []
measurements = []

'''
# only load the center camera data
for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = './Data/IMG/' + filename
	#print(current_path)
	image = cv2.imread(current_path)
	#print(image)
	images.append(image)
	measurement = line[3]
	measurements.append(measurement)
'''

#load all center left and right camera data
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './Data/IMG/' + filename
        #print(current_path)
        image = cv2.imread(current_path)
        #print(image)
        images.append(image)
	# correction angle for the left and right camera data
    correction = 0.25
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(measurement + correction)
    measurements.append(measurement - correction)

# augment the original input dataset by flipping 
# the original image and adjust the correspoding steering angle
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image, 1)
    #print(measurement)
    flipped_measurement = measurement * (-1.0)
    #print(measurement)
    #print(flipped_measurement)
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)
	
x_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)








