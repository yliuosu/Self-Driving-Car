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

# in order to have a better understanding of the label data 
# display the distribution of the label data
fig, ax = plt.subplots(figsize=(15,5))
ax.hist(y_train, bins = 40, label='steering')
ax.set_xlabel('Steering Angle')
ax.set_ylabel('Number of Frames')
plt.show()

# crop the image by removing the top 63 lines and bottom 25 lines of data
# because those lines of data don't reflect the shape of the road 
frame_index = np.random.randint(1, x_train.shape[0] + 1)
image = x_train[frame_index]
fig = plt.figure(figsize=(25,25))
ax = fig.add_subplot(1, 2, 1)
ax.set_xlabel('Original Image')
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
cropped_image = image[63:-25,:,:]
ax = fig.add_subplot(1, 2, 2)
ax.set_xlabel('Cropped Image')
plt.imshow(cv2.cvtColor(flipped_image,cv2.COLOR_BGR2RGB))
plt.show()








