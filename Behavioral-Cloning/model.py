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








