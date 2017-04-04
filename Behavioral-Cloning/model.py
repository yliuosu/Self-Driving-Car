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

'''
# first training model a single layer network
model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))
'''

'''
# second training model a Lenet 5 network
model = Sequential()
#normalize the image data to -1 to 1
model.add(Lambda(lambda x: (x / 127.5) - 1, input_shape=(160,320,3)))
model.add(Convolution2D(32, 5, 5, input_shape=(160, 320, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 5, 5))
model.add(MaxPooling2D((2, 2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.6))
model.add(Dense(1))
'''
# thrid training model learn from the 
# learn from NVIDIA’s self-driving car
# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
model = Sequential()
#normalize the image data to -0.5 to 0.5
model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((63,25),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample = (2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample = (2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample = (2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.75))
model.add(Dense(50))
model.add(Dropout(0.75))
model.add(Dense(10))
model.add(Dense(1))

# use mean-squared error to calculate the loss
# use adam to train the model
model.compile(loss='mse', optimizer = 'adam', metrics=['accuracy'])
# calculate the training time
t0 = time.time()
# split the whole data to train dataset 80% and validation dataset 20%
# shuffle the dataset for each epoch of training
# train the network in three epochs
history = model.fit(x_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 10)
t1 = time.time() 
print("Time: %.3f seconds" % (t1 - t0))








