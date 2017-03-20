import tensorflow as tf
import numpy as np
import cv2
import csv

from keras.models import Sequential
from keras.layers import Flatten, Dense

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#load training data
lines = []
with open('./Data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
		#print(line)

images = []
measurements = []

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

print(len(measurements))
print(len(images))

x_train = np.asarray(images)
y_train = np.asarray(measurements)

print(x_train.shape)
print(y_train.shape)

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss='mse', optimizer = 'adam')
model.fit(x_train, y_train, validation_split = 0.2, shuffle = True)

model.save('model.h5')






