# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./testimages/aheadonly_35.jpg "Traffic Sign 1"
[image5]: ./testimages/nopassing_9.jpg "Traffic Sign 2"
[image6]: ./testimages/speedlimit30_1.jpg "Traffic Sign 3"
[image7]: ./testimages/stop_14.jpg "Traffic Sign 4"
[image8]: ./testimages/turnrightahead_33.jpg "Traffic Sign 5"

[image9]: ./testimages/explorationvisualization.JPG "Exploration Visualization on Dataset"
[image10]: ./testimages/histogram.JPG "Image Sign Count on Training Dataset"
[image11]: ./testimages/rgbtograyscale.JPG "Before and After Change to Gray Scale"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

import numpy as np

import pandas as pd

#### TODO: Number of training examples
n_train = X_train.shape[0]

#### TODO: Number of testing examples.
n_test = X_test.shape[0]

#### TODO: What's the shape of an traffic sign image?
image_shape = np.array([X_train.shape[1], X_train.shape[2]])

#### TODO: How many unique classes/labels there are in the dataset.
n_total = np.concatenate((y_train, y_valid, y_test))

n_classes = np.unique(n_total)

print("Number of training examples =", n_train)

print("Number of testing examples =", n_test)

print("Image data shape =", image_shape)

print("Number of classes =", len(n_classes))

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is [32 32]
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. I randomly dispklay 10 images in the training data set and show the histogram of the count of each sign in the training data set

![alt text][image9]
![alt text][image10]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because data in the RGB format contains three layers which takes longer time to train. In order to have a good prediction performance, I normalize the input training data to be in the range of [0,1]  and standardize the input training data  to have a mean of zero and a standard deviation of one the input training data.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image11]

Changing the input data to gray scale is able to reduce the training time, however, the validation accuracy is very low (after 10 epoch, the validation accuracy is around 50%) by using the gray scale images to train my network. Therefore, I decide to use the input data in the original RGB format to train my netowrk.  

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the first code cell of the IPython notebook. 

The code loads the training and validation sets from different files.

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.


#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the fourth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x6    	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5x1	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Fully connected		| input 400, output 120							|
| RELU					|												|
| Fully connected		| input 120, output 84							|
| RELU					|												|
| Fully connected		| input 84, output 10							|

 


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the fifth cell of the ipython notebook. 

To train the model, I choose an epoch number 25 and the batch size to be 512. When I tried to use the epoch number 10 and batch size 256 to train the network, the validation accuracy is very low (around 50%) after 10 epochs of training . Therefore, I increased both the epoch number and batch size. For the Lenet5 network to train the MNIST dataset, the hyperparameters mu and sigma are 0 and 0.1. I found those parameters do not work well for the traffic sign dataset, because after the 25 epoch training, the validation accuracy ends up to around 70%. Therefore, I reduce the sigma to be 0.05. After that, the performance boosts a lot.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

* What architecture was chosen?
I choose the LeNet to train the traffic sign data set, because it is the first Convolutional neural network architecture I learn so far. 
* Why did you believe it would be relevant to the traffic sign application?
I play the LeNet with the MNIST dataset and it is able to achieve very high performance. 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
My final model results were:
* validation set accuracy of 92%
* test set accuracy of 91.4%
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the nineth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead only      		| Ahead only   									| 
| No passing     		| No passing 									|
| Stop					| Yield											|
|Speed limit (30km/h)	| Speed limit (30km/h)			 				|
| Turn right ahead		| Turn right ahead     							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100.00%      			| Ahead only  									| 
| 100.00%  				| No passing									|
| 99.75%				| Speed limit (30km/h)							|
| < 0.03%      			| Stop			 			 	                |
| 100.00% 			    | Turn right ahead      						|


