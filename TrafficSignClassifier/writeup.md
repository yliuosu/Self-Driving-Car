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

[image12]: ./testimages/Histogram_Equalization.JPG "histogram equalizer"
[image13]: ./testimages/ImageSharpener.JPG "Image sharpener"


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

Here is an exploratory visualization of the data set. I randomly dispklay 10 images in the training data set and show the histogram of the training data set (blue bar) and the validation data set (orange bar).

![alt text][image9]
![alt text][image10]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because data in the RGB format contains three layers which takes longer time to train. In order to have a good prediction performance, I normalize the input training data to be in the range of [0,1]  and standardize the input training data  to have a mean of zero and a standard deviation of one the input training data.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image11]

Changing the input data to gray scale is able to reduce the training time, however, the validation accuracy is very low (after 10 epoch, the validation accuracy is around 50%) by using the gray scale images to train my network. Therefore, I decide to use the original RGB data format to train my netowrk. Different traffic signs may have different colors. Therefore, color can be used as a feature to distiguish different traffic signs.

Learn from the suggetions of the first reviewer, I add two preprocess functions: imagequalizer and imagesharpener. The results of the image equalizer is displayed below. The image displayed on the left is the original image. The image displayed on the right is the processed result.

![alt text][image12]

Because some traffic sign image are very dark. After processed by the imageequalizer, the brightness of the interested area of the traffic sign improves a lot. For the final test data set, if I use my trained network to classify the original image directly. The classification accuracy of my network is 85.9%. If I preprocess the test data set by the imageequalizer, the classification accuracy of my network boosts to 94.8% . Therefore, brightness of the image affects the classification results a lot.


The results of the image sharpener is displayed below. The image displayed on the left is the original image. The image displayed in the middle is the result processed by the image equalizer. The image displayed on the right is the result first processed by the image equalizer then processed by the image sharpener. 

![alt text][image13]


Although the image sharpener is able to enhace the edges in the image, it introduces some bright noise in the image also. If I process the test data set first by the image equalizer then process it by the image sharpener, the classification accuracy of my network on the test data set drops from 85.9% (without any preprocess) to 78.9%. Therefore, only use image equalizer to preprocess image.


#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the first code cell of the IPython notebook. 

The code loads the training and validation sets from different files.

My final training set had 34799 number of images. My validation set and test set had 4410 and 12630 number of images.


#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the fourth cell of the ipython notebook. 

My first submitted model consisted of the following layers:

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
| Fully connected		| input 84, output 43							|

Learn from the suggetions of the first reviewer, I add the batch normalization and dropout layers to my model and I also increase the layer of kernels in the first two convolution layers. My first submitted model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5x6    	| 1x1 stride, valid padding, outputs 28x28x32 	|
| Batch Normalization	|												|
| RELU					|												|
| Dropout	            |												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32				    |
| Convolution 5x5x1	    | 1x1 stride, valid padding, outputs 10x10x64	|
| Batch Normalization	|												|
| RELU					|												|
| Dropout	            |												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				    |
| Fully connected		| input 1600, output 768						|
| Batch Normalization	|												|
| RELU					|												|
| Dropout	            |												|
| Fully connected		| input 768, output 256							|
| Batch Normalization	|												|
| RELU					|												|
| Dropout	            |												|
| Fully connected		| input 256, output 43							|
 
I also tried to use the Tensorboard to log more training information for later analysis. However, the code is not compiled sucessfully. I will keep work on it.


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

My final model results for the first submission were:
* maximum validation accuracy of 92%
* test set accuracy of 91.4%

After I revised my model, the final results are:
* maximum validation accuracy of 96.6%, minimum validation accuracy of 92.8%
* test set accuracy of 94.8%
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

I think the stop sign and the speed limit (30km/h) sign are difficult to classify. First, the stop sign image is clipped not very well. It didn't reflect the normal shape of the sign that may lead to negtive results. The speed limit (30km/h) sign is not easy to classify due to the abnormal view angle of image. For the turn right ahread sign, part of its interested region is not showed in the image which may also lead to negtive results. For the ahead only and no passing signs, images are a littble bit blur which increase the difficulty to classify them correctly.   


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the nineth cell of the Ipython notebook.

Here are the results of the prediction for the first submission:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead only      		| Ahead only   									| 
| No passing     		| No passing 									|
| Stop					| Yield											|
|Speed limit (30km/h)	| Speed limit (30km/h)			 				|
| Turn right ahead		| Turn right ahead     							|

After revise the model and tune the training parameters, the below is the new results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Ahead only      		| Ahead only   									| 
| No passing     		| No passing 									|
| Stop					| Stop											|
|Speed limit (30km/h)	| Speed limit (30km/h)			 				|
| Turn right ahead		| Turn right ahead     							|


The model was able to correctly classify 4 of the 5 traffic signs, which gives an accuracy of 80%.

The new model is able to correctly classify all of 5 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

![alt text][image4]
- Correct Sign Name: **Ahead Only**, Top 5 probabilites are:
- Sign Name: Ahead only, Sign Id:35, Probability: 96.88%
- Sign Name: Speed limit (80km/h), Sign Id: 5, Probability: 2.66%
- Sign Name: Keep right, Sign Id:38, Probability: 0.21%
- Sign Name: Turn left ahead, Sign Id:34, Probability: 0.05%
- Sign Name: No passing for vehicles over 3.5 metric tons, Sign Id:10, Probability: 0.03%


![alt text][image5]
- Correct Sign Name: **No Passing**, Top 5 probabilites are:
- Sign Name: No passing, Sign Id: 9, Probability: 99.94%
- Sign Name: Speed limit (60km/h), Sign Id: 3, Probability: 0.03%
- Sign Name: Slippery road, Sign Id:23, Probability: 0.01%
- Sign Name: End of no passing, Sign Id:41, Probability: 0.01%
- Sign Name: Dangerous curve to the right, Sign Id:20, Probability: 0.01%

![alt text][image6]
- Correct Sign Name: **Speed limit (30km/h)**, Top 5 probabilites are:
- Sign Name: Speed limit (30km/h), Sign Id: 1, Probability: 95.62%
- Sign Name: Speed limit (50km/h), Sign Id: 2, Probability: 3.47%
- Sign Name: Priority road, Sign Id:12, Probability: 0.25%
- Sign Name: End of speed limit (80km/h), Sign Id: 6, Probability: 0.19%
- Sign Name: Speed limit (80km/h), Sign Id: 5, Probability: 0.17%

![alt text][image7]
- Correct Sign Name: **Stop**, Top 5 probabilites are:
- Sign Name: Stop, Sign Id:14, Probability: 85.76%
- Sign Name: General caution, Sign Id:18, Probability: 9.52%
- Sign Name: No vehicles, Sign Id:15, Probability: 2.82%
- Sign Name: Traffic signals, Sign Id:26, Probability: 0.47%
- Sign Name: Bicycles crossing, Sign Id:29, Probability: 0.27%

![alt text][image8]
- Correct Sign Name: **Turn right ahead**, Top 5 probabilites are:
- Sign Name: Turn right ahead, Sign Id:33, Probability: 86.44%
- Sign Name: Right-of-way at the next intersection, Sign Id:11, Probability: 4.33%
- Sign Name: Speed limit (30km/h), Sign Id: 1, Probability: 4.10%
- Sign Name: Roundabout mandatory, Sign Id:40, Probability: 2.44%
- Sign Name: No entry, Sign Id:17, Probability: 1.36%
