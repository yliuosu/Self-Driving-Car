# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/autonomous_driving_1.JPG "simulator in autonomous mode"
[image2]: ./examples/autonomous_driving_2.JPG "simulator in autonomous mode"
[image3]: ./examples/autonomous_driving_3.JPG "simulator in autonomous mode"
[image4]: ./examples/autonomous_driving_4.JPG "simulator in autonomous mode"
[image5]: ./examples/autonomous_driving_5.JPG "simulator in autonomous mode"

[image6]: ./examples/staycenter_1.JPG "stay in the center"
[image7]: ./examples/staycenter_2.JPG "stay in the center"
[image8]: ./examples/staycenter_3.JPG "stay in the center"
[image9]: ./examples/staycenter_4.JPG "stay in the center"

[image10]: ./examples/lefttocenter_1.JPG "recover from left side to center"
[image11]: ./examples/lefttocenter_2.JPG "recover from left side to center"
[image12]: ./examples/lefttocenter_3.JPG "recover from left side to center"
[image13]: ./examples/lefttocenter_4.JPG "recover from left side to center"

[image14]: ./examples/righttocenter_1.JPG "recover from left side to center"
[image15]: ./examples/righttocenter_2.JPG "recover from left side to center"
[image16]: ./examples/righttocenter_3.JPG "recover from left side to center"
[image17]: ./examples/righttocenter_4.JPG "recover from left side to center"

[image18]: ./examples/original_flipped.JPG "original image and flipped image"

[image19]: ./examples/model.png "Model Architecture"

[image20]: ./examples/loss_epoch.JPG "Loss vs Epochs"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* a video shows the autonomous driving

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

The following pictures show some screenshots of the simulator in autonomous mode

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I tried three models to train the input data. The first model is just a signle fully connected network  and the second model is learned from the Lenet 5 network. After training those models on the input images and steering angles, the autonomous vehcile is only able to keep in the drivable portion of the track surface for a short of distance before it drives across the curb.


Then I learn the model used in [NVIDIA’s Self-Driving Car](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) and implement it.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I used a combination of images from the left camera, the right camera and the center camera. I record the driving data when the vehicle is driving in the center of the road for most of time and I also record the driving data about the recovering process when the vehicle is driving from the left and right sides of the road to the center of the road. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture is to extract features from the road images and output steering angles which keep the vehicle driving in the center of road as much time as possible. The model should be able to extract features from images, therefore convolution neural network is a good choice.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set so that I am able to know whether the model is overfitting (a low mean squared error on the training set but a high mean squared error on the validation set) or underfitting (a high mean squared error on the training set and a high mean  squared error on the validation set).

To combat the overfitting, I add two dropout layers in the first and second fully connected layers.

After several trainings, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

My goal is to train the model to minimize the mean-squared error between the predicted steering output and the steering command from the human.

The final model architecture is illustrated below which consists of 16 layers, including a normalization layer, a cropping layer, 5 convolutional layers, 2 dropout layers, 3 fully connected layers and 1 output layer. The input images are trained in the RGB format.

![alt text][image19] 


The first layer of the network performs the normalization operation which is implemented in the lambad layer of keras. The normalization is able to accelerate the training process.

The second layer of the network performs the cropping operation which is aim to remove the distraction area of the input image so that the network is able to focus on the meaning input features.

The convolutional layers are used to perform feature extraction. Learn from the NVIDIA’s Self-Driving Car post, 5 layers is able to provide good feature extraction. The first three convolutional layers with a stride of 2X2 and a kernel of 5X5 is to extract some basic features such as lines and shapes. The last two convolutional layers with a non-strided and a kernel size of 3X3 is to combine the basic features from the previous layers to infere more complicated features.

Three fully connected layers are followed with the five convolutional layers. The fully connected lyaers are used to predict the controller output based on the features from the convolutional layers.

Two dropout layers are added between the fully connected layers to reduce the chance of overfitting.



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image6] ![alt text][image7] ![alt text][image8] ![alt text][image9]

                                              Driving in the center of the road


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to drive back to the center of the track from road edges. These images show what a recovery looks like starting from left edges to center :

![alt text][image10] ![alt text][image11] ![alt text][image12] ![alt text][image13]

                                             Recoverying from the left side edges


![alt text][image14] ![alt text][image15] ![alt text][image16] ![alt text][image17]

                                             Recoverying from the right side edges

To augment the data sat, I also flipped images and angles thinking that this would help the model to learn the multiple features of the track rather than overfit to one special case.  For example, here is an image that has then been flipped:

![alt text][image18]


After the collection process, I had around 4000 number of data points. I then preprocessed this data by the augmentation mentioned above.

Before training the image data, I normalize them to be in the range of [-0.5,0.5].

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I try to train the network in 10 epochs. However, it is better to use the parameters trained from epoch 5 based on the loss and epochs figure illustrated below. Because epoch 5 gets the lowest validation loss.

![alt text][image20]

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.
