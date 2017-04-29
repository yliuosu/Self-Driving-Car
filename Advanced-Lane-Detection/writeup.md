---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/chessboard_original_undistored.JPG "Original and Undistorted"

[image2]: ./output_images/roadimage_original_undistored1.JPG "Original and Undistorted"
[image3]: ./output_images/roadimage_original_undistored2.JPG "Original and Undistorted"
[image4]: ./output_images/roadimage_original_undistored3.JPG "Original and Undistorted"

[image5]: ./output_images/pipeline_original.JPG "Image Process Pipeline"
[image6]: ./output_images/pipeline_IHLselection.JPG "Image Process Pipeline"
[image7]: ./output_images/pipeline_gradientselection.JPG "Image Process Pipeline"
[image8]: ./output_images/pipeline_combination.JPG "Image Process Pipeline"
[image9]: ./output_images/pipeline_areaselection.JPG "Image Process Pipeline"

[image10]: ./output_images/perspecitve_transform.JPG "Image Process Pipeline"
[image11]: ./output_images/polynomial_fitted_lane_lines.JPG "Image Process Pipeline"
[image12]: ./output_images/RadiusofCurvature.JPG "Image Process Pipeline"
[image13]: ./output_images/detectedresults.JPG "Image Process Pipeline"

[image14]: ./output_images/badexample1.JPG "Image Process Pipeline"
[image15]: ./output_images/badexample2.JPG "Image Process Pipeline"
[image16]: ./output_images/badexample3.JPG "Image Process Pipeline"
[image17]: ./output_images/badexample4.JPG "Image Process Pipeline"

[video1]: ./output_images/project_result.mp4 "Video"



## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is from code line 9 to 43. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function and use pickle to save the camera calibration results for later use.  I applied this distortion correction to the one of the chessboard image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
After I generated the coefficients, I applied the distortion correction to some road testing images by using `cv2.undistort()` function and obtained this result:
![alt text][image2]
![alt text][image3]
![alt text][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #146 through #159 in `AdvanceLaneDetection.py`).  Here's an example of my output for this step.  

                                                         Original Image

![alt text][image5]

                                               Thresholded Image by HLS Selection
                                                
![alt text][image6]

                                              Thresholded Image by Gradient Selection
![alt text][image7]  

                                              Combination of the two Results Above
                                                
![alt text][image8]                           
                                                
                                             Select the Sub-Area Most Likely the Lane Located

![alt text][image9] 
                                                    
#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `bird_eye()`, which appears in lines 343 in the file `AdvanceLaneDetection.py`. The `bird_eye()` function takes as inputs an image (`img`), as well as source (`src_pts`) and destination (`dst_pts`) points.  I chose the hardcode the source and destination points in the following manner:

```
# define 4 source points 
src_pts=np.array([[270, img_size[1]], [450, 575], [925, 575], [1180, img_size[1]]],np.int32)
# define 4 destination points 
dst_pts=np.array([[285, img_size[1]],[285, 0.85* img_size[1]],[995,0.85*img_size[1]],[995,img_size[1]]], np.int32)
```
The index of the points are chosen after a lot of experiments which is able to provide good results. 

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image10]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I use the equation that f(y) = A*y^2 + B*y + C and the detected lane line points to fit a second order polynomial curve:  in lines 357 through 358 in my code in `AdvanceLaneDetection.py`. Here are the fitted ploynomial curve for the left and right lane.

![alt text][image11]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I use the following equation to calculate the curvautre of the lane.
![alt text][image12] 

I use the center of the detected left lane line and right lane line respect to the center of the image to determine the position of the vehicle with respect to center.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #360 through #382 in my code in `AdvanceLaneDetection.py` in the function `lane_detection()`.  Here is an example of my result on a test image:

![alt text][image13]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][video1]:

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

By implementing this project, I learn how to do the camera calibration to remove the distortion caused by the cameras lens before retrieve information from the captured image. 

Compared to lane detection implemented by the Hough transform, HLS selection and multiple gradient selection methods is able to achieve more stable and better detection resutls.

After that I am able to represent the parallel lane lines by using the perspecitve transform. Based on the lane line points detected under the bird's eye veiw, I am able to fit the lane line by using a second order polynomial curve and calculate the center of the lane and the raidus of the radius of the lane curvature. 

In the end, I use the perspecitve inverse transform to transform the bird's eye view back to the front camera view and label out the detected the lane area.

In this project, I found that sometimes when the surface of the pavement is very bright, it will introduce a lot of noise after using the HLS selection method. In the end, the noise will make the lane line fitting algorithm failed and make the lane area detection failed like below.


                                       Noise Introduce After the HLS Selection On Bright Pavement
![alt text][image14]

                                         Gradient Selection Results and the Combined Results
![alt text][image15]

                                                    SubArea Selection Results

![alt text][image16]

                                                     Final Detection Results
![alt text][image17]        

Solution to this problem, I use the griadent select method to process the results from the HLS selection method, it is able to reduce the noise to make the pipeline function correctly.
