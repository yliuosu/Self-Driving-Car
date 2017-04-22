import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

# prepare object points
objp = np.zeros((6*9, 3), np.float32)
# print(objp)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
# print(objp)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# read a list of calibration images
cal_images = glob.glob('camera_cal/calibration*.jpg')
#print(len(cal_images))

# Step through the list and search for chessboard corners
img_size = []
for idx, fname in enumerate(cal_images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = (img.shape[1], img.shape[0])
	
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points
    if ret == True:
        print('working on', fname)
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the detected corners
        cv2.drawChessboardCorners(img, (9,6), corners, ret)
		# save the corners detected chessboard image
        #write_name = 'camera_cal/corners_found'+str(idx)+'.jpg'
        #cv2.imwrite(write_name, img)

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

# Save the camera calibration result for later use
#dist_pickle = {}
#dist_pickle["mtx"] = mtx
#dist_pickle["dist"] = dist
#pickle.dump( dist_pickle, open( "camera_cal/wide_dist_pickle.p", "wb" ) )

# read in the saved camera calibration parameters
'''
def load_cal_parameters(fname):
    #dist_pickle = pickle.load( open( "camera_cal/wide_dist_pickle.p", "rb" ) )
    dist_pickle = pickle.load( open( fname, "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    return mtx, dist

mtx, dist = load_cal_parameters('camera_cal/wide_dist_pickle.p')
'''

# Test undistortion on a corners detected chessboard image
img = cv2.imread('camera_cal/corners_found1.jpg')
# img_size = (img.shape[1], img.shape[0])

dst = cv2.undistort(img, mtx, dist, None, mtx)
#cv2.imwrite('camera_cal/test_undist1.jpg',dst)

# Visualize undistortion
dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)

# function to perform the camera calibration, image distortion correction and 
# returns the undistorted image
def cal_undistort(img, mtx, dist):
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

# Make a list of test images
test_images = glob.glob('test_images/*.jpg')

# Step through the testing list and display the original and the un-distorted image
for idx, fname in enumerate(test_images):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_size = (img.shape[1], img.shape[0])
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    undis_img = cal_undistort(img, mtx , dist)
    ax2.imshow(undis_img)
    ax2.set_title('Undistorted Image', fontsize=30)

# function that takes an image to threshold it according to the min / max gradient values
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh_min=0, thresh_max=255):
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # use inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return binary_output

# function that takes an image to threshold it according to the min / max magnitude of the gradient
def mag_thresh(img, sobel_kernel=3, thresh_min=0, thresh_max=255):
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh_min) & (gradmag <= thresh_max)] = 1
    return binary_output

# function that takes an image to threshold it according to the min / max direction of the gradient
def dir_threshold(img, sobel_kernel=3, thresh_min=0, thresh_max=np.pi/2):
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh_min) & (absgraddir <= thresh_max)] = 1
    # Return the binary image
    return binary_output

# define a gradient selection function by using multiple different gradient threshold functions
def gradient_select(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel_size = 9
    gradient_x = abs_sobel_thresh(gray, orient='x', sobel_kernel=kernel_size, thresh_min=15,  thresh_max = 255)
    #gradient_y = abs_sobel_thresh(gray, orient='y', sobel_kernel=kernel_size, thresh_min=15, thresh_max = 255)
    gradient_mag = mag_thresh(gray, sobel_kernel=kernel_size, thresh_min=60,thresh_max = 255)
    gradient_direction = dir_threshold(gray, sobel_kernel=kernel_size, thresh_min=0.12*np.pi/2, thresh_max = 0.80*np.pi/2)
    edges = np.zeros_like(gray)
	# not use the grad threshold from the y direction because most of time lane lines 
	# have good detection results from the x direction
    # edges[((gradient_x == 1) & (gradient_y == 1) & (gradient_direction == 1) & (gradient_mag == 1))] = 1
    edges[((gradient_x == 1)& (gradient_mag == 1))& (gradient_direction == 1)] = 1
    return edges
	
# define a function that thresholds the S-channel of HLS
def hls_select(img, thresh_min=0, thresh_max=255):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh_min) & (s_channel <= thresh_max)] = 1
    return binary_output

# select the subarea where lane lines are most likely located in
def area_select(img):
    img_size = (img.shape[1], img.shape[0])
    #print(img.shape[1])
	
	# define the 4 corners of the subarea
    left_bottom = [0.0*img_size[0], img_size[1]-1]
    left_top = [0.45*img_size[0], 0.6*img_size[1]]
    right_top = [0.55*img_size[0], 0.6*img_size[1]]
    right_bottom = [1*img_size[0], img_size[1]-1]
    
	# define the 4 boundaries of the subarea
    left_boundary = np.polyfit((left_bottom[0], left_top[0]), (left_bottom[1], left_top[1]), 1)
    top_boundary = np.polyfit((left_top[0], right_top[0]), (left_top[1], right_top[1]), 1)
    right_boundary = np.polyfit((right_top[0], right_bottom[0]), (right_top[1], right_bottom[1]), 1)
    bottom_boundary = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)
    
    # select the region inside the boundaries
    XX, YY = np.meshgrid(np.arange(0, img_size[0]), np.arange(0, img_size[1]))
    region_thresholds = (YY > (XX*left_boundary[0] + left_boundary[1])) & \
                        (YY > (XX*top_boundary[0] + top_boundary[1])) & \
                        (YY > (XX*right_boundary[0] + right_boundary[1])) & \
                        (YY < (XX*bottom_boundary[0] + bottom_boundary[1]))
                
    lane_area =np.zeros_like(img)
    lane_area[region_thresholds & (img==1)]=1
	
    return lane_area 

# doing the perspective transform to generate the birds eye view
def bird_eye(img, src_pts, dst_pts):
    warp_matrix = cv2.getPerspectiveTransform(src_pts,dst_pts)
    warped=cv2.warpPerspective(img, warp_matrix, img.shape[::-1],flags=cv2.INTER_LINEAR)
    return warped, warp_matrix
	
# Find the boundary of the lane lines
# wdw_dx: the width of the windows +/- margin
# wdw_dy: the height of the windows +/- margin, using 720/80 = 9 sliding windows 
def find_lane(img, x_init_pos, wdw_dx=120, wdw_dy= 80):
    img_size = (img.shape[1], img.shape[0])
    windowed=np.zeros_like(img)
    # start the search from peak
    x_mean=x_init_pos
    for y in range(img_size[1], 0, -wdw_dy):
        wdw=img[y-wdw_dy:y,x_mean-wdw_dx:x_mean+wdw_dx].copy()
        windowed[y-wdw_dy:y,x_mean-wdw_dx:x_mean+wdw_dx]=wdw.copy()
        # Calculate new mean of x
        histogram = np.sum(wdw, axis=0)
        if (np.sum(histogram)!=0): # found pixels, update the mean location        
            x_mean=x_mean-wdw_dx+np.argmax(histogram)
            # print(histogram)
            # print(x_init_pos, np.argmax(histogram), x_mean) 
    return windowed

# Detect the left and right lane pixels
def find_lane_boundary(img, left_start_point, right_start_point):
    img_size = (img.shape[1], img.shape[0])
    x_center=int(img_size[0]/2)
    # mask the bottom centre area 
    img[img_size[1]-100:img_size[1], x_center-150:x_center+150]=0        
    left_lane=find_lane(img, left_start_point)
    right_lane=find_lane(img, right_start_point)
    return left_lane, right_lane 
	
# used their x and y pixel positions to fit a second order polynomial curve
# used lane pixel positions to fit a second order polynomial curve
def fit_polynomial(lane_img):
    img_size = (img.shape[1], img.shape[0])
    y = np.linspace(0, img_size[1], num=img_size[1])
    # Get the x,y indexes with non-zero elements i.e. white lines
    ally,allx=np.nonzero(lane_img)
    # Polynomial fitting
    fit=np.polyfit(ally,allx,2)
    recent_xfitted = fit[0]*y**2 + fit[1]*y + fit[2]
    y_bottom=img_size[1]-1
    line_base_pos = fit[0]*y_bottom**2 + fit[1]*y_bottom + fit[2]
    '''
    print("polynomial", fit)
    print("base pos ", line_base_pos)
    # Plt on graph
    plt.plot(allx, ally, '.', color='white')
    plt.plot(recent_xfitted, y, color='red', linewidth=3)
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    ax = plt.gca()
    ax.set_axis_bgcolor('black')
    plt.gca().invert_yaxis()
    plt.show()
    '''
    return allx, ally, fit, line_base_pos

    



