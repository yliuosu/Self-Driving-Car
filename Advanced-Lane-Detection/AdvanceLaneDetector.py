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

    



