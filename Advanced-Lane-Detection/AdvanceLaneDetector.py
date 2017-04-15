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


    



