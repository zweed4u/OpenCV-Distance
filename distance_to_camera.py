#!/usr/bin/python3
# import the necessary packages
import numpy as np
import cv2
 
def find_marker(image):
    # convert the image to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)
    resize_edged = cv2.resize(edged, (500, 640))                    # Resize image
    cv2.imshow('edged', resize_edged)
    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our rectangle of paper in the image
    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key = cv2.contourArea)
 
    # compute the bounding box of the of the paper region and return it
    return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth

# initialize the known distance from the camera to the object, which
# in this case is 12 inches
KNOWN_DISTANCE = 12.0
 
# initialize the known object width, which in this case, the rectangle of
# paper is 5 inches wide
KNOWN_WIDTH = 5.0
 
# initialize the list of images that we'll be using
IMAGE_PATHS = ["new/1ft.JPG", "new/2ft.JPG", "new/3ft.JPG"]
 
# load the furst image that contains an object that is KNOWN TO BE 1 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
image = cv2.imread(IMAGE_PATHS[0])
marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

# loop over the images
for imagePath in IMAGE_PATHS:
    # load the image, find the marker in the image, then compute the
    # distance to the marker from the camera
    image = cv2.imread(imagePath)
    marker = find_marker(image)
    inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
 
    # draw a bounding box around the image and display it
    box = np.int0(cv2.boxPoints(marker))
    cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
    cv2.putText(image, "%.2fft" % (inches / 12),
        (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 255, 0), 3)
    resized_image = cv2.resize(image, (500, 640))                    # Resize image
    cv2.imshow("image", resized_image)
    cv2.waitKey(0)
