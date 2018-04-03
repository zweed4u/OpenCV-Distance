#!/usr/bin/python3
# import the necessary packages
import numpy as np
import cv2
import imutils

class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
 
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
 
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
 
        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"
 
        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"
 
        # return the name of the shape
        return shape

def find_marker(image):
    # convert the image to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 20, 35)
    resize_edged = cv2.resize(edged, (400, 540))                    # Resize image
    cv2.imshow('edged', resize_edged)

    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our rectangle of paper in the image
    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)

    # load the image and resize it to a smaller factor so that
    # the shapes can be approximated better
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])
    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
     
    # find contours in the thresholded image and initialize the
    # shape detector
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    sd = ShapeDetector()

    # loop over the contours
    for c in cnts:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        shape = sd.detect(c)
        if shape != 'rectangle':
            return -1
        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
     
        # show the output image
        cv2.imshow("ShapeDetector", image)
 
    # compute the bounding box of the of the paper region and return it
    return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth


def bgr_to_hsv(b, g, r):
    return cv2.cvtColor(np.uint8([[[b,g,r]]]),cv2.COLOR_BGR2HSV)[0][0]


# initialize the known distance from the camera to the object, which
# in this case is 12 inches
KNOWN_DISTANCE = 12.0
 
# initialize the known object width, which in this case, the rectangle of
# paper is 5 inches wide
KNOWN_WIDTH = 5.0
 
 
# load the furst image that contains an object that is KNOWN TO BE 1 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
image = cv2.imread("new/2018-04-02-165955.jpg")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# HSV
## mask of green (36,0,0) ~ (70, 255,255)
yellow_hsv_array = bgr_to_hsv(94,73,135) # bgr value pulled from picture of petals
print(bgr_to_hsv(94,73,135)) # LIGHTER PINK HSV = 170 117 135
print(bgr_to_hsv(133,113,190)) # DARKER PINK HSV = 172 103 190
print(bgr_to_hsv(142,122,187)) # LIGHTER PINK HSV
print(bgr_to_hsv(126,103,158)) # LIGHTER PINK HSV

mask1 = cv2.inRange(hsv, (160, 75, 100), (180, 130, 200))
#cv2.imshow('mask', mask1)
## final mask and masked
target = cv2.bitwise_and(image,image, mask=mask1)
#cv2.imshow('mask &', target)
# TODO filter/mask should go here
marker = find_marker(target)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH


cap = cv2.VideoCapture(0)
while 1:
    _, frame = cap.read()

    # Close all on 'q' press
    k = cv2.waitKey(5) & 0xFF
    if k == ord('q'):
        break

    # convert the frame to hsv and create mask in attempt to isolate the pink square 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #mask1 = cv2.inRange(hsv, (int(hsv_array[0]-10), 0, 0), (int(hsv_array[0]+10), 255,255))
    mask1 = cv2.inRange(hsv, (160, 75, 100), (180, 130, 200))
    target = cv2.bitwise_and(frame,frame, mask=mask1)

    try:
        # Edge detect on the and'ed mask
        marker = find_marker(target)
        if marker == -1:
            continue
    except:
        continue

    inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
    print(f'{inches / 12}ft')
    box = np.int0(cv2.boxPoints(marker))
    cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
    cv2.putText(frame, "%.2fft" % (inches / 12), (frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    resized_image = cv2.resize(frame, (400, 540))                    # Resize image
    cv2.imshow("image", frame)

cv2.destroyAllWindows()
cap.release()
