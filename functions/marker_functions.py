#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Data and information that support the findings of the article:
# M. Pieber and Z. Zhang and P. Manzl and J. Gerstmayr,  
# Surrogate models for compliant joints in programmable structures
# Proceedings of the ASME 2024
#
#
# Author:   M. Pieber, E. Ulrich
# Date:     2024-03-11
# Testet with exudyn version: 1.8.0
#
# Copyright:This file is part of Exudyn. Exudyn is free software. You can 
#           redistribute it and/or modify it under the terms of the Exudyn 
#           license. See 'LICENSE.txt' for more details.
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


import numpy as np
import logging
import config
import cv2
import imutils

config.init()

def SortPoints2D(Points,boundary = 10):
    # function to Sort Point 2-Dimensional
    for j in range(1,len(Points[:,1])):
        for i in range(int((len(Points[1])))):
            if abs(Points[j-1,i,0]-Points[j,i,0])>=boundary or abs(Points[j-1,i,1]-Points[j,i,1])>=boundary:
                for k in range(int(len(Points[1]))):
                    if abs(Points[j-1,i,0]-Points[j,k,0])<=boundary and abs(Points[j-1,i,1]-Points[j,k,1])<=boundary:
                        # switching Positions of measured Points
                        Points[j,i],Points[j,k]=Points[j,k].copy(),Points[j,i].copy()
    return Points

def brightnessContrastCorrection(img, brightness=255, contrast=127):
    """
    Function to edit the brightness and Contrast of an Image. Contrast goes from 0 to 255, while 127 means no change. Brightness goes from 0 to 512, while 255 means no change. 
    
    :param img: Image file
    :type img: opencv Image

    :param brightness: brightness value betweeen 0 and 512
    :type brightness: int

    :param contrast: contrast value betweeen 0 and 255
    :type contrast: int

    :return: changed Image
    :rtype: Opencv Image
    """
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))
    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))
    if brightness != 0:
        if brightness > 0:  
            shadow = brightness  
            max = 255  
        else:  
            shadow = 0
            max = 255 + brightness  
        al_pha = (max - shadow) / 255
        ga_mma = shadow
  
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(img, al_pha, 
                              img, 0, ga_mma)
  
    else:
        cal = img
  
    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)
  
        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(cal, Alpha, 
                              cal, 0, Gamma)
  
    # putText renders the specified text string in the image.
    #cv2.putText(cal, 'B:{},C:{}'.format(brightness,
    #                                    contrast), (10, 30),
    #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
  
    return cal

def sort_coordinates(coord_array):
    """
    Function to Sort Coordinates

    :param coord_array: 
    :type coord_array: np.array

    :return: Sorted Coordinates
    :rtype: np.array
    """
    coord_sorted = []
    if (abs(coord_array[0][0]+coord_array[0][1]) < abs(coord_array[1][0]+coord_array[1][1])):
        coord_sorted.append(coord_array[0])
        coord_sorted.append(coord_array[1])
    else:
        coord_sorted.append(coord_array[1])
        coord_sorted.append(coord_array[0])
    if (abs(coord_array[2][0]+coord_array[2][1]) < abs(coord_array[3][0]+coord_array[3][1])):
        coord_sorted.append(coord_array[3])
        coord_sorted.append(coord_array[2])
    else:
        coord_sorted.append(coord_array[2])
        coord_sorted.append(coord_array[3])
    return coord_sorted

def order_points(pts):
    """
    initialzie a list of coordinates that will be ordered
    such that the first entry in the list is the top-left,
    the second entry is the top-right, the third is the
    bottom-right, and the fourth is the bottom-left
    
    :param pts: list of coordinates
    :type pts: np.array

    :return: ordered Coordinates
    :rtype: np.array
    """
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image,ptsOrg):
    """
    Transform an Image by seting four EdgePoints

    :param image: Image to be transformed
    :type image: opencv Image

    :param ptsOrg: four Edge Points
    :type ptsOrg: np.array

    :return: warped Image and distance between the edgePoints in Pixel
    :rtype: (OpenCV Image, lst)
    """
    pts = ptsOrg[:,0:2] #only X and Y Components
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    #width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    #height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
            [0, 0],
            [maxWidth -1,0],
            [maxWidth - 1, maxHeight -1],
            [0, maxHeight -1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped,dst

def GetMarker(orgImage,lowerLimit=config.measurement.markerColorLower ,upperLimit=config.measurement.markerColorUpper ,pixelRadius=None,detectionMode = 'Area'):
    """
    Get the Centerpoint of Markers in Image by Color detection

    :param orgImage: Image containing the Markers
    :type orgImage: Opencv Image

    :param lowerLimit: lower HSV Color of Marker
    :type lowerLimit: np.array

    :param upperLimit: upper HSV Color of Marker
    :type upperLimit: np.array

    :param pixelRadius: Radius of Marker in Pixel
    :type pixelRadius: int

    :return: Coordinates of Markers
    :rtype: np.array
    """
    image = orgImage.copy()

    if pixelRadius != None:
        minArea = (pixelRadius-config.measurement.radiusRange)**2*np.pi
        maxArea = (pixelRadius+config.measurement.radiusRange)**2*np.pi
    else:
        minArea = 30
        maxArea = 1000000
        pixelRadius = 5

    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(pixelRadius/2),int(pixelRadius/2)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #hsv = cv2.morphologyEx(hsv, cv2.MORPH_CLOSE, kernel, iterations=2)
    #hsv = cv2.morphologyEx(hsv, cv2.MORPH_OPEN, kernel, iterations=2)
    
    mask = cv2.inRange(hsv, lowerLimit, upperLimit)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cord = []
    for contour in cnts:
        
        if detectionMode == 'Area':
            area = cv2.contourArea(contour)
            if(area >= minArea) and (area <= maxArea):
                M = cv2.moments(contour)
                if (M["m00"] == 0):
                    pass
                else:
                    cX = M["m10"] / M["m00"]
                    cY = M["m01"] / M["m00"]
                    cZ = 0
                    cord.append([cX,cY,cZ])
                    #nur zum einstellen nötig kann wieder rausgemacht werden später
                    if config.output.plot:
                        image = cv2.circle(image, (int(cX),int(cY)),pixelRadius,(0,0,255),2)
                        image = cv2.circle(image, (int(cX),int(cY)), radius=1, color=(0, 0, 255), thickness=2)
        elif detectionMode == 'minCircle':
            (cX, cY), radius = cv2.minEnclosingCircle(contour)
            cZ = 0
            if (radius >= (pixelRadius-config.measurement.radiusRange)) and (radius <= (pixelRadius+config.measurement.radiusRange)):
                cord.append([cX,cY,cZ])
                if config.output.plot:
                    image = cv2.circle(image, (int(cX),int(cY)),int(radius),(0,0,255),2)
                    image = cv2.circle(image, (int(cX),int(cY)), radius=1, color=(0, 0, 255), thickness=2)
    if config.output.plot:
        scaling = 3
        #hsvS = cv2.resize(hsv, (int(config.camera.videoFormat[0]/scaling), int(config.camera.videoFormat[1]/scaling)))
        #cv2.imshow("hsv", hsvS)
        maskS = cv2.resize(mask, (int(config.camera.videoFormat[0]/scaling), int(config.camera.videoFormat[1]/scaling)))
        cv2.imshow("Mask", maskS)
        imS = cv2.resize(image, (int(config.camera.videoFormat[0]/scaling), int(config.camera.videoFormat[1]/scaling)))
        cv2.imshow("Detected Circle", imS)
        cv2.waitKey(0)
    return cord

def getSquare(image,lowerLimit,upperLimit):
    """
    Get Center Coordinates of an Square in an Image

    :param image: Image containing the Markers
    :type image: Opencv Image

    :param lowerLimit: lower HSV Color of Marker
    :type lowerLimit: np.array

    :param upperLimit: upper HSV Color of Marker
    :type upperLimit: np.array
    
    :return: Coordinates of Squares
    :rtype: np.array
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowerLimit, upperLimit)
    if config.output.plot:
        cv2.namedWindow("ohne morph", cv2.WINDOW_NORMAL)
        cv2.imshow("ohne morph",mask)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    #mask = cv2.erode(mask,kernel,iterations = 1)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    #if config.output.plot:
    #    cv2.namedWindow("morph1", cv2.WINDOW_NORMAL)
    #    cv2.imshow("morph1",mask)
    #mask  = cv2.dilate(mask,kernel,iterations = 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if config.output.plot:
        cv2.namedWindow("mitmorph", cv2.WINDOW_NORMAL)
        cv2.imshow("mitmorph",mask)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cord = []
    for contour in cnts:
        area = cv2.contourArea(contour)    
        if(area > config.measurement.minArea):

            x,y,w,h = cv2.boundingRect(contour)
            ROI = image[y:y+h, x:x+h]
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)

            cX = int(x+w/2)
            cY = int(y+h/2)
            cord.append([cX,cY])

            if config.output.plot:
                imS = cv2.resize(image, (int(config.camera.videoFormat[0]/2), int(config.camera.videoFormat[1]/2)))
                cv2.imshow("rectangles",imS)
    return cord,image,mask

def get_scale(cordlist):
    """
    Calculate the Scale in X and Y

    :param cordlist: Coordinates of Edge Points
    :type cordlist: np.array

    :return: Scale in X and Y
    :rtype:(float, float)
    """
    xscale = config.measurement.xCord/cordlist[1][0]
    yscale = config.measurement.yCord/cordlist[3][1]
    return xscale,yscale

def writte_cords_to_text(cordlist,xscale,yscale):
    """
    Function to write the Coordinates to text

    :param cordlist: List of Coordinates
    :type cordlist: np.array

    :param xscale: Scale in X
    :type xscale: float

    :param yscale: Scale in y
    :type yscale: float
    """
    counter = 0
    with open(config.output.coordinateFile,'w+') as file:
        for cords in cordlist:
            file.write("Punkt " + str(counter) + "x " + str(cords[0]*xscale) + "  y " + str(cords[1]*yscale*-1+config.measurement.yCord) + "\n")
            counter = counter + 1
    return 

def CircleDetection(orgImage, pixelRadius):
    """
    Get the Centerpoint of Markers in Image by Circle detection

    :param orgImage: Image containing the Markers
    :type orgImage: Opencv Image

    :param pixelRadius: Radius of Marker in Pixel
    :type pixelRadius: int

    :return: Coordinates of Markers
    :rtype: np.array
    """

    image = orgImage.copy()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, config.measurement.markerColorLower, config.measurement.markerColorUpper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask,kernel,iterations = 1)

    # do some processing on the image
    processed = brightnessContrastCorrection(image,contrast= 150)
    processed = cv2.GaussianBlur(processed, (5, 5),0)
    processed = cv2.Canny(processed, 70,40)
    processed = cv2.GaussianBlur(processed, (7, 7),0)
    processed = cv2.bitwise_and(processed,processed,mask = mask)

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(processed, 
                        cv2.HOUGH_GRADIENT,1, int(pixelRadius*2), param1 = 20,
                        param2 =13, minRadius = int(pixelRadius-2), maxRadius = int(pixelRadius+2))
    # Draw circles that are detected.
    if detected_circles is not None:
        
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        if config.output.plot:
            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]
        
                # Draw the circumference of the circle.
                cv2.circle(image, (a, b), r, (0, 255, 0), 2)
        
                # Draw a small circle (of radius 1) to show the center.
                cv2.circle(image, (a, b), 1, (0, 0, 255), 3)
    if config.output.plot:
        maskS = cv2.resize(mask, (int(config.camera.videoFormat[0]/2), int(config.camera.videoFormat[1]/2)))
        cv2.imshow("Mask", maskS)
        imS = cv2.resize(image, (int(config.camera.videoFormat[0]/2), int(config.camera.videoFormat[1]/2)))
        cv2.imshow("Detected Circle", imS)
        prS = cv2.resize(processed, (int(config.camera.videoFormat[0]/2), int(config.camera.videoFormat[1]/2)))
        cv2.imshow("Processed", prS)
        #cv2.imwrite(folder+str(i+1)+'cutout.jpg',warped)
        cv2.waitKey(0)
    return detected_circles


