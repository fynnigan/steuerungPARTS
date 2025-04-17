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

import os
import sys

if __name__ == "__main__":
    if os.getcwd().split('\\')[-1].__contains__('functions'):
        os.chdir('..')
    if sys.path[0].split('\\')[-1].__contains__('functions'):
        sys.path.append(os.getcwd())
    
import scipy as sp
from scipy import optimize
from copy import deepcopy
import numpy as np
import logging
from functions import com_functions as cf
from functions import data_functions as df
from functions import marker_functions as mf
from functions import webserver_functions as wf
from functions import control_functions as cof

from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import config
import cv2
import time
import json

L1=config.lengths.L1
L2=config.lengths.L2
L3=config.lengths.L3
L4=config.lengths.L4
L5=config.lengths.L5

phi=math.atan(L3/L1)
ce=L3/math.sin(phi)
L0=config.lengths.L0
L100=config.lengths.L100

TESTADEPOS = np.array([[1,1,1,0,1,2,0,1,3,0,2,2,0,2,3,0,3,3,0],[2,1.1,1.1,0,1.1,2.1,0,1.1,3.1,0,2.1,2.1,0,2.1,3.1,0,3.1,3.1,0],[3,1.2,1.2,0,1.2,2.2,0,1.2,3.2,0,2.2,2.2,0,2.2,3.2,0,3.2,3.2,0]])
TESTWRONGADEPOS = np.array([[1,1,1,0,1,2,0,1,3,0,2,2,0,2,3,0,3,3,0],[2,1.1,2.1,0,1.1,1.1,0,1.1,3.1,0,2.1,2.1,0,3.1,3.1,0,2.1,3.1,0],[3,1.2,1.2,0,1.2,3.2,0,1.2,2.2,0,2.2,3.2,0,2.2,2.2,0,3.2,3.2,0]])
TESTACTORLENGTH = np.array([0,0.14,0.28,0.32])
TESTACTORLENGTHMEASURED = np.array([0,0.13,0.29,0.32])

def OpenCamera():
    logging.debug('Opening Camera')
    cap = cv2.VideoCapture(config.camera.ID)
    cap.set(cv2.CAP_PROP_FOURCC, config.camera.codec)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.videoFormat[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.videoFormat[1])

    while True:
        logging.debug('Trying to read Frame')
        ret,testImg = cap.read()
        if ret == False:
            logging.debug('Open Camera Failed: ret = {}'.format(ret))
            cap.release()
            cap = cv2.VideoCapture(config.camera.ID)
            cap.set(cv2.CAP_PROP_FOURCC, config.camera.codec)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.videoFormat[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.videoFormat[1])
            continue
        elif testImg.max() == 0:
            logging.debug('Open Camera Failed: TestImg.max = {}'.format(testImg.max()))
            cap.release()
            cap = cv2.VideoCapture(config.camera.ID)
            cap.set(cv2.CAP_PROP_FOURCC, config.camera.codec)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.videoFormat[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.videoFormat[1])
            
        else:
            break
    logging.debug('Camera Open')
    return cap

def TakeImage(cap):
    logging.debug('Reading Image')
    while True:
        #for j in range(1):
        #    cap.read()
        ret,image = cap.read()
        if ret == False:
            continue
        return image

def MeasurePoints(image):    
    logging.debug('Undistort Image')
    image = UndistordImage(image)
    logging.debug('Detecting Coordinate Markers')
    detected_circles = mf.GetMarker(image,lowerLimit=config.measurement.markerColorLower ,upperLimit=config.measurement.markerColorUpper,pixelRadius = 38)
    if (detected_circles is None) or (int(np.shape(detected_circles)[0])!=4):
        #logging.warning("Coordinate Markers not detected")
        raise ValueError("Coordinate Markers not detected")
    logging.debug('Detected Coordinate Markers')
    cord = np.squeeze(detected_circles)
    cord = cord.astype(float)

    circlesImg,cord_sys = mf.four_point_transform(image,cord)
    x_scale,y_scale = mf.get_scale(cord_sys)
    logging.debug('Detecting Markers')
    detected_circles = mf.GetMarker(circlesImg, lowerLimit=config.measurement.markerColorLower ,upperLimit=config.measurement.markerColorUpper,pixelRadius = int(config.measurement.markerRadius/x_scale))
    if detected_circles is None:
        #logging.warning("No Circles Detectet")
        imS = cv2.resize(circlesImg, (1160, 600))
        cv2.imshow("No Circles Detectet", imS)
        cv2.waitKey(0)
        raise ValueError("No Circles Detectet")
    logging.info(f'{int(len(detected_circles)):2d}: Markers Detected')
    for n in range(len(detected_circles)):
        circlesImg = cv2.circle(circlesImg, (int(detected_circles[n][0]),int(detected_circles[n][1])),int(config.measurement.markerRadius/x_scale),(0,0,255),2)
        circlesImg = cv2.circle(circlesImg, (int(detected_circles[n][0]),int(detected_circles[n][1])), radius=1, color=(0, 0, 255), thickness=2)
    circlesImg = cv2.resize(circlesImg, (int(config.measurement.xCord*1000),int(config.measurement.yCord*1000)))#(int(circlesImg.shape[1]/2),int(circlesImg.shape[0]/2)))
    #cv2.imwrite(config.output.folder+'detectedCircles/'+ str(i+1) + ".jpg",circlesImg)
    #cv2.imwrite(config.output.folder+'orgImg/'+ str(i+1) + ".jpg",image)
    #cv2.imwrite(config.output.folder+'warpedImg/'+ str(i+1) + ".jpg",warped)

    cordAde = np.squeeze(detected_circles)
    cordAde = cordAde.astype(float)
    cordAde[:,0] = (cordAde[:,0])*x_scale
    cordAde[:,1] = (cordAde[:,1])*y_scale*-1+config.measurement.yCord
    cordAde = np.reshape(cordAde, -1)
    cordAde = np.insert(cordAde, 0, 1., axis=0)
    return cordAde, circlesImg

def RunCameraMeasurement(adeMoveLst,adeLst):
    """
    Run the ADE Movelist and Measure the Movement

    :param adeMoveLst: List containing the ADE Movements
    :type adeMoveLst: np.array

    :param adeLst: List containing the ADE's
    :type adeLst: list of ADEs

    :return: Returns the Measured Points
    :rtype: np.array
    """
    measurementPoints = np.zeros((len(adeMoveLst.data),config.measurement.PointsPerAde*len(adeLst)+2,3))
    logging.debug('Opening Camera')
    cap = cv2.VideoCapture(0, config.camera.API)
    cap.set(cv2.CAP_PROP_FOURCC, config.camera.codec)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.videoFormat[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.videoFormat[1])
    
    while True:
        logging.debug('Trying to read Frame')
        ret,testImg = cap.read()
        if ret == False or testImg.max() == 0:
            logging.debug('Open Camera Failed: ret = {} testImg.max = {}'.format(ret,testImg.max()))
            cap.release()
            cap = cv2.VideoCapture(0, config.camera.API)
            cap.set(cv2.CAP_PROP_FOURCC, config.camera.codec)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera.videoFormat[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera.videoFormat[1])
            
        else:
            break

    logging.debug('Camera Open')
    for i in range(int(len(adeMoveLst.data[:,0]))):
        logging.info("""----------------------------------------------------- \n {}. - Move List Zeile \n----------------------------------------------------- """.format(i+1))
        adeMoveLst.FillADE(i,adeLst)
        logging.debug('Moving ADEs')
        for ade in adeLst:
            ade.GoTo()           
        for ade in adeLst:
            ade.ReadBattery()
        time.sleep(cf.CalcPause(adeMoveLst,i))

        while True:
            logging.debug('Reading Image')
            for j in range(30):
                cap.read()
            ret,image = cap.read()
            if ret == False:
                continue
            logging.debug('Undistort Image')
            image = UndistordImage(image)
            logging.debug('Detecting Coordinate Markers')
            detected_circles = mf.GetMarker(image,pixelRadius = 39)
            if (detected_circles is None) or (int(np.shape(detected_circles)[0])!=4):
                continue
            logging.debug('Detected Coordinate Markers')
            cord = np.squeeze(detected_circles)
            cord = cord.astype(float)

            warped,cord_sys = mf.four_point_transform(image,cord)
            x_scale,y_scale = mf.get_scale(cord_sys)
            logging.debug('Detecting Markers')
            detected_circles = mf.GetMarker(warped, pixelRadius = int(config.measurement.markerRadius/x_scale))
            if detected_circles is None:
                logging.warning("No Circles Detectet")
                imS = cv2.resize(warped, (1160, 600))
                cv2.imshow("No Circles Detectet", imS)
                cv2.waitKey(0)
                continue
            logging.debug('{int(len(detected_circles)):2d}: Markers Detected')
            #cv2.imwrite(config.output.folder+'orgImg/'+ str(i+1) + ".jpg",image)
            #cv2.imwrite(config.output.folder+'warpedImg/'+ str(i+1) + ".jpg",warped)

            cordAde = np.squeeze(detected_circles)
            cordAde = cordAde.astype(float)
            cordAde[:,0] = (cordAde[:,0])*x_scale
            cordAde[:,1] = (cordAde[:,1])*y_scale*-1+config.measurement.yCord
            measurementPoints[i,:cordAde.shape[0]] = cordAde[:len(measurementPoints[i,:])].copy()
            del ret,image,detected_circles,cordAde,cord,x_scale,y_scale
            break

    cap.release()
    #measurementPoints = SortPoints(measurementPoints,boundary = 40)
    np.save(config.output.folder+"ADEFahrt.npy", measurementPoints)
    np.savetxt(config.output.folder+"actorlength.txt",adeMoveLst.data, delimiter=',', fmt='%i')
    return measurementPoints 

def UndistordImage(image):
    """
    Undistorts the taken Images with the Camera Parameters

    :param image: Image to be undistorted
    :type image: opencv Image

    :return: Return the undistorted Image
    :rtype: opencv Image
    """
    

    h,  w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(config.measurement.mtx, config.measurement.dist, (w,h), 1, (w,h))

    # undistort
    image = cv2.undistort(image, config.measurement.mtx, config.measurement.dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    image = image[y:y+h, x:x+w]

    return image

def RunImageMeasurement(folder,adeMoveLst,adeLst):
    """
    Measure the Movement of ADE's by using saved Images of the Movement

    :param folder: folder containing the images
    :type folder: string

    :param adeMoveLst: List containing the ADE Movements
    :type adeMoveLst: np.array

    :param adeLst: List containing the ADE's
    :type adeLst: list of ADEs
    
    :return: Returns the Measured Points
    :rtype: np.array
    """
    measurementPoints = np.zeros((len(adeMoveLst.data),(config.measurement.PointsPerAde*len(adeLst)+2)*3+1))
    for i in range(int(len(adeMoveLst.data[:,0]))):
        logging.debug('Reading Image')
        image = cv2.imread(folder+str(i+1)+'.jpg')
        logging.debug('Undistord Image')
        image = UndistordImage(image)
        logging.debug('Detecting Coordinate Markers')
        detected_circles = mf.GetMarker(image,pixelRadius = 38,lower_limit = np.array([50, 25, 0]),upper_limit = np.array([130, 90, 255]))
        if (detected_circles is None) or (int(np.shape(detected_circles)[0])!=4):
            continue
        logging.debug('Detected Coordinate Markers')
        cord = np.squeeze(detected_circles)
        cord = cord.astype(float)

        warped,cord_sys = mf.four_point_transform(image,cord)
        x_scale,y_scale = mf.get_scale(cord_sys)
        logging.debug('Detecting Markers')
        detected_circles = mf.GetMarker(warped, pixelRadius = int(config.measurement.markerRadius/x_scale),lower_limit = np.array([50, 25, 0]),upper_limit = np.array([130, 90, 255]))
        if detected_circles is None:
            logging.warning('No Circles Detectet')
            imS = cv2.resize(warped, (1160, 600))
            cv2.imshow("No Circles Detectet", imS)
            cv2.waitKey(0)
            continue
        logging.debug(f'{int(len(detected_circles)):2d} Markers Detected')
        for n in range(len(detected_circles)):
            circlesImg = cv2.circle(warped, (int(detected_circles[n][0]),int(detected_circles[n][1])),int(config.measurement.markerRadius/x_scale),(0,0,255),2)
            circlesImg = cv2.circle(circlesImg, (int(detected_circles[n][0]),int(detected_circles[n][1])), radius=1, color=(0, 0, 255), thickness=2)
        cv2.imwrite(config.output.folder+'detectedCircles/'+ str(i+1) + ".jpg",circlesImg)

        cordAde = np.squeeze(detected_circles)
        cordAde = cordAde.astype(float)
        cordAde[:,0] = (cordAde[:,0])*x_scale
        cordAde[:,1] = (cordAde[:,1])*y_scale*-1+config.measurement.yCord
        cordAde= cordAde.reshape((1,(config.measurement.PointsPerAde*len(adeLst)+2)*3))
        measurementPoints[i] = np.c_[i+1,cordAde.copy()]

        del image,detected_circles,cordAde,cord,x_scale,y_scale

    np.save(config.output.folder+"ADEFahrt.npy", measurementPoints)
    np.savetxt(config.output.folder+"actorlength.txt",adeMoveLst.data, delimiter=',', fmt='%i')
    return measurementPoints

def SortPointsADEBoundary(adePos, pauseSort,figureID):
    """
    Splits the Measured Points into a tuple containing the Measured ADE Points and plots the Points

    :param adePos: List of measured ADE Positions
    :type adePos: np.array

    :param pauseSort: Command to Pause after Sorting
    :type pauseSort: boolean

    :param figureID: Name of the Figure to plot the Points in
    :type figureID: string

    :return: Returns split up measurement array
    :rtype: np.array

    >>> type(SortPointsADEBoundary(TESTADEPOS, 0)) == list
    True
    >>> SortPointsADEBoundary(TESTADEPOS, 0)
    [array([1., 2., 3.]), array([[0. , 1. , 1. ],
           [0. , 1.1, 1.1],
           [0. , 1.2, 1.2]]), array([[0. , 1. , 2. ],
           [0. , 1.1, 2.1],
           [0. , 1.2, 2.2]]), array([[0. , 1. , 3. ],
           [0. , 1.1, 3.1],
           [0. , 1.2, 3.2]]), array([[0. , 2. , 2. ],
           [0. , 2.1, 2.1],
           [0. , 2.2, 2.2]]), array([[0. , 2. , 3. ],
           [0. , 2.1, 3.1],
           [0. , 2.2, 3.2]]), array([[0. , 3. , 3. ],
           [0. , 3.1, 3.1],
           [0. , 3.2, 3.2]])]
    >>> len(SortPointsADEBoundary(TESTADEPOS, 0)) == 7
    True

    """


    time = adePos[:,0]
    # Spliting up the Measurement array
    vp = tuple([time])
    
    for n in range(int((np.shape(adePos)[1]-1)/3)):
        vp = vp+tuple([adePos[:,(n)*3+1:(n)*3+4]])
        #vp = vp+tuple([adePos[:,n]])

    # Plotting the first line of Measurements
    if pauseSort == True:
        plt.figure(figureID)
        for i in range(1,len(vp)):
            plt.scatter(vp[i][0,0],vp[i][0,1],facecolors='green', edgecolors = "green" )
            plt.text(vp[i][0,0],vp[i][0,1],str(i),fontsize=8);
        plt.show(block = False)
        plt.pause(0.001)
    return vp

def CalcStroke( aktor,actorLength,nr ):
    """
    Calculate the desired actor Length and the actually measured Length

    :param aktor: Measurement of Aktor Points
    :type aktor: np.array

    :param actorLength: Array of Actor Lengths
    :type actorLength: np.array

    :param nr: Number of ADE
    :type nr: int

    :return: Returns the desired actor Length and the actually measured Length
    :rtype: tuple of np.arrays
    """
    actorLengthDesired=actorLength[nr-1]/10000-actorLength[nr-1,0]/10000
    actorLengthMeasured=[]
    for i in range(len(aktor)):
        actorLengthMeasured=np.concatenate([actorLengthMeasured,[LA.norm(aktor[i])]])
    actorLengthMeasured = actorLengthMeasured-L0
    return actorLengthDesired,actorLengthMeasured

def SortMeasuredPoints( adePos, actorLength = None, boundary = 10,execution = 1, nADE = None , draw = False,adeNodes = None,viconMode = False, offsetAngle = 0, sortByAngle = False):
    """
    Sort the Measured Points of the ADE Position with tow Methodes:
    Methode 1: Sort them by Proximiy to each other (best to use when calibrating the Aktuators)
    Methode 2: Sort them by simulating the ADE Movement and then by Proximity to ideal Points (use when Measuring an ADE Movement)

    :param adePos: List of measured ADE Positions
    :type adePos: np.array

    :param actorLength: List of desired Actor Lengths
    :type actorLength: np.array

    :param boundary: boundary parameter defining the max distance between consecutive measurement Points
    :type boundary: int

    :param L1: L1 Parameter for Executionmode 2
    :type L1: float

    :param L3: L3 Parameter for Executionmode 2
    :type L3: float

    :param execution: Mode of execution
    :type execution: int

    :param nADE: Number of ADE or Executionmode 2 
    :type nADE: Int

    :return: Returns sorted Array of ADE Positions
    :rtype: np.array

    >>> (SortMeasuredPoints(TESTADEPOS,TESTACTORLENGTH,0.5,1,2,1,1) == TESTADEPOS).all()
    True
    >>> ((SortMeasuredPoints(TESTWRONGADEPOS,TESTACTORLENGTH,0.5,1,2,1,1)) == TESTADEPOS).all()
    True
    >>> len(SortMeasuredPoints(TESTADEPOS,TESTACTORLENGTH,0.5,1,2,1,1)) == len(TESTADEPOS)
    True
    >>> len(SortMeasuredPoints(TESTWRONGADEPOS,TESTACTORLENGTH,0.5,1,2,1,1)) == len(TESTADEPOS)
    True
    """
    if execution == 1:
        # execution Mode 1: 
        # Sorting measured Points by absolute distance between Points
        for j in range(1,adePos.shape[0]):
            for i in range(int((adePos.shape[1]-1)/3)):
                if abs(adePos[j-1,i*3+1]-adePos[j,i*3+1])>=boundary or abs(adePos[j-1,i*3+2]-adePos[j,i*3+2])>=boundary or abs(adePos[j-1,i*3+3]-adePos[j,i*3+3])>=boundary:
                    for k in range(int((adePos.shape[1]-1)/3)):
                        if abs(adePos[j-1,i*3+1]-adePos[j,k*3+1])<=boundary and abs(adePos[j-1,i*3+2]-adePos[j,k*3+2])<=boundary and abs(adePos[j-1,i*3+3]-adePos[j,k*3+3])<=boundary:
                            # switching Positions of measured Points
                            adePos[j,i*3+1:i*3+1+3],adePos[j,k*3+1:k*3+1+3]=adePos[j,k*3+1:k*3+1+3].copy(),adePos[j,i*3+1:i*3+1+3].copy()

    if execution == 2:
        # execution Mode 2: 
        # Sorting measured Points by distance to ideal Triangle
        #####
        ## Important needs rework, Data returned from Stroke2desiredMeshChanged to dictionary !!!!!!!!!!!!!!!!!!!!
        for j in range(len(adePos[:,1])):
            #if draw:
            #    plt.figure()
            [Ai1,Bi1,Ci1,Ai2,Bi2,Ci2,Ai3,Bi3,Ci3,Ai4,Bi4,Ci4,Ai5,Bi5,Ci5,Ai6,Bi6,Ci6,Bound1,Bound2,nodes, trigs, strokeLength] = Stroke2desiredMesh(actorLength[j,:]/10,boundary[0,:],L1,L3,nADE,draw,figureName="Line "+str(adePos[j,0]))


            if nADE==6:
                [As1,As2,Bs1,Bs2,Cs1,Cs2]=IdealMesh2idealActuatorPoints( Ai1,Bi1,Ci1,L1,L3,draw );

                [Ai1,Bi1,Ci1]=IdealMesh2idealMeasurement( Ai1,Bi1,Ci1,draw )
                [Ai2,Bi2,Ci2]=IdealMesh2idealMeasurement( Ai2,Bi2,Ci2,draw )
                [Ai3,Bi3,Ci3]=IdealMesh2idealMeasurement( Ai3,Bi3,Ci3,draw )
                [Ai4,Bi4,Ci4]=IdealMesh2idealMeasurement( Ai4,Bi4,Ci4,draw )
                [Ai5,Bi5,Ci5]=IdealMesh2idealMeasurement( Ai5,Bi5,Ci5,draw )
                [Ai6,Bi6,Ci6]=IdealMesh2idealMeasurement( Ai6,Bi6,Ci6,draw )

                P=np.concatenate((Ai1,Bi1,Ci1,Ai2,Bi2,Ci2,Ai3,Bi3,Ci3,Ai4,Bi4,Ci4,Ai5,Bi5,Ci5,Ai6,Bi6,Ci6,Bound1,Bound2),axis=1)

            if nADE==4:
                [As1,As2,Bs1,Bs2,Cs1,Cs2]=IdealMesh2idealActuatorPoints( Ai1,Bi1,Ci1,L1,L3,draw );

                [Ai1,Bi1,Ci1]=IdealMesh2idealMeasurement( Ai1,Bi1,Ci1,draw );
                [Ai2,Bi2,Ci2]=IdealMesh2idealMeasurement( Ai2,Bi2,Ci2,draw );
                [Ai3,Bi3,Ci3]=IdealMesh2idealMeasurement( Ai3,Bi3,Ci3,draw );
                [Ai4,Bi4,Ci4]=IdealMesh2idealMeasurement( Ai4,Bi4,Ci4,draw );
                #plt.show()
                P=np.concatenate((Ai1,Bi1,Ci1,Ai2,Bi2,Ci2,Ai3,Bi3,Ci3,Ai4,Bi4,Ci4,Bound1,Bound2),axis=1)

            if nADE == 1:
                [As1,As2,Bs1,Bs2,Cs1,Cs2]=IdealMesh2idealActuatorPoints( Ai1,Bi1,Ci1,L1,L3 )
                [Ai1,Bi1,Ci1]=IdealMesh2idealHinge( Ai1,Bi1,Ci1,L1,L3 )
                P=[Ai1,Bi1,Ci1,As1,As2,Bs1,Bs2,Cs1,Cs2,Bound1,Bound2]
            for m in range(len(P[0,:])):     
                posPi=(m*3)+1
                pi = P[:,m]
                
                disAlt = 100
                posPirSmallest = 1
                for i in range(int((len(adePos[1,:])-1)/3)):
                    posPir = i*3+1
                    pir=adePos[j,posPir:posPir+2].transpose().copy()
                    dis = LA.norm(pir-pi)
                    if(dis < disAlt):
                        posPirSmallest = posPir
                        disAlt = dis              
                adePos[j,posPi:posPi+3],adePos[j,posPirSmallest:posPirSmallest+3]=adePos[j,posPirSmallest:posPirSmallest+3].copy(),adePos[j,posPi:posPi+3].copy()

                #posPi=(m)
                #pi = P[:,m]
                #
                #disAlt = 100
                #posPirSmallest = 1
                #for i in range(int((len(adePos[1,:])-1))):
                #    posPir = i+1
                #    pir=adePos[j,posPir,0:2].transpose().copy()
                #    dis = LA.norm(pir-pi)
                #    if(dis < disAlt):
                #        posPirSmallest = posPir
                #        disAlt = dis              
                #adePos[j,posPi],adePos[j,posPirSmallest]=adePos[j,posPirSmallest].copy(),adePos[j,posPi].copy()
            #plt.figure("ADE Movement")
            #for i in range(1,len(vp)):
            #    plt.scatter(vp[i][0,0],vp[i][0,1],facecolors='none', edgecolors = "red" )
            #    plt.text(vp[i][0,0],vp[i][0,1],str(i),fontsize=8);
            if draw:
                plt.figure("Line "+str(adePos[j,0]))
                for i in range(int((len(adePos[1,:])-1)/3)):
                    plt.scatter(adePos[j,i*3+1],adePos[j,i*3+2],facecolors='none', edgecolors='green')
            nB=len(P[0,:])-1
            P1=adePos[j,(nB-1)*3+1:(nB-1)*3+1+2][np.newaxis].transpose().copy()
            P2=adePos[j,(nB+1-1)*3+1:(nB+1-1)*3+1+2][np.newaxis].transpose().copy()
            for i in range(int((len(adePos[0,:])-1)/3)):  
                adePos[j,i*3+1:i*3+1+2]=SetInitialADE(adePos[j,i*3+1:i*3+1+2].copy(),P1,P2)

    if execution == 3:
        MarkerIndex = {}
        if type(boundary) != int:
            P1=boundary[:,0:2]
            P2=boundary[:,3:-1]
            for i in range(int((adePos[0,:].shape[1]-1)/3)):  
                adePos[:,i*3+1:i*3+1+2]=SetInitialADE(adePos[:,i*3+1:i*3+1+2].copy(),P1,P2,viconMode = viconMode,offsetAngle = offsetAngle)
        
        for ID in adeNodes.elements:
            adeNodes.elementMarkerPosition[ID] = deepcopy(adeNodes.elementMarkerPositionEstimate[ID])
            MarkerIndex[ID] = {}
            for i in range(3):
                pi = adeNodes.elementMarkerPositionEstimate[ID][i].T
                    
                disAlt = 1
                posPirSmallest = 1
                for j in range(int((adePos.shape[1]-1)/3)):
                    posPir = j*3+1
                    pir=adePos[0,posPir:posPir+2].transpose().copy()
                    dis = LA.norm(pir-pi)
                    if(dis < disAlt):
                        posPirSmallest = posPir
                        disAlt = dis
                #adePos[0,posPi:posPi+3],adePos[0,posPirSmallest:posPirSmallest+3]=adePos[0,posPirSmallest:posPirSmallest+3].copy(),adePos[0,posPi:posPi+3].copy()
                adeNodes.elementMarkerPosition[ID][i][:], adePos = adePos[0,posPirSmallest:posPirSmallest+2].copy(), np.delete(adePos,[posPirSmallest,posPirSmallest+1,posPirSmallest+2],1)
                MarkerIndex[ID][i] = posPirSmallest
        if sortByAngle:
            avgMarkerPosition = cof.AverageMeshPoints(adeNodes.elementMarkerPosition,adeNodes,False)
            avgIdealMarkerPosition = cof.AverageMeshPoints(adeNodes.elementMarkerPositionEstimate,adeNodes,False)
            #markerPosition= adeNodes.elementMarkerPosition.copy()
            
            

            for node in adeNodes.adePointsOfNode:
                markerPosition= deepcopy(adeNodes.elementMarkerPosition)

                #offsetAngle = 0
                #alphaIdeal = 0
                #alphaMarker2 = 0
                #for ID in adeNodes.adePointsOfNode[node]:
                #    anIdeal = (adeNodes.elementMarkerPositionEstimate[ID][adeNodes.adePointsOfNode[node][ID]] - avgIdealMarkerPosition[ID][adeNodes.adePointsOfNode[node][ID]])#/LA.norm(adeNodes.elementMarkerPositionEstimate[ID][adeNodes.adePointsOfNode[node][ID]] - avgIdealMarkerPosition[ID][adeNodes.adePointsOfNode[node][ID]])
                #    alphaIdeal +=  math.atan2(anIdeal[0,1],anIdeal[0,0])
                #
                #    anMarker2 = (markerPosition[ID][adeNodes.adePointsOfNode[node][ID]] - avgMarkerPosition[ID][adeNodes.adePointsOfNode[node][ID]])#/LA.norm(markerPosition[ID2][adeNodes.adePointsOfNode[node][ID2]] - avgMarkerPosition[ID2][adeNodes.adePointsOfNode[node][ID2]])
                #    alphaMarker2 += math.atan2(anMarker2[0,1],anMarker2[0,0])
                #
                #alphaIdeal = alphaIdeal/len(adeNodes.adePointsOfNode[node])
                #alphaMarker2 = alphaMarker2/len(adeNodes.adePointsOfNode[node])
                #
                #diff_rad2 = alphaIdeal - alphaMarker2
                #offsetAngle = min(diff_rad2, 2 * math.pi - diff_rad2)

                if len(adeNodes.adePointsOfNode[node]) <=1:
                    continue
                for ID in adeNodes.adePointsOfNode[node]:
                    
                    anIdeal = (adeNodes.elementMarkerPositionEstimate[ID][adeNodes.adePointsOfNode[node][ID]] - avgIdealMarkerPosition[ID][adeNodes.adePointsOfNode[node][ID]])#/LA.norm(adeNodes.elementMarkerPositionEstimate[ID][adeNodes.adePointsOfNode[node][ID]] - avgIdealMarkerPosition[ID][adeNodes.adePointsOfNode[node][ID]])
                    alphaIdeal =  math.atan2(anIdeal[0,1],anIdeal[0,0])
                    if alphaIdeal < 0:
                        alphaIdeal += 2 * math.pi

                    #anMarker = (adeNodes.elementMarkerPosition[ID][adeNodes.adePointsOfNode[node][ID]] - avgMarkerPosition[ID][adeNodes.adePointsOfNode[node][ID]])#/LA.norm(adeNodes.elementMarkerPosition[ID][adeNodes.adePointsOfNode[node][ID]] - avgMarkerPosition[ID][adeNodes.adePointsOfNode[node][ID]])
                    #alphaMarker = math.atan2(anMarker[0,1],anMarker[0,0])
                    #if alphaMarker < 0:
                    #    alphaMarker += 2 * math.pi
                    #
                    #diff_rad = abs(alphaIdeal - alphaMarker)
                    smallestDist = 100
                    #nearedsPoint = ID
                    for ID2 in adeNodes.adePointsOfNode[node]:
                        if ID2 not in markerPosition:
                            continue
                        
                        anMarker2 = (markerPosition[ID2][adeNodes.adePointsOfNode[node][ID2]] - avgMarkerPosition[ID2][adeNodes.adePointsOfNode[node][ID2]])#/LA.norm(markerPosition[ID2][adeNodes.adePointsOfNode[node][ID2]] - avgMarkerPosition[ID2][adeNodes.adePointsOfNode[node][ID2]])
                        alphaMarker2 = math.atan2(anMarker2[0,1],anMarker2[0,0])
                        if alphaMarker2 < 0:
                            alphaMarker2 += 2 * math.pi
                        
                        diff_rad2 = abs(alphaIdeal - alphaMarker2)
                        alphaDist = min(diff_rad2, 2 * math.pi - diff_rad2)

                        if alphaDist < smallestDist:
                            smallestDist = deepcopy(alphaDist)
                            nearedsPoint = ID2
                    #if nearedsPoint != ID:
                    #adeNodes.elementMarkerPosition[ID][adeNodes.adePointsOfNode[node][ID]],adeNodes.elementMarkerPosition[nearedsPoint][adeNodes.adePointsOfNode[node][nearedsPoint]] = adeNodes.elementMarkerPosition[nearedsPoint][adeNodes.adePointsOfNode[node][nearedsPoint]].copy(),adeNodes.elementMarkerPosition[ID][adeNodes.adePointsOfNode[node][ID]].copy()
                    adeNodes.elementMarkerPosition[ID][adeNodes.adePointsOfNode[node][ID]] = deepcopy(markerPosition[nearedsPoint][adeNodes.adePointsOfNode[node][nearedsPoint]])
                    markerPosition.pop(nearedsPoint)

        return MarkerIndex
    if execution == 4:
        # execution Mode 4:
        # Sorting measured Points by minimal distance between Points
        for i in range(1,adePos.shape[0]):
            for j in range(1,adePos.shape[1],3):
                posPi=j
                pi = adePos[0,posPi:posPi+2]
                
                disAlt = 100
                posPirSmallest = 1
                for k in range(1,adePos.shape[1],3):
                    posPir = k
                    pir=adePos[i,posPir:posPir+2].transpose().copy()
                    dis = LA.norm(pir-pi)
                    if(dis < disAlt):
                        posPirSmallest = posPir
                        disAlt = dis              
                adePos[i,posPi:posPi+3],adePos[i,posPirSmallest:posPirSmallest+3]=adePos[i,posPirSmallest:posPirSmallest+3].copy(),adePos[i,posPi:posPi+3].copy()
    return adePos

def LoadViCon(adeID,itteration):
    """
    Load ViCon Data to calibrate the Aktuators

    :param adeID: ID of ADE 
    :type adeID: int

    :param itteration: itteration of Measurment
    :type itteration: int

    :return: Returns the previous ADE Parameter, the desired Actor Lengths and the ViCon Measurements
    :rtype: numpy array, numpy array, numpy array
    """
    if itteration == 0:
        # Default Parameters
        param = np.array([[1000.,1000.,1000.],[0.,0.,0.],[0.,0.,0.],[1.,1.,1.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[1.,1.,1.],[0.,0.,0.]])
    else:
        # previous corrected Parameters
        param = np.genfromtxt('..\MessdatenADE\ADE_'+str(adeID)+'_0_1000_corr_'+str(itteration-1)+'\ADE_'+str(adeID)+'.txt', delimiter=',')
    # Read desired Actor Lengths
    adeActorlength = np.genfromtxt('..\MessdatenADE\ADE_'+str(adeID)+'_0_1000_corr_'+str(itteration)+'\ViCon_Actorlenght.txt', delimiter=',')
    adeActorlength=adeActorlength[:,2+7*(adeID-1):5+7*(adeID-1)]
    # Read measured Points
    viConPos = np.genfromtxt('..\MessdatenADE\ADE_'+str(adeID)+'_0_1000_corr_'+str(itteration)+'\Vicon_Messung.txt', delimiter=',')

    return(param,adeActorlength,viConPos)

def SetActorPoint(pointStr):
    """
    :param pointStr: shown on Input
    :type pointStr: String

    :return: Returns sanitized number
    :rtype: int
    """
    while True:
        try:
            number = int(input(pointStr))
            if number < 1 or number > 6:
                raise ValueError
            break
        except ValueError:
            logging.warning('The Input has to be an Integer between 1 and 6')
    return number

def PlotAndCalcPoly(actorLengthDesired,actorLengthMeasured,startLength,endLength,actorNumber,travelDirection):
    """
    Plot the Movement and calculate and plot the Corretion Polynom of the Aktuator calibration

    :param actorLengthDesired: Array of desired Actor Lengths
    :type actorLengthDesired: np.array

    :param actorLengthMeasured: Array of measured Actor Lengths
    :type actorLengthMeasured: np.array
    
    :param startLength: Index of first Measurment to be considered
    :type startLength: int
    
    :param endLength: Index of last Measurment to be considered
    :type endLength: int

    :param actorNumber: Actor Number
    :type actorNumber: Int

    :param travelDirection: String containing the description of the Movement Direction
    :type travelDirection: string

    :return: Returns the coefficients of the Polynom and the mean Offset of the datapoints
    :rtype: tuple of floats

    >>> len(PlotAndCalcPoly(TESTACTORLENGTH,TESTACTORLENGTHMEASURED,0,3,1,"extend")) == 5
    True
    """
    # set Plot Position in Subplots
    if travelDirection == "extend":
        plotPosition = actorNumber
    else:
        plotPosition = actorNumber + 3

    # Plot measured Actor travel
    # Can be uncommented if needed
    #plt.figure('measured Actor travel')
    #plt.subplot(2,3,plotPosition)
    #plt.plot(actorLengthDesired[startLength:endLength],actorLengthMeasured[startLength:endLength],linewidth=2,color='black',markersize=8,linestyle='dashed', marker='x',label='Hin')
    #
    #plt.xlabel('actorLengthDesired in mm')
    #plt.ylabel('actorLengthMeasured in mm')
    #plt.title('Actor'+str(actorNumber) + ' ' + travelDirection)
    #plt.grid()
    #plt.legend()

    

    
    # calculate absolute Error of Actor travel 
    absoluteErr0g=actorLengthMeasured[startLength:endLength]-actorLengthDesired[startLength:endLength]
    
    # calculate mean Offset of Actor travel 
    offset=sum(absoluteErr0g)/len(absoluteErr0g)

    # calculate mean squared Error of Actor travel 

    summation = 0  #variable to store the summation of differences
    n = len(absoluteErr0g) #finding total number of items in list
    for i in range (0,n):  #looping through each element of the list
      difference = absoluteErr0g[i]  #finding the difference between observed and predicted value
      squared_difference = difference**2  #taking square of the differene 
      summation = summation + squared_difference  #taking a sum of all the differences
    RMS = math.sqrt(summation/n)  #dividing summation by total values to obtain average
    
    
    # Plot absolute Error of Actor travel 
    x=actorLengthDesired[startLength:endLength]
    bound=np.full(x.shape[0],0.5)
    offset_line = np.full(x.shape[0],offset)

    plt.figure('absolute Error')
    plt.subplot(2,3,plotPosition)
    plt.plot(x,absoluteErr0g,linestyle='dashed',linewidth=2,color='black',markersize=8,marker='x',label='uncorrErr')
    plt.plot(x,bound,linewidth=2,color='red')
    plt.plot(x,-bound,linewidth=2,color='red')
    plt.plot(x,offset_line,linewidth=2,color='blue')
    plt.grid()
    plt.xlabel('actorLengthDesired in mm')
    plt.ylabel('absolute error in mm')
    plt.title('Actor '+str(actorNumber) + ' ' + travelDirection+ ' RMS: ' + "{:.2f}".format(RMS))
    #plt.ticklabel_format(style='sci',scilimits=(-3,-3))

    
    # calculate coefficients of correcture Polynom
    p = np.polyfit(actorLengthMeasured[startLength:endLength],actorLengthDesired[startLength:endLength],3);

    p1 = p[0]
    p2 = p[1]
    p3 = p[2]
    p4 = p[3]

    # plot correcture Polynom
    
    y=np.zeros(len(x))
    for i in range(len(x)):
        y[i]=p1*x[i]*x[i]*x[i] + p2*x[i]*x[i] + p3*x[i] + p4
    
    corrPolyErr=y.transpose()-actorLengthDesired[startLength:endLength]
    
   
    plt.plot(x,corrPolyErr,linestyle='dashed',linewidth=2,color='red',markersize=8,marker='x',label='corrErr')
    plt.legend()
    
    # plot correcture Polynom, measured Actor travel and ideal Actor travel
    plt.figure('correcture Polynom')
    plt.subplot(2,3,plotPosition)
    plt.plot(x,actorLengthDesired[startLength:endLength],linestyle='dashed',linewidth=2,color='red',markersize=8,marker='x',label='desired')
    plt.plot(x,actorLengthMeasured[startLength:endLength],linestyle='dashed',linewidth=2,color='green',markersize=8,marker='x',label='measured')    
    plt.plot(x,y,linestyle='dashed',linewidth=2,color='blue',markersize=8,marker='*',label='corrected')
    plt.grid()
    plt.xlabel('actorLengthDesired in mm')
    plt.ylabel('actorLengthMeasured in mm')
    plt.title('Actor'+str(actorNumber) + ' ' +  travelDirection + ' RMS: ' + "{:.2f}".format(RMS))
    #plt.ticklabel_format(style='sci',scilimits=(-3,-3))
    plt.legend()


    
    return p1,p2,p3,p4, offset, RMS

def SetInitialADE(adePos,P1,P2,viconMode=False,offsetAngle = 0):
    """
    Define the initial ADE

    :param adePos: List of measured ADE Positions
    :param_type atePos: np.array

    :param P1: 
    :param_type P1: np.array

    :param P2: 
    :param_type P2: np.array

    :param nADE: Number of ADE
    :param_type nADE: int

    :return: Returns ??Pd??
    :rtype: np.array
    """
    ex = np.array([[1],[0]])
    bound = P2-P1
    if viconMode:
        bound = bound.T
        sign = -1
        if bound[1] < 0:
            sgin = 1

        alpha = math.acos(ex.T.dot(bound)/(LA.norm(ex.T)*LA.norm(bound)))*sign+offsetAngle*math.pi/180
        rotAngle = -0
                
        an = ((P2-P1)/LA.norm(P2-P1)).T
        P1=P1-(np.matmul(np.matrix([[math.cos(rotAngle*2*math.pi/360),-math.sin(rotAngle*2*math.pi/360)],[math.sin(rotAngle*2*math.pi/360),math.cos(rotAngle*2*math.pi/360)]]),an)*(config.lengths.L1+4.9e-3)+np.matmul(np.array([[math.cos((rotAngle+90)*2*math.pi/360),-math.sin((rotAngle+90)*2*math.pi/360)],[math.sin((rotAngle+90)*2*math.pi/360),math.cos((rotAngle+90)*2*math.pi/360)]]),an)*(-2.7*1e-3)).T
        pd = np.matmul(RotAz(alpha),np.subtract(adePos,P1).T)
        pd = pd.T
    else:
        bound = bound.T
        sign = -1
        if bound[1,0] < 0:
            sgin = 1
        an = ((P2-P1)/LA.norm(P2-P1))
        alpha = math.acos(an[0,0])+offsetAngle*math.pi/180#math.acos(ex.T.dot(bound)/(LA.norm(ex.T)*LA.norm(bound)))*sign+offsetAngle*math.pi/180
        P1 = P1 - (an*(config.lengths.boundaryXdistance)+np.matrix(Rot(an))*(-config.lengths.boundaryYdistance))
        #alpha = math.acos(ex.T.dot(bound)/(LA.norm(ex.T)*LA.norm(bound)))
        pd =  np.matmul(RotAz(alpha),np.subtract(adePos,P1).T)
        pd = pd.T
    return pd
 
def RotAz(angle):
    """
    Calculate the Rotation matrix of an angle

    :param angle: angle 
    :type angle: float

    :param nADE: Number of ADE
    :type nADE: int

    :return: Returns the Rotation Matrix
    :rtype: np.array
    """

    rotAzVal = np.array([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]])
    return rotAzVal

def RotA(angle,phi):
    """
    Calculate the Rotation matrix of an angle minus an offset phi

    :param angle: angle 
    :type angle: float

    :param phi: offset angle
    :type phi: float

    :return: Returns the Rotation Matrix
    :rtype: np.array
    """

    rotAVal = np.array([[math.cos(angle-phi),-math.sin(angle-phi)],[math.sin(angle-phi),math.cos(angle-phi)]])
    return rotAVal

def Stroke2desiredMesh(adeIDs,strokeLength=None,boundary=None,adeNodes=None,draw=False,figureName=""):
    """
    Calculate an desired Mesh of 1,2,4 or 6 ADEs with the Aktor Lengths (stroke Length)

    :param strokeLength: Lengths of the Aktors
    :type strokeLength: np.array

    :param boundary: boundary Points
    :type boundary: np.array

    :param L1: Parameter L1
    :type L1: float

    :param L3: Parameter L3
    :type L3: float
    
    :param nADE: Number of ADEs
    :type nADE: int

    :return: returns desired Mesh
    :rtype: tuple
    """
    if draw:
        plt.figure(figureName)
    strokeOffset = (L0-2*L1)
    strokeADE = {}
    for ID in adeIDs:
        if type(strokeLength) == df.adeMoveList:
            index = strokeLength.GetIndexOfID(ID)
            strokeADE[ID] = strokeLength.data[0,index+1:index+4]/10000+strokeOffset
        else:
            strokeADE[ID] = np.matrix([0,0,0])+strokeOffset
    if type(strokeLength) == df.adeMoveList:
        index = strokeLength.GetIndexOfID(adeIDs[-1])
        B1 = strokeLength.data[0,index+1]/10000+strokeOffset
    else: 
        B1 = 0+strokeOffset

    #Vorgabe des 1. Dreiecks über Boundary
    if type(boundary) == np.matrix:
        P1=boundary[0,0:2].T
        P2=boundary[0,3:5].T



        Bound1=P1.T
        Bound2=P2.T#P1+(P2-P1)/LA.norm(P2-P1)*(B1-29)
        an = (P2-P1)/LA.norm(P2-P1)
    
        rotAngle = 0
        P1 = P1 - (an*(config.lengths.boundaryXdistance)+np.matrix(Rot(an.T)).T*(-config.lengths.boundaryYdistance))
        #P1=P1-np.matmul(np.matrix([[math.cos(rotAngle*2*math.pi/360),-math.sin(rotAngle*2*math.pi/360)],[math.sin(rotAngle*2*math.pi/360),math.cos(rotAngle*2*math.pi/360)]]),an)*(L1-L4)+np.matmul(np.array([[math.cos((rotAngle+90)*2*math.pi/360),-math.sin((rotAngle+90)*2*math.pi/360)],[math.sin((rotAngle+90)*2*math.pi/360),math.cos((rotAngle+90)*2*math.pi/360)]]),an)*(L3-L5+1e-3)
        P2=P1+np.matmul(np.matrix([[math.cos(rotAngle*2*math.pi/360),-math.sin(rotAngle*2*math.pi/360)],[math.sin(rotAngle*2*math.pi/360),math.cos(rotAngle*2*math.pi/360)]]),an)*(B1)

        P1=P1.T
        P2=P2.T
    else:
        P1 = np.matrix([0,0])
        P2 = np.matrix([B1,0])

        Bound1 = P1
        Bound2 = P2

    
    if adeNodes == None:    #if adeNodes is not defined, the Points Ai,Bi,Ci get stored in a Dictionary
            Ai = {}
            Bi = {}
            Ci = {}

    for ID in adeIDs:
        if adeNodes != None:
            if adeIDs.index(ID) != 0:   #Setting Point P1 and P2 of the ADEs except the First
                Side = adeNodes.connectionList[ID][0][1]
                Nodes = adeNodes.GetNodeOfElementSide(ID,Side)
                P1=adeNodes.nodeDefinition[Nodes[0]]
                P2=adeNodes.nodeDefinition[Nodes[1]]
            else:                       
                Side = adeNodes.GetSideOfElement(ID,adeNodes.baseNodes)
        else: 
            Side = 1
        strokes = rotateMatrixRow(strokeADE[ID],Side-1)     # Rotating the strokes, so they correlate to the Sides AB,BC,CA
        [Aicalc,Bicalc,Cicalc]=DrawStrokeTriangle(P1,P2,B1,np.array([strokes[0,0].copy(),strokes[0,1].copy(),strokes[0,2].copy()]),draw)
        if adeNodes == None:        #Storing the Points Ai,Bi,Ci in a Dictionary
            Ai[ID]=deepcopy(Aicalc)
            Bi[ID]= deepcopy(Bicalc)
            Ci[ID] = deepcopy(Cicalc)
            P1=Ci[ID]
            P2=Bi[ID]
            if draw:
                LabelADE(Ai[ID],Bi[ID],Ci[ID],ID)
        else: 
            if adeIDs.index(ID) == 0:   #Storing Points for First ADE
                adeNodes.nodeDefinition[adeNodes.baseNodes[0]]= deepcopy(Aicalc)
                adeNodes.nodeDefinition[adeNodes.baseNodes[1]]= deepcopy(Bicalc)
                setNodes = set(adeNodes.baseNodes)
                setNodesOfID = set(adeNodes.elements[ID])
                notinList = setNodesOfID - setNodes
                adeNodes.nodeDefinition[notinList.pop()]= deepcopy(Cicalc)
            else:                       #Storing Points for ADEs except the First
                Nodes = adeNodes.GetNodeOfElementSide(ID,adeNodes.connectionList[ID][0][1])
                setNodes = set(Nodes)
                setNodesOfID = set(adeNodes.elements[ID])
                notinList = setNodesOfID - setNodes
                adeNodes.nodeDefinition[notinList.pop()]= deepcopy(Cicalc)

            adeNodes.elementMarkerPositionEstimate[ID]=IdealMesh2idealMeasurement(adeNodes.nodeDefinition[adeNodes.elements[ID][0]],adeNodes.nodeDefinition[adeNodes.elements[ID][1]],adeNodes.nodeDefinition[adeNodes.elements[ID][2]],draw,color='green')

            if draw:
                LabelADE(adeNodes.nodeDefinition[adeNodes.elements[ID][0]],adeNodes.nodeDefinition[adeNodes.elements[ID][1]],adeNodes.nodeDefinition[adeNodes.elements[ID][2]],ID)
  
    if adeNodes == None:
        return (Ai,Bi,Ci,Bound1,Bound2)#,nodes, trigs, strokeLength)
    else:
        if draw:
            for Node in adeNodes.listOfNodes:
                plt.text(adeNodes.nodeDefinition[Node][0,0]+5e-3,adeNodes.nodeDefinition[Node][0,1]-5e-3,str(Node),fontsize=12)

        return adeNodes

def LabelADE(A,B,C,nr,scale = 1):
    """
    Label the ADE in the Plot

    :param A: Point A of ADE
    :type A: np.array

    :param B: Point B of ADE
    :type B: np.array

    :param C: Point C of ADE
    :type C: np.array
    
    :param nr: ID of ADE
    :type nr: int
    """
    if type(A) ==np.matrix:
        pos=CalcPos(A,B)
        # plt.text(pos[0,0],pos[0,1],str(nr)+'1',horizontalalignment='center', verticalalignment='center',fontsize=12)
        # pos=CalcPos(B,C)
        # plt.text(pos[0,0],pos[0,1],str(nr)+'3',horizontalalignment='center', verticalalignment='center',fontsize=12)
        # pos=CalcPos(C,A)
        # plt.text(pos[0,0],pos[0,1],str(nr)+'2',horizontalalignment='center', verticalalignment='center',fontsize=12)
        pos=(A+B+C)/3
        plt.text(pos[0,0],pos[0,1],'ATC'+str(nr),horizontalalignment='center', verticalalignment='center',fontsize=12)
    else:
        # pos=CalcPos(A,B)
        # plt.text(pos[0]+5e-3*scale,pos[1]-5e-3*scale,str(nr)+'1',fontsize=12)
        # pos=CalcPos(B,C)
        # plt.text(pos[0]+5e-3*scale,pos[1]-5e-3*scale,str(nr)+'3',fontsize=12)
        # pos=CalcPos(C,A)
        # plt.text(pos[0]-5e-3*scale,pos[1]+5e-3*scale,str(nr)+'2',fontsize=12)
        pos=(A+B+C)/3
        plt.text(pos[0],pos[1],'ATE'+str(nr),fontsize=12)

def CalcPos(P1,P2):
    """
    Calc Centerpoint between P1 and P2

    :param P1: Point 1
    :type P1: float

    :param P2: Point 2
    :type P2: float

    :return: Centerpoint
    :rtype: float
    """
    n = (P2-P1)/LA.norm(P2-P1)
    pos = (P1+P2)/2+np.multiply(Rot(n),(20e-3))
    return pos

def DrawStrokeTriangle(P1,P2,B1,strokeADE,draw,L3 = 0):
    """
    Draw ADE Triangle with the Points P1, P2, B1 and the Aktor Lengths

    :param P1: Point 1
    :type P1: float

    :param P2: Point 2
    :type P2: float

    :param B1: Boundary Point 1
    :type B1: float

    :param strokeADE: Aktor Lengths
    :type strokeADE: np.array

    :param L1: Parameter L1
    :type L1: float

    :param L3: Parameter L3
    :type L3: float

    :return: Returns Points Ai, Bi and Ci
    :rtype: tuple
    """
    an=(P2-P1)/LA.norm(P2-P1)
    mi=(P2+P1)/2

 
    strokeAB=strokeADE[0]
    strokeBC=strokeADE[2]
    strokeCA=strokeADE[1]

    AB=strokeAB+2*L1
    BC=strokeBC+2*L1
    CA=strokeCA+2*L1

    alpha,beta,gamma=Stroke2angle(BC,CA,AB)

    mAi=np.add(mi,np.multiply(Rot(an),L3)[0:2])
    Ai=P1;
    Bi=Ai+an*(AB);
    Ci=Ai+an*CA*math.cos(alpha)+np.multiply(Rot(an)[0:2],CA)*math.sin(alpha);

    boundi1=Ai+an*(L1-L4)
    boundi2=Bi-an*(L1-L4)

    if draw:
        plt.scatter(Ai[0,0],Ai[0,1],facecolors='red', edgecolors='red')
        plt.scatter(Bi[0,0],Bi[0,1],facecolors='red', edgecolors='red')
        plt.scatter(Ci[0,0],Ci[0,1],facecolors='red', edgecolors='red')

        #plt.scatter(mi[0,0],mi[0,1],facecolors='none', edgecolors='red')
        plt.scatter(boundi1[0,0],boundi1[0,1],facecolors='none', edgecolors='red')
        plt.scatter(boundi2[0,0],boundi2[0,1],facecolors='none', edgecolors='red')
        plt.plot(np.array([boundi1[0,0],boundi2[0,0]]),np.array([boundi1[0,1],boundi2[0,1]]),color='black',linewidth=1)

        PlotTriangle( Ai,Bi,Ci, 'black' )
    return (Ai,Bi,Ci)

def PlotTriangle( A,B,C, color,label='',linestyle='-'):
    """
    Plot the Triangle of the ADE in the Plot

    :param A: Point A of ADE
    :type A: np.array

    :param B: Point B of ADE
    :type B: np.array

    :param C: Point C of ADE
    :type C: np.array

    :param color: Color of the Triangle
    :type color: string
    """
    #plt.figure()
    if type(A) ==np.matrix:
        plt.plot(np.array([A[0,0],B[0,0]]), np.array([A[0,1],B[0,1]]),color=color,linewidth=1,linestyle=linestyle,label=label)
        plt.plot(np.array([B[0,0],C[0,0]]), np.array([B[0,1],C[0,1]]),color=color,linewidth=1,linestyle=linestyle)
        plt.plot(np.array([C[0,0],A[0,0]]), np.array([C[0,1],A[0,1]]),color=color,linewidth=1,linestyle=linestyle)
    else:
        plt.plot(np.array([A[0],B[0]]), np.array([A[1],B[1]]),color=color,linewidth=1,linestyle=linestyle)
        plt.plot(np.array([B[0],C[0]]), np.array([B[1],C[1]]),color=color,linewidth=1,linestyle=linestyle)
        plt.plot(np.array([C[0],A[0]]), np.array([C[1],A[1]]),color=color,linewidth=1,linestyle=linestyle)

def Rot(w):
    """
    :param w:
    :type w:

    :return: 
    :rtype:
    """
    if w.size == 2:
        v=[-w[0,1].copy(),w[0,0].copy()]
    else:
        v=[-w[0,1].copy(),w[0,0].copy(),0]
    return v

def Stroke2angle(a,b,c):
    """
    Calculate the Angle of the ADE corners with the Aktor Lengths

    :param a: Length of Side A
    :type a: float

    :param b: Length of Side B
    :type b: float

    :param c: Length of Side C
    :type c: float

    :return: Returns the Angles (alpha,beta,gamma)
    :rtype: tuple
    """
    alpha=math.acos((c**2+b**2-a**2)/(2*b*c))
    beta=math.acos((a**2+c**2-b**2)/(2*a*c))
    gamma=math.acos((a**2+b**2-c**2)/(2*a*b))
    return (alpha, beta, gamma)

def IdealMesh2idealActuatorPoints(a,b,c,L1,L3,draw):
    """
    Calculate the ideal Actuator Points from ideal Mesh Points

    :param a: Point A of ADE
    :type a: float

    :param b: Point B of ADE
    :type b: float

    :param c: Point C of ADE
    :type c: float

    :param L1: Parameter L1
    :type L1: float

    :param L3: Parameter L3
    :type L3: float

    :return: Returns ideal Actuator Points (as1,as2,bs1,bs2,cs1,cs2)
    :rtype: tuple
    """

    L2=29

    ab=b-a
    ac=c-a
    bc=c-b
    alpha=math.acos(ab.transpose().dot(ac)/(LA.norm(ab)*LA.norm(ac)));
    beta=math.acos(-bc.transpose().dot(ab)/(LA.norm(bc)*LA.norm(ab)));
    gamma=math.acos(ac.transpose().dot(bc)/(LA.norm(ac)*LA.norm(bc)));

    [as1,as2]=CalcPoints(a,b,alpha,L1,L3);
    [bs1,bs2]=CalcPoints(b,c,beta,L1,L3);
    [cs1,cs2]=CalcPoints(c,a,gamma,L1,L3);
    
    if draw:
        plt.scatter(as1[0,0],as1[1,0],facecolors='none', edgecolors='black',linewidths=1)
        plt.scatter(bs1[0,0],bs1[1,0],facecolors='none', edgecolors='black',linewidths=1)
        plt.scatter(cs1[0,0],cs1[1,0],facecolors='none', edgecolors='black',linewidths=1)
        plt.scatter(as2[0,0],as2[1,0],facecolors='none', edgecolors='black',linewidths=1)
        plt.scatter(bs2[0,0],bs2[1,0],facecolors='none', edgecolors='black',linewidths=1)
        plt.scatter(cs2[0,0],cs2[1,0],facecolors='none', edgecolors='black',linewidths=1)

    return (as1,as2,bs1,bs2,cs1,cs2)

def CalcPoints(p1,p2,angle,L1,L3):
    """
    Calculate Points Ps1 and Ps2 from P1 and P2

    :param p1: Point 1
    :type p1: float

    :param p2: Point 2
    :type p2: float

    :param angle: Angle between Aktors
    :type angle: float

    :param L1: Parameter L1
    :type L1: float

    :param L3: Parameter L3
    :type L3: float

    :return: Returns Points (ps1,ps2)
    :rtype: tuple
    """
    n=(p2-p1)/LA.norm(p2-p1);
    ps1=p1+n*L1+np.multiply(Rot(n),L3);
    ps2=p2-n*L1+np.multiply(Rot(n),L3);    
    return (ps1,ps2)

def IdealMesh2idealHinge(a,b,c,L1,L3,draw):
    """
    Calculate ideal Hinge Point from ideal Mesh 

    :param a: Point A of ADE
    :type a: float

    :param b: Point B of ADE
    :type b: float

    :param c: Point C of ADE
    :type c: float

    :param L1: Parameter L1
    :type L1: float

    :param L3: Parameter L3
    :type L3: float

    :param draw: Draw Hinge into Plot
    :type draw: boolean

    :return: Returns ideal Hinge Points (As,Bs,Cs)
    :rtype: tuple
    """

    ab=b-a
    ac=c-a
    bc=c-b

    alpha=math.acos(ab.transpose().dot(ac)/(LA.norm(ab)*LA.norm(ac)))
    beta=math.acos(-bc.transpose().dot(ab)/(LA.norm(bc)*LA.norm(ab)))
    gamma=math.acos(ac.transpose().dot(bc)/(LA.norm(ac)*LA.norm(bc)))


    As=CalcPointHinge(a,b,alpha,L1,L3)
    Bs=CalcPointHinge(b,c,beta,L1,L3)
    Cs=CalcPointHinge(c,a,gamma,L1,L3)

    if draw:
        plt.scatter(As[0,0],As[1,0],color = 'black',linewidth=2)
        plt.scatter(Bs[0,0],Bs[1,0],color = 'black',linewidth=2)
        plt.scatter(Cs[0,0],Cs[1,0],color = 'black',linewidth=2)


    return (As,Bs,Cs)

def CalcPointHinge(p1,p2,angle,L1,L3):
    """
    Calculate Hinge Point from Points p1, p2 and the angle between aktors

    :param p1: Point 1
    :type p1: float

    :param p2: Point 2
    :type p2: float

    :param angle: Angle between Aktors
    :type angle: float

    :param L1: Parameter L1
    :type L1: float

    :param L3: Parameter L3
    :type L3: float

    :return: Hinge Point
    :rtype: float
    """
    phi=math.atan(L3/L1)
    ce=L3/math.sin(phi)
    n=(p2-p1)/LA.norm(p2-p1)
    p=p1+n*(L1+ce*math.cos(angle-phi))+np.multiply(Rot(n),(ce*math.sin(angle-phi)+L3))
    return p

def IdealMesh2idealMeasurement(a,b,c,draw,color='blue'):
    """
    Calculate the ideal Measurement Points from ideal Mesh

    :param a: Point A of ADE
    :type a: float

    :param b: Point B of ADE
    :type b: float

    :param c: Point C of ADE
    :type c: float

    :param L1: Parameter L1
    :type L1: float

    :param L3: Parameter L3
    :type L3: float

    :param draw: Draw Hinge into Plot
    :type draw: boolean

    :return: Returns the ideal Measurement Points (M1,M2,M3)
    :rtype: tuple
    """

    M1=CalcPointMeasurement(a,c)
    M2=CalcPointMeasurement(b,a)
    M3=CalcPointMeasurement(c,b)

    if draw:
        plt.scatter(M1[0,0],M1[0,1],facecolors=color, edgecolors=color,linewidths=1)
        plt.scatter(M2[0,0],M2[0,1],facecolors=color, edgecolors=color,linewidths=1)
        plt.scatter(M3[0,0],M3[0,1],facecolors=color, edgecolors=color,linewidths=1)


    return [M1,M2,M3]

def CalcPointMeasurement(p1,p2):
    """
    Calculate Measurement Point from Points p1, p2 and the angle between aktors

    :param p1: Point 1
    :type p1: float

    :param p2: Point 2
    :type p2: float

    :param angle: Angle between Aktors
    :type angle: float

    :param L1: Parameter L1
    :type L1: float

    :param L3: Parameter L3
    :type L3: float

    :return: Measurement Point
    :rtype: float
    """
    n=(p2-p1)/LA.norm(p2-p1)
    p=p1+n*(L1-L4)+np.asmatrix(np.multiply(Rot(-n),(L3-L5)))       #From ideal Hinge to measurement Point. Parallel to Acctuator: L1+D1. Normal to Acctuator rot(n)*D2
    return p

def Mesh2Topology(trigs):
    n = len(trigs);
    topology = np.zeros((n*3,2))

    for i in range(n):
        for j in range(3):
            topology[i*3+j, :] = np.array([trigs[i, j], trigs[i, (j+1)%3]])
    return topology

def CalcNewADE( ade, L1, L2, L3,draw ):

    adei1 = ade
    ai1=ade[0,1:3].T
    bi1=ade[0,4:6].T
    ci1=ade[0,7:9].T
    
    [ as1,bs1,cs1 ] = IdealMesh2idealHinge( ai1,bi1,ci1,L1,L3,draw );
    as1,bs1,cs1 = as1[np.newaxis],bs1[np.newaxis],cs1[np.newaxis]
    ades1=np.c_[0, as1,0,bs1,0,cs1,0]

    if draw:
        plotTriangle(adei1,'blue')
    
    adeIdealizedGeometry = ade.copy()
    adeRealGeometry = Measured2realGeometry(ades1,adei1, L1, L3,draw);

    ar1=adeRealGeometry[0,1:3].T
    br1=adeRealGeometry[0,4:6].T
    cr1=adeRealGeometry[0,7:9].T
    
    if draw:
        PlotTriangle(adeRealGeometry,'red')   
    
    [actorLength,A2, A2M, A4, A4M, A5, A5M, A6, A6M, A7, A7M,B2, B2M, B4, B4M, B5, B5M, B6, B6M, B7, B7M,C2, C2M, C4, C4M, C5, C5M, C6, C6M, C7, C7M]=Measured2drawHingeNewModel( ades1,adei1, L1, L2, L3,draw ) # ActorLength

    #nodes = np.array([[A6.T],[A7.T],[A2.T],[A4.T],[A5.T],[as1.T],[A5M.T],[A4M.T],[A2M.T],[A7M.T],[A6M.T],[B6.T],[B7.T],[B2.T],[B4.T],[B5.T],[as1.T],[B5M.T],[B4M.T],[B2M.T],[B7M.T],[B6M.T],[C6.T],[C7.T],[C2.T],[C4.T],[C5.T],[as1.T],[C5M.T],[C4M.T],[C2M.T],[C7M.T],[C6M.T]])
    as1,bs1,cs1 = as1[0],bs1[0],cs1[0]
    nodes = np.stack((A6,A7,A2,A4,A5,as1,A5M,A4M,A2M,A7M,A6M,B6,B7,B2,B4,B5,bs1,B5M,B4M,B2M,B7M,B6M,C6,C7,C2,C4,C5,cs1,C5M,C4M,C2M,C7M,C6M))
    #topology = np.array([[1,2],[2,3],[3,4],[4,5],[3,6],[5,6],[6,7],[7,8],[6,9],[8,9],[9,10],[10,11],[12,13],[13,14],[14,15],[15,16],[14,17],[16,17],[17,18],[18,19],[17,20],[19,20],[20,21],[21,22],[23,24],[24,25],[25,26],[25,28],[26,27],[27,28],[28,29],[29,30],[30,31],[28,31],[31,32],[32,33],[4,19],[15,30],[8,26]])
    topology = np.array([[0,1],[1,2],[2,3],[3,4],[2,5],[4,5],[5,6],[6,7],[5,8],[7,8],[8,9],[9,10],[11,12],[12,13],[13,14],[14,15],[13,16],[15,16],[16,17],[17,18],[16,19],[18,19],[19,20],[20,21],[22,23],[23,24],[24,25],[24,27],[25,26],[26,27],[27,28],[28,29],[29,30],[27,30],[30,31],[31,32],[3,18],[14,29],[7,25]])
       # 1 3; %hilfsstäbe
       # 20 22;
       # 14 12;
       # 33 31;
       # 23 25;
       # 11 9  ];

    nodesF = np.array([[3],[4],[19],[20],[14],[15],[30],[31],[25],[26],[8],[9]])

    nodesB = np.array([[3,6,7],[9,6,5],[20,17,16],[14,17,18],[25,28,29],[31,28,27],[3,4,19],[4,19,20],[14,15,30],[15,30,31],[25,26,8],[26,8,9],[2,3,4],[19,20,21],[13,14,15],[32,31,30],[24,25,26],[8,9,10],[1,2,3],[22,21,20],[12,13,14],[33,32,31],[23,24,25],[11,10,9]])
    
    trigs = np.array([[1,2],[2,3],[3,4],[4,5],[3,6],[5,6],[6, 7],[7, 8],[6, 9],[8, 9],[9, 10],[10, 11],[12, 13],[13, 14],[14, 15],[15, 16],[14, 17],[16, 17],[17, 18],[18, 19],[17, 20],[19, 20],[20, 21],[21, 22],[23, 24],[24, 25],[25, 26],[25, 28],[26, 27],[27, 28],[28, 29],[29, 30],[30, 31],[28, 31],[31, 32],[32, 33],[4, 19],[15, 30],[8, 26]])
       # 1 3; %hilfsstäbe
       # 20 22;
       # 14 12;
       # 33 31;
       # 23 25;
       # 11 9  ];
    
    
    actuators=np.array([[ 37],[38],[39]])  
    actuators2=np.array([[1],[2],[3],[22],[23],[24],[13],[14],[15],[33],[35],[36],[25],[26],[27],[10],[11],[12]])
                  #  40;%Hilfsstäbe
                  #  41;
                  #  42; 
                  #  43;
                  #  44;
                  #  45 ];    

    springsA=np.array([[7],[8],[9],[10],[11],[12],[13],[14],[15],[16],[17],[18]])
            
    springsC= np.array([[19],[20],[21],[22],[23],[24]])
          
    return (actorLength, adeIdealizedGeometry, nodes, topology, trigs, nodesF, nodesB, actuators, springsA, springsC,actuators2)

def Measured2realGeometry( adePos,adeIdealizedGeometry, L1, L3, draw ):
    if draw:
        plt.scatter(adePos[0,1],adePos[0,2],color = 'blue',linewidth = 1.5)
        plt.scatter(adePos[0,4],adePos[0,5],color = 'blue',linewidth = 1.5)
        plt.scatter(adePos[0,7],adePos[0,8],color = 'blue',linewidth = 1.5)

    AS=adePos[0,1:3].T
    BS=adePos[0,4:6].T
    CS=adePos[0,7:9].T

    AB=(adeIdealizedGeometry[0,4:6]-adeIdealizedGeometry[0,1:3]).T
    AC=(adeIdealizedGeometry[0,7:9]-adeIdealizedGeometry[0,1:3]).T
    BC=(adeIdealizedGeometry[0,7:9]-adeIdealizedGeometry[0,4:6]).T

    A=adeIdealizedGeometry[0,1:3].T
    B=adeIdealizedGeometry[0,4:6].T
    C=adeIdealizedGeometry[0,7:9].T


    alpha=Anglev1v2(AB,AC)
    beta=Anglev1v2(-BC,AB)
    gamma=Anglev1v2(AC,BC)

    if draw:
        plt.scatter(A[0],A[1],color = 'blue',linewidth = 1)
        plt.scatter(B[0],B[1],color = 'blue',linewidth = 1)
        plt.scatter(C[0],C[1],color = 'blue',linewidth = 1)

    ar=M2r(A,AS,L3,alpha);
    br=M2r(B,BS,L3,beta);
    cr=M2r(C,CS,L3,gamma);

    if draw:
        plt.scatter(ar[0],ar[1],color = 'red',linewidth = 1)
        plt.scatter(br[0],br[1],color = 'red',linewidth = 1)
        plt.scatter(cr[0],cr[1],color = 'red',linewidth = 1)
    ar,br,cr = ar[np.newaxis],br[np.newaxis],cr[np.newaxis]
    adeRealGeometry=np.c_[1, ar,0, br,0, cr,0]
    return adeRealGeometry

def M2r(p,ps,L3,angle):
    pn=(ps-p)/LA.norm(ps-p)
    pr=p+pn*L3/math.sin(angle/2)
    return pr

def Anglev1v2(v1,v2):
    """
    Function to calculate the Angle between two Vectors

    """
    angle = math.acos(v1.transpose().dot(v2)/(LA.norm(v1)*LA.norm(v2)))
    return angle

def MeasurementToMesh(adeNodes,adeIDList,draw):
    """
    Function calculate the Mesh of ADEs from Measured Markerpoints
    """
   
    #MeshPoints = np.asmatrix(np.zeros((len(adeIDList),3)))
    adePointsList = {}
    MeshPoints = {}

    for ID in adeIDList:
        adePointsList[ID] = adeNodes.elementMarkerPosition[ID]
        #NewtonFunction(list(adePointsList[adeIDList[i]]),list(adePointsList[adeIDList[i]]))
        NewNewtonFunction = lambda estimations: NewtonFunction(estimations,measuredPoints = list(adePointsList[ID]))
        MeshPoints[ID] = sp.optimize.newton(NewNewtonFunction, list(adePointsList[ID]))
        if draw:
            PlotTriangle(MeshPoints[ID].squeeze()[0,0:2],MeshPoints[ID].squeeze()[1,0:2],MeshPoints[ID].squeeze()[2,0:2],color='green')

            plt.scatter(adeNodes.elementMarkerPosition[ID][0][0,0],adeNodes.elementMarkerPosition[ID][0][0,1],facecolors='green', edgecolors='green')
            plt.scatter(adeNodes.elementMarkerPosition[ID][1][0,0],adeNodes.elementMarkerPosition[ID][1][0,1],facecolors='green', edgecolors='green')
            plt.scatter(adeNodes.elementMarkerPosition[ID][2][0,0],adeNodes.elementMarkerPosition[ID][2][0,1],facecolors='green', edgecolors='green')

    return MeshPoints

def NewtonFunction(estimations,measuredPoints):
    """
    Newton Funciton to Calculate the Points ABC of an ADE from the MarkerPoints

    """


    estimationA = np.asmatrix(estimations[0])
    estimationB = np.asmatrix(estimations[1])
    estimationC = np.asmatrix(estimations[2])

    estimationsToMeasuredPointsA,estimationsToMeasuredPointsB,estimationsToMeasuredPointsC = IdealMesh2idealMeasurement(estimationA,estimationB,estimationC,False)

    returnValue = [measuredPoints[0]-estimationsToMeasuredPointsA,measuredPoints[1]-estimationsToMeasuredPointsB,measuredPoints[2]-estimationsToMeasuredPointsC]

    return returnValue

def ViconMeasurementToMesh(adeNodes,adeIDList,draw):
    """
    Function calculate the Mesh of ADEs from Measured Viconpoints
    """
   
    #MeshPoints = np.asmatrix(np.zeros((len(adeIDList),3)))
    adePointsList = {}
    MeshPoints = {}

    for ID in adeIDList:
        adePointsList[ID] = adeNodes.elementMarkerPosition[ID]
        #NewtonFunction(list(adePointsList[adeIDList[i]]),list(adePointsList[adeIDList[i]]))
        NewNewtonFunction = lambda estimations: ViconNewtonFunction(estimations,measuredPoints = list(adePointsList[ID]))
        MeshPoints[ID] = sp.optimize.newton(NewNewtonFunction, list(adePointsList[ID]))
        if draw:
            PlotTriangle(MeshPoints[ID].squeeze()[0,0:2],MeshPoints[ID].squeeze()[1,0:2],MeshPoints[ID].squeeze()[2,0:2],color='green')
            plt.scatter(adePointsList[ID][0][0,0],adePointsList[ID][0][0,1],color='green', marker='x',linewidths=1)
            plt.scatter(adePointsList[ID][1][0,0],adePointsList[ID][1][0,1],color='green', marker='x',linewidths=1)
            plt.scatter(adePointsList[ID][2][0,0],adePointsList[ID][2][0,1],color='green', marker='x',linewidths=1)

    return MeshPoints

def ViconNewtonFunction(estimations,measuredPoints):
    """
    Newton Funciton to Calculate the Points ABC of an ADE from the ViconPoints

    """


    estimationA = np.asmatrix(estimations[0])
    estimationB = np.asmatrix(estimations[1])
    estimationC = np.asmatrix(estimations[2])
    #plt.clf()
    estimationsToMeasuredPointsA,estimationsToMeasuredPointsB,estimationsToMeasuredPointsC = ViconIdealMesh2idealMeasurement(estimationA,estimationB,estimationC,False)

    returnValue = [measuredPoints[0]-estimationsToMeasuredPointsA,measuredPoints[1]-estimationsToMeasuredPointsB,measuredPoints[2]-estimationsToMeasuredPointsC]
    #plt.scatter(measuredPoints[0][0,0],measuredPoints[0][0,1],color='green', marker='x',linewidths=1)
    #plt.scatter(measuredPoints[1][0,0],measuredPoints[1][0,1],color='green', marker='x',linewidths=1)
    #plt.scatter(measuredPoints[2][0,0],measuredPoints[2][0,1],color='green', marker='x',linewidths=1)
    #plt.show()

    return returnValue

def ViconIdealMesh2idealMeasurement(a,b,c,draw):
    """
    Calculate the ideal Vicon Measurement Points from ideal Mesh

    :param a: Point A of ADE
    :type a: float

    :param b: Point B of ADE
    :type b: float

    :param c: Point C of ADE
    :type c: float

    :param L1: Parameter L1
    :type L1: float

    :param L3: Parameter L3
    :type L3: float

    :param draw: Draw Hinge into Plot
    :type draw: boolean

    :return: Returns the ideal Measurement Points (M1,M2,M3)
    :rtype: tuple
    """

    M1=ViconCalcPointMeasurement(a,c)
    M2=ViconCalcPointMeasurement(b,a)
    M3=ViconCalcPointMeasurement(c,b)

    if draw:
        plt.scatter(M1[0,0],M1[0,1],facecolors='blue', edgecolors='blue',linewidths=1)
        plt.scatter(M2[0,0],M2[0,1],facecolors='blue', edgecolors='blue',linewidths=1)
        plt.scatter(M3[0,0],M3[0,1],facecolors='blue', edgecolors='blue',linewidths=1)


    return (M1,M2,M3)

def ViconCalcPointMeasurement(p1,p2):
    """
    Calculate Vicon Measurement Point from Points p1, p2 and the angle between aktors

    :param p1: Point 1
    :type p1: float

    :param p2: Point 2
    :type p2: float

    :param angle: Angle between Aktors
    :type angle: float

    :param L1: Parameter L1
    :type L1: float

    :param L3: Parameter L3
    :type L3: float

    :return: Measurement Point
    :rtype: float
    """
    n=(p2-p1)/LA.norm(p2-p1)
    p=p1+n*(L1+4.9e-3)-np.multiply(Rot(n),(2.7e-3))       #From ideal Hinge to Vicon measurement Point. Parallel to Acctuator: L1+D1. Normal to Acctuator rot(n)*D2 (4.9 to first and 14.9 to Second Point)
    
    if False:
        plt.scatter(p2[0,0],p2[0,1],facecolors='red', edgecolors='red',linewidths=1)
        plt.scatter(p[0,0],p[0,1],facecolors='orange', edgecolors='red',linewidths=1)
        plt.scatter(p1[0,0],p1[0,1],facecolors='red', edgecolors='red',linewidths=1)
        plt.plot(np.array([p1[0,0],p2[0,0]]), np.array([p1[0,1],p2[0,1]]),color='red',linewidth=1)
        plt.show()
    return p

def rotateMatrixRow(matrix, n):
    """
    Function Rotate Matrix Rows clockwise

    """

    n=n%matrix.size
    return np.c_[matrix[:,n:],matrix[:,:n]]


if __name__ == "__main__":
    ##+++++++++++++++++++++++++++++++++++++++
    ## Initializing Module
    config.init()
    logging.info('Starting Measurement Module')

    client = wf.WebServerClient()

    adeIDList = []
    measuredCordsOld = []

    ##+++++++++++++++++++++++++++++++++++++++
    ## Open Camera if Available
    if config.camera.available:
        cap = OpenCamera()
    while True:
        ##+++++++++++++++++++++++++++++++++++++++
        ## Checking if to end the Programm
        try:
            if json.loads(client.Get('EndProgram')):
                os._exit(0)
        except:
            pass

        try:
            line = json.loads(client.Get('Lines'))
        except:
            line = 1
        if config.camera.available:
            ##+++++++++++++++++++++++++++++++++++++++
            ## Getting Image from Camera
            image = TakeImage(cap)
            try:
                ##+++++++++++++++++++++++++++++++++++++++
                ## Saving Image
                cv2.imwrite("output\\ImageFolder\\" + str(line+1) + ".jpg",image)
            except:
                pass
        else:
            ##+++++++++++++++++++++++++++++++++++++++
            ## Getting Image from folder
            try:
                image = cv2.imread(config.camera.imageFolder+str(line+1)+'.jpg')
            except:
                logging.warning('Couldnt Load Image')
                pass
        try:
            ##+++++++++++++++++++++++++++++++++++++++
            ## Measure Marker in Image
            measuredCords, circleImg = MeasurePoints(image)
            measuredCords[0] = line+1
            #try:
            #    if type(measuredCordsOld) != list:
            #        ##+++++++++++++++++++++++++++++++++++++++
            #        ## Sorting Markers
            #        measuredCordsOld = np.c_[measuredCordsOld,measuredCords].T
            #        measuredCordsOld = SortMeasuredPoints(measuredCordsOld,execution = 4)
            #        measuredCords = measuredCordsOld[-1]
            #except:
            #    pass

            ##+++++++++++++++++++++++++++++++++++++++
            ## Sending Measured Markers to Webserver
            client.Put('MeasuredCords',df.NumpyToJson(measuredCords))
            client.Put('MeasuredImage',df.ImageToJson(circleImg))
            measuredCordsOld = measuredCords
            #else: 
            #    CalibrateADEParameters(calibrateADE,cap,client)
        except Exception as ExeptionMessage:
            logging.warning(ExeptionMessage)
        #time.sleep(config.measurement.looptime-time.time()%config.measurement.looptime)
