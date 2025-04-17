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
from numpy.core.shape_base import block

#from functions.simulation_functions import SensorValues

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
import pickle
from functions import com_functions as cf
from functions import data_functions as df
from functions import marker_functions as mf
from functions import measurement_functions as mef
from functions import webserver_functions as wf
from functions import visualization_functions as vf

from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import config
import cv2
import time
import json















    


def WriteMeshPointsToFile(MeshPoint, adeNodes): 
    line = ''
    for ID in MeshPoint:
        line += str(ID)
        for i in range(3):
            line += " " + str(MeshPoint[ID][i][0][0]) + " " + str(MeshPoint[ID][i][0][1])
        line += " "
    line += "\n"

    MeshPointFile = open(r"output\\MeshPoints.txt","a") # berechnete Positionen der Drehpunkte aus der Messung
    MeshPointFile.write(line)

    line = ''
    for ID in adeNodes.elementMarkerPosition:
        line += str(ID)
        for i in range(3):
            line += " " + str(adeNodes.elementMarkerPosition[ID][i][0,0]) + " " + str(adeNodes.elementMarkerPosition[ID][i][0,1])
        line += " "
    line += "\n"

    MeshPointFile = open(r"output\\adeNodes_elementMarkerPosition.txt","a") # gemessene Position der Marker
    MeshPointFile.write(line)

    line = ''
    for ID in adeNodes.elementMarkerPositionEstimate:
        line += str(ID)
        for i in range(3):
            line += " " + str(adeNodes.elementMarkerPositionEstimate[ID][i][0,0]) + " " + str(adeNodes.elementMarkerPositionEstimate[ID][i][0,1])
        line += " "
    line += "\n"

    MeshPointFile = open(r"output\\adeNodes_elementMarkerPositionEstimate.txt","a") # ideale Position der Marker
    MeshPointFile.write(line)

    line = ''
    for ID in adeNodes.nodeDefinition:
        line += str(ID)
        line += " " + str(adeNodes.nodeDefinition[ID][0]) + " " + str(adeNodes.nodeDefinition[ID][1])
        line += " "
    line += "\n"

    MeshPointFile = open(r"output\\adeNodes_nodeDefinition.txt","a") # berechnete Positionen der idealen Drehpunkte
    MeshPointFile.write(line)

def MoveSensorValues(sensorValues,boundaryPoints,line,ViconMode = False):
        P1=boundaryPoints[0,0:2].T
        P2=boundaryPoints[0,3:5].T

        if LA.norm(P2) < LA.norm(P1):
            P1,P2 = P2,P1

        Bound1=P1.T
        Bound2=P2.T#P1+(P2-P1)/LA.norm(P2-P1)*(B1-29)
        an = (P2-P1)/LA.norm(P2-P1)
        
        if not ViconMode:
            rotAngle = 0
            P1=P1-np.matmul(np.matrix([[math.cos(rotAngle*2*math.pi/360),-math.sin(rotAngle*2*math.pi/360)],[math.sin(rotAngle*2*math.pi/360),math.cos(rotAngle*2*math.pi/360)]]),an)*(config.lengths.L1-config.lengths.L4)+np.matmul(np.array([[math.cos((rotAngle+90)*2*math.pi/360),-math.sin((rotAngle+90)*2*math.pi/360)],[math.sin((rotAngle+90)*2*math.pi/360),math.cos((rotAngle+90)*2*math.pi/360)]]),an)*(config.lengths.L3-config.lengths.L5+1)
            
            ex = np.array([[1],[0]])
            bound = P2-P1
            alpha = math.acos(ex.T.dot(bound)/(LA.norm(ex.T)*LA.norm(bound)))
            

        else:
            rotAngle = 0
            

            P1=P1-np.matmul(np.matrix([[math.cos(rotAngle*2*math.pi/360),-math.sin(rotAngle*2*math.pi/360)],[math.sin(rotAngle*2*math.pi/360),math.cos(rotAngle*2*math.pi/360)]]),an)*(config.lengths.L1+4.9)+np.matmul(np.array([[math.cos((rotAngle+90)*2*math.pi/360),-math.sin((rotAngle+90)*2*math.pi/360)],[math.sin((rotAngle+90)*2*math.pi/360),math.cos((rotAngle+90)*2*math.pi/360)]]),an)*(2.7)
            P2=P1+np.matmul(np.matrix([[math.cos(rotAngle*2*math.pi/360),-math.sin(rotAngle*2*math.pi/360)],[math.sin(rotAngle*2*math.pi/360),math.cos(rotAngle*2*math.pi/360)]]),an)*(152)

           
            ex = np.array([[1],[0]])
            bound = P2-P1
            alpha = math.acos(ex.T.dot(bound)/(LA.norm(ex.T)*LA.norm(bound)))


        P1=P1.T

        
        for ID in sensorValues[line]:
            for i in range(3):
                sensorValues[line][ID][i] = np.matmul(mef.RotAz(alpha),np.matrix(sensorValues[line][ID][i].T)).T+P1/1000
    
def CalculateActorLengths(meshPoints):
    lengths = {}
    for ID in meshPoints:
        A = LA.norm(meshPoints[ID][0] -meshPoints[ID][1])-config.lengths.L0
        C = LA.norm(meshPoints[ID][1] -meshPoints[ID][2])-config.lengths.L0
        B = LA.norm(meshPoints[ID][2] -meshPoints[ID][0])-config.lengths.L0

        lengths[ID] = [A,B,C]

    return lengths

def AverageMeshPoints(meshPoints,adeNodes,draw = True):
    averageMeshPoints = {}
    for ID in adeNodes.elements:
        if adeNodes.elements[ID][0] in averageMeshPoints:
            averageMeshPoints[adeNodes.elements[ID][0]] = (averageMeshPoints[adeNodes.elements[ID][0]]+meshPoints[ID][0])/2
        else:
            averageMeshPoints[adeNodes.elements[ID][0]] = meshPoints[ID][0]
        if adeNodes.elements[ID][1] in averageMeshPoints:
            averageMeshPoints[adeNodes.elements[ID][1]] = (averageMeshPoints[adeNodes.elements[ID][1]]+meshPoints[ID][1])/2
        else:
            averageMeshPoints[adeNodes.elements[ID][1]] = meshPoints[ID][1]
        if adeNodes.elements[ID][2] in averageMeshPoints:
            averageMeshPoints[adeNodes.elements[ID][2]] = (averageMeshPoints[adeNodes.elements[ID][2]]+meshPoints[ID][2])/2
        else:
            averageMeshPoints[adeNodes.elements[ID][2]] = meshPoints[ID][2]

        pass
    returnmeshPoints={}
    for ID in adeNodes.elements:
        returnmeshPoints[ID] = [averageMeshPoints[adeNodes.elements[ID][0]],averageMeshPoints[adeNodes.elements[ID][1]],averageMeshPoints[adeNodes.elements[ID][2]]]

        if draw:
            plt.scatter(returnmeshPoints[ID][0][0,0],returnmeshPoints[ID][0][0,1],facecolors='red', edgecolors='red')
            plt.scatter(returnmeshPoints[ID][1][0,0],returnmeshPoints[ID][1][0,1],facecolors='red', edgecolors='red')
            plt.scatter(returnmeshPoints[ID][2][0,0],returnmeshPoints[ID][2][0,1],facecolors='red', edgecolors='red')

    return returnmeshPoints




def AverageMeshPointsOverElementDefinitionTime(idealMesh,adeNodes,adeMoveList,elementDefinition=True):  
    adeIDs = []
    TimeStamp=0
    if type(adeNodes) == dict:
        for ID in adeNodes[0].elements:
                adeIDs.append(ID)
    else:
        for ID in adeNodes.elements:
                adeIDs.append(ID)
                
    for line in range(len(adeMoveList.data)):
    
        if elementDefinition:
            nextTimeStamp = False 
            if line >= 0 and line < len(adeMoveList.data)-1:
                ##+++++++++++++++++++++++++++++++++++++++
                ## Checking if Connections changed with the MoveList
                for j in range(len(adeIDs)):
                    index = adeMoveList.GetIndexOfID(adeIDs[j])
                    if index != None:
                        if config.simulation.connectionInfoInMoveList:
                            if adeMoveList.data[line,index+4] != adeMoveList.data[line+1,index+4] or adeMoveList.data[line,index+5] != adeMoveList.data[line+1,index+5] or adeMoveList.data[line,index+6] != adeMoveList.data[line+1,index+6]:
                                nextTimeStamp = True
                        else:
                            if  adeMoveList.data[line+1,index+4] != 0 or adeMoveList.data[line+1,index+5] != 0 or adeMoveList.data[line+1,index+6] != 0: #for 6ADEs
                                nextTimeStamp = True
    
            for time in idealMesh[line]:
                idealMesh[line][time] = AverageMeshPoints(idealMesh[line][time],adeNodes[TimeStamp],draw=False)
        
            if nextTimeStamp == True:
                TimeStamp += 1    
        else: 
            for time in idealMesh[line]:
                idealMesh[line][time] = AverageMeshPoints(idealMesh[line][time],adeNodes,draw=False)
            
    return idealMesh


def AverageMeshPointsOverElementDefinition(idealMesh,adeNodes,adeMoveList,elementDefinition=True):  
    adeIDs = []
    TimeStamp=0
    if type(adeNodes) == dict:
        for ID in adeNodes[0].elements:
                adeIDs.append(ID)
    else:
        for ID in adeNodes.elements:
                adeIDs.append(ID)
    
    for line in range(len(adeMoveList.data)):

        if elementDefinition:
            nextTimeStamp = False 
            if line >= 0 and line < len(adeMoveList.data)-1:
                ##+++++++++++++++++++++++++++++++++++++++
                ## Checking if Connections changed with the MoveList
                for j in range(len(adeIDs)):
                    index = adeMoveList.GetIndexOfID(adeIDs[j])
                    if index != None:
                        if config.simulation.connectionInfoInMoveList:
                            if adeMoveList.data[line,index+4] != adeMoveList.data[line+1,index+4] or adeMoveList.data[line,index+5] != adeMoveList.data[line+1,index+5] or adeMoveList.data[line,index+6] != adeMoveList.data[line+1,index+6]:
                                nextTimeStamp = True
                        else:
                            if  adeMoveList.data[line+1,index+4] != 0 or adeMoveList.data[line+1,index+5] != 0 or adeMoveList.data[line+1,index+6] != 0: #for 6ADEs
                                nextTimeStamp = True
        
            idealMesh[line] = AverageMeshPoints(idealMesh[line],adeNodes[TimeStamp],draw=False)
        
            if nextTimeStamp == True:
                TimeStamp += 1    
        else: 
            for time in idealMesh[line]:
                idealMesh[line] = AverageMeshPoints(idealMesh[line],adeNodes,draw=False)    
    return idealMesh

def CalculateError(currentLine,lengths,previousError = None):
    errorLengths = {}
    if type(previousError) == dict:
        for ID in lengths:
            index = currentLine.GetIndexOfID(ID)

            errorA = currentLine.data[0,index+1]/10000 - lengths[ID][0] - previousError[ID][0]
            errorB = currentLine.data[0,index+2]/10000 - lengths[ID][1] - previousError[ID][1]
            errorC = currentLine.data[0,index+3]/10000 - lengths[ID][2] - previousError[ID][2]

            errorLengths[ID] = [errorA,errorB,errorC]
    else:
        for ID in lengths:
            index = currentLine.GetIndexOfID(ID)

            errorA = currentLine.data[0,index+1]/10000 - lengths[ID][0]
            errorB = currentLine.data[0,index+2]/10000 - lengths[ID][1]
            errorC = currentLine.data[0,index+3]/10000 - lengths[ID][2]

            errorLengths[ID] = [errorA,errorB,errorC]
    return errorLengths

def CalculateNextMovementCommands(adeMoveList,line,errorLengths,errorCorrection= True):
    nextLine = df.adeMoveList()
    nextLine.data = adeMoveList.data[line+1,:]
    if errorCorrection:
        for ID in errorLengths:
            index = nextLine.GetIndexOfID(ID)

            nextLine.data[0,index+1] += int(errorLengths[ID][0]*10000)
            nextLine.data[0,index+2] += int(errorLengths[ID][1]*10000)
            nextLine.data[0,index+3] += int(errorLengths[ID][2]*10000)

            for i in range(3):
                if nextLine.data[0,index+1+i] < 0:
                    nextLine.data[0,index+1+i] = 0
                elif nextLine.data[0,index+1+i] > 1000:
                    nextLine.data[0,index+1+i] = 1000
    return nextLine

if __name__ == "__main__":
    ##+++++++++++++++++++++++++++++++++++++++
    ## Initialize Module
    config.init()
    logging.info('Starting Control Module')

    client = wf.WebServerClient()
    adeIDList = []
    line = 0
    offsetAngle = 0
    while True:
        startControl = False
        #while not startControl:
        try:
            ##+++++++++++++++++++++++++++++++++++++++
            ## Checking if to end the Programm
            if json.loads(client.Get('EndProgram')):
                os._exit(0)

            ##+++++++++++++++++++++++++++++++++++++++
            ## Getting adeNodes(Mesh) from Webserver
            adeIDList = []
            sensorValues = {}
            meshPoints = {}
            adeNodes =  pickle.loads(client.Get('adeNodes'))
            TimeStamp = 0
            if type(adeNodes) == dict:
                adeNodesDict = deepcopy(adeNodes)
                adeNodes = deepcopy(adeNodesDict[TimeStamp])
            for ID in adeNodes.elements:
                    adeIDList.append(ID)

            ##+++++++++++++++++++++++++++++++++++++++
            ## Getting Flags from Webserver
            startControl = json.loads(client.Get('StartControl'))
            startSimulation = json.loads(client.Get('StartSimulation'))

            ##+++++++++++++++++++++++++++++++++++++++
            ## Starting Control
            if startControl:

                if config.simulation.useMoveList:
                    ##+++++++++++++++++++++++++++++++++++++++
                    ## Getting MoveList from Webserver
                    logging.debug('Getting MoveList from Webserver')
                    adeMoveList = df.adeMoveList()
                    adeMoveList.FromJson(client.Get('MoveList'))
                    adeMoveList.ConvertConnectionInfoToMagnetCommands()

                    try:
                        os.remove("output\\MeshPoints.txt")
                        os.remove("output\\adeNodes_elementMarkerPosition.txt")
                        os.remove("output\\adeNodes_elementMarkerPositionEstimate.txt")
                        os.remove("output\\adeNodes_nodeDefinition.txt")
                    except:
                        pass
                    
                    currentLine = df.adeMoveList()
                    previousError = None

                    for line in range(adeMoveList.data.shape[0]):
                        
                        try:
                            for j in range(len(adeIDList)):
                                index = adeMoveList.GetIndexOfID(adeIDList[j])
                                if index != None:
                                    if  adeMoveList.data[line,index+4] != 0 or adeMoveList.data[line,index+5] != 0 or adeMoveList.data[line,index+6] != 0: #for 6ADEs
                                        nextTimeStamp = True
                                    
                                if nextTimeStamp == True:
                                    nextTimeStamp = False
                                    TimeStamp += 1
                                    adeNodes = deepcopy(adeNodesDict[TimeStamp])
                        except:
                            pass
                        ##+++++++++++++++++++++++++++++++++++++++
                        ##Itterating through MoveList
                        client.Put('runADE',json.dumps(False))
                        client.Put('Lines',json.dumps(line))

                        if line > 0:
                            currentLine = nextLine
                        else:
                            currentLine.data = adeMoveList.data[line,:]

                        ##+++++++++++++++++++++++++++++++++++++++
                        ## Sending Next Movement Commands to Webserver
                        logging.debug('Line '+str(line)+': Sending Next Movement Commands to Webserver')

                        client.Put('MovementCommands',currentLine.ToJson())
                        client.Put('runADE',json.dumps(True))
                        ##+++++++++++++++++++++++++++++++++++++++
                        ## Waiting for ADEs to Complete Drive
                        logging.debug('Line '+str(line)+': Waiting for ADEs to Complete Drive')
                        if line > 0:
                            #time.sleep(cf.CalcPause(adeMoveList,line)+config.control.pauseValue)
                            time.sleep(0.5)
                        else:
                            #time.sleep(config.control.pauseValue)

                            time.sleep(5)


                        draw = True
                        if draw and line > 0:
                            plt.clf()
                        ##+++++++++++++++++++++++++++++++++++++++
                        ## Getting Measured Coordinates from Webserver
                        while True:
                            logging.debug('Line '+str(line)+': Getting Measured Coordinates from Webserver')

                            measuredCords = df.NumpyFromJson(client.Get('MeasuredCords'))
                            image = df.ImageFromJson(client.Get('MeasuredImage'))

                            if measuredCords.shape[1] != ((len(adeIDList)*3+2)*3+1):
                                logging.warning('Line '+str(line)+': Not enough Markers detectet')
                                if True:
                                    cv2.imshow("Detected Circle", image)
                                    cv2.waitKey(0)
                            else:
                                break


                        ##+++++++++++++++++++++++++++++++++++++++
                        ## Setting Boundary Points
                        if line == 0:
                            logging.debug('Line '+str(line)+': Setting Boundary Points')
                            boundaryPoints = mef.SortPointsADEBoundary(measuredCords[0,:], False ,"MeasuredPoints")[1:3]
                            #boundaryPoints = (mef.SortPointsADEBoundary(measuredCords[0,:], False ,"MeasuredPoints")[1],mef.SortPointsADEBoundary(measuredCords[0,:], False ,"MeasuredPoints")[3])
                            if LA.norm(boundaryPoints[1]) < LA.norm(boundaryPoints[0]):
                                boundaryPoints = np.c_[boundaryPoints[1],boundaryPoints[0]]
                            else:
                                boundaryPoints = np.c_[boundaryPoints[0],boundaryPoints[1]]
                            boundaryPoints[0,4] = boundaryPoints[0,1]

                        
                            
                        if not startSimulation: 
                            ##+++++++++++++++++++++++++++++++++++++++
                            ## Calculating expected Marker Positions
                            mef.Stroke2desiredMesh(adeIDList,boundary = boundaryPoints,strokeLength= currentLine,adeNodes = adeNodes,draw = True,figureName = "MeasuredPoints")
                            for ID in adeNodes.elements:
                                adeNodes.elementMarkerPositionEstimate[ID]=mef.IdealMesh2idealMeasurement(adeNodes.nodeDefinition[adeNodes.elements[ID][0]],adeNodes.nodeDefinition[adeNodes.elements[ID][1]],adeNodes.nodeDefinition[adeNodes.elements[ID][2]],True)

                                
                        else:
                            while True:
                                ##+++++++++++++++++++++++++++++++++++++++
                                ## Getting/Waiting for the Sensor Variables of corresponding MoveList Line
                                sensorValues = pickle.loads(client.Get('SensorValues'))
                                if line in sensorValues.keys():
                                    logging.debug('Line '+str(line)+': Recived Simulation Data')
                                    break
                                else:
                                    time.sleep(config.control.looptime-time.time()%config.control.looptime)
                                
                            ##+++++++++++++++++++++++++++++++++++++++
                            ## Moving Sensor Variable Points over Measured Points
                            #MoveSensorValues(sensorValues,boundaryPoints,line+1)
                            #client.Put('SensorValues',pickle.dumps(sensorValues))
                            for ID in adeIDList:
                                adeNodes.elementMarkerPositionEstimate[ID]= mef.IdealMesh2idealMeasurement(sensorValues[line][ID][0],sensorValues[line][ID][1],sensorValues[line][ID][2],False)
                            
                        ##+++++++++++++++++++++++++++++++++++++++
                        ## Sorting Measured Variable Points into adeNodes.elementMarkerPosition
                        logging.debug('Line '+str(line)+': Sorting Markers')
                        sortByMarkerIndex = False
                        if line == 0:
                            MarkerIndex = mef.SortMeasuredPoints( measuredCords, boundary = boundaryPoints,execution = 3, draw = False,adeNodes = adeNodes,sortByAngle = True)

                        elif sortByMarkerIndex:
                            P1=boundaryPoints[:,0:2]
                            P2=boundaryPoints[:,3:-1]
                            for i in range(int((measuredCords[0,:].shape[1]-1)/3)):  
                                measuredCords[:,i*3+1:i*3+1+2]=mef.SetInitialADE(measuredCords[:,i*3+1:i*3+1+2].copy(),P1,P2)
                            for ID in adeIDList:
                                for i in range(3):
                                    adeNodes.elementMarkerPosition[ID][i][:] = measuredCords[0,MarkerIndex[ID][i]:MarkerIndex[ID][i]+2].copy()
                        else:
                            mef.SortMeasuredPoints( measuredCords, boundary = boundaryPoints,execution = 3, draw = False,adeNodes = adeNodes, sortByAngle = True)

                        ##+++++++++++++++++++++++++++++++++++++++
                        ## Calculation ADE Node Positions
                        logging.debug('Line '+str(line)+': Calculation ADE Node Positions')
                        try:
                            meshPoints[line] = mef.MeasurementToMesh(adeNodes,adeIDList,False)
                        except Exception as ExeptionMessage:
                            logging.warning(ExeptionMessage)
                            continue

                        ##+++++++++++++++++++++++++++++++++++++++
                        ## Sending meshPoints to Webserver
                        client.Put('MeshPoints',pickle.dumps(meshPoints))

                        #WriteMeshPointsToFile(MeshPoints, adeNodes)

                        ##+++++++++++++++++++++++++++++++++++++++
                        ## Averaging Mesh Points and Calculating Error
                        logging.debug('Line '+str(line)+': Averaging Mesh Points and Calculating Error')
                        #meshPoints[line] = AverageMeshPoints(meshPoints[line],adeNodes,draw=False)
                        #avgSensorValues = sensorValues
                        #avgSensorValues[line] = AverageMeshPoints(sensorValues[line],adeNodes,draw=False)
                        lengths = CalculateActorLengths(meshPoints[line])
                        currentLine.data = adeMoveList.data[line,:]
                        errorLengths = CalculateError(currentLine,lengths,previousError)

                        ##+++++++++++++++++++++++++++++++++++++++
                        ## Calculating next Movement Commands
                        logging.debug('Line '+str(line)+': Calculating next Movement Commands')
                        if line+1 < adeMoveList.data.shape[0]:
                            nextLine = CalculateNextMovementCommands(adeMoveList,line,errorLengths,errorCorrection = False)

                        previousError = deepcopy(errorLengths)
                        if draw:
                            ##+++++++++++++++++++++++++++++++++++++++
                            ## Plotting Mesh over Measured Image
                            logging.info('Line '+str(line)+':  Plotting Mesh over Measured Image')
                            #plt.axis('equal')
                            #mef.Stroke2desiredMesh(adeIDList,boundary = np.matrix([[0,-config.lengths.boundaryYdistance,0,0.01,-config.lengths.boundaryYdistance,0]]),strokeLength= currentLine,adeNodes = adeNodes,draw = False)
                            currentAvgMeshPoints = {}
                            currentAvgMeshPoints[0] = meshPoints[line]
                            currentAvgSensorValues = {}
                            currentAvgSensorValues[0] = sensorValues[line]
                            currentIdealMesh = {}
                            currentIdealMesh[0] = {}
                            #for ID in adeNodes.elements:
                            #    currentIdealMesh[0][ID] = [adeNodes.nodeDefinition[adeNodes.elements[ID][0]],adeNodes.nodeDefinition[adeNodes.elements[ID][1]],adeNodes.nodeDefinition[adeNodes.elements[ID][2]]]
                            
                            
                            vf.PlotSensorValues(currentAvgMeshPoints,linecolor = 'blue',markercolor = 'blue',marker = 'o',figureName = '',combinePlots=True,labelATC = False)
                            vf.PlotSensorValues(currentAvgSensorValues,linecolor = 'orange',markercolor = 'orange',marker = 'x',figureName = '',combinePlots=True,labelATC = False)
                            #vf.PlotSensorValues(currentIdealMesh,markercolor = 'red',marker = '^',figureName = '',combinePlots=True)
                            #image = cv2.resize(image, (config.measurement.xCord+1,config.measurement.yCord))
                            #image = np.flipud(image)
                            #P1=boundaryPoints[:,0:2]-[config.lengths.boundaryXdistance,config.lengths.boundaryYdistance]
                            #P2=boundaryPoints[:,3:-1]-[config.lengths.boundaryXdistance,config.lengths.boundaryYdistance]
                            #an = ((P2-P1)/LA.norm(P2-P1)).T
                            #imageRotationMatrix = cv2.getRotationMatrix2D((P1[0,0],P1[0,1]),(-(math.acos(an[0,0]))*180/math.pi-1),1)
                            #image = cv2.warpAffine(image,imageRotationMatrix,(image.shape[1],image.shape[0])) 


                            translationOffsetX = 0
                            translationOffsetY = 0
                            #
                            #
                            #translationOffsetX = 0.025
                            #translationOffsetY = -0.025
                            plt.imshow(image,extent=[-boundaryPoints[0,0]-config.lengths.boundaryXdistance+translationOffsetX, config.measurement.xCord-boundaryPoints[0,0]-config.lengths.boundaryXdistance+translationOffsetX, config.measurement.yCord -boundaryPoints[0,1]-config.lengths.boundaryYdistance+translationOffsetY, -boundaryPoints[0,1]-config.lengths.boundaryYdistance+translationOffsetY])
                            plt.gca().invert_yaxis()

                            #plt.rcParams.update({'font.size': 22})
                            plt.ylim([-boundaryPoints[0,1]-config.lengths.boundaryYdistance,config.measurement.yCord -boundaryPoints[0,1]-config.lengths.boundaryYdistance])
                            plt.xlim([-boundaryPoints[0,0]-config.lengths.boundaryXdistance,config.measurement.xCord -boundaryPoints[0,0]-config.lengths.boundaryXdistance])
                            
                            plt.gcf().set_size_inches(15/2.54, 15/2.54)
                            #plt.xlabel('mm')
                            #plt.ylabel('mm')
                            
                            
                            plt.savefig(config.camera.imageFolder + str(line+1) + "_Mesh.jpg",dpi = 300)
                            plt.show(block=False)
                            plt.pause(0.5)
                            
                        if True:  
                            WriteMeshPointsToFile(meshPoints[line], adeNodes)
                
                startControl = False
                client.Put('StartControl',json.dumps(startControl))
                
        except Exception as ExeptionMessage:

            logging.warning(ExeptionMessage)
            logging.warning('Error in Line ' + str(line))

            startControl = False
            #client.Put('StartControl',json.dumps(startControl))
        
        time.sleep(config.control.looptime-time.time()%config.control.looptime)

        