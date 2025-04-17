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
import pickle
import sys

if __name__ == "__main__":
    if os.getcwd().split('\\')[-1].__contains__('functions'):
        os.chdir('..')
    if sys.path[0].split('\\')[-1].__contains__('functions'):
        sys.path.append(os.getcwd())
    

import config

from functions import com_functions as cf
from functions import data_functions as df
from functions import marker_functions as mf
from functions import measurement_functions as mef




#import pickle
import logging
import requests
import json
import cv2
import time
import numpy as np
from copy import deepcopy
from flask import Flask, request, jsonify, render_template
from datetime import datetime
import os


config.init()

class WebServerClient:
    """
    WebServer client to Get and Put Data from Webserver
    """
    def __init__(self):
        """
        Initialize WebServerClient with WebServer Address
        """
        self.Address = 'http://127.0.0.1:5000'
    def Put(self,name,data):
        """
        PUT Data to Webserver
        """
        try:
            response = requests.put(self.Address+'/'+name, data)
            return response.content
        except:
            logging.warning('No Connection to Webserver')
            raise ConnectionError
        
    def Get(self,name,ID = ''):
        """
        GET Data from Webserver
        """
        try:
            response = requests.get(self.Address+'/'+name)
            return response.content
        except:
            logging.warning('No Connection to Webserver')
            raise ConnectionError


template_dir = os.path.abspath('./html')
app = Flask(__name__,template_folder = template_dir)

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Main Index:
@app.route('/', methods=['GET'])
@app.route('/home')
def index(ID=None):
    """
    Test Function
    """
    detectedMarkersdir= os.path.abspath('./html/img/detectedCircles.jpg')
    detectedMarkersPlotdir=   os.path.abspath('./html/img/detectetMarkerPlot.jpg')
    return render_template(
        'index.html',
        title='Home Page',
        adeIDList = runADE,
        year=datetime.now().year,
        detectedMarkers = detectedMarkersdir,
        detectedMarkersPlot = detectedMarkersPlotdir,
    )

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# System APIs:

@app.route('/EndProgram', methods=['GET'])
def GetEndProgram():
    """
    GET Battery Status
    """
    return jsonify(endProgram)

@app.route('/EndProgram', methods=['PUT'])
def SetEndProgram():
    """
    GET Battery Status
    """
    response = request.data
    global endProgram
    endProgram = json.loads(response)
    if endProgram == True:
        time.sleep(0.5)
        os._exit(0)
    return jsonify(endProgram)

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Comunication APIs:

@app.route('/BatteryStatus', methods=['GET'])
def GetBatteryStatus():
    """
    GET Battery Status
    """
    return jsonify(batteryStatus)

@app.route('/BatteryStatus', methods=['PUT'])
def SetBatteryStatus():
    """
    GET Battery Status
    """
    response = request.data
    global batteryStatus
    batteryStatus = json.loads(response)
    return jsonify(batteryStatus)

@app.route('/BatteryStatus/<int:ID>', methods=['GET'])
def GetBatteryStatusID(ID):
    """
    GET Battery Status
    """
    return jsonify(batteryStatus[ID])

@app.route('/BatteryStatus/<int:ID>',methods=['PUT'])
def SetBatteryStatusID(ID):
    """
    PUT Battery Status
    """
    global batteryStatus
    batteryStatus[ID] = json.loads(request.data)
    return jsonify(batteryStatus[ID])

@app.route('/adeIDList',methods=['GET'])
def GetAdeIDList():
    """
    Get ADE ID List
    """
    return jsonify(adeIDList)

@app.route('/adeIDList',methods=['PUT'])
def PutAdeIDList():
    """
    Put ADE ID List
    """
    response = request.data
    global adeIDList
    global runADE
    newadeIDList = json.loads(response)
    if adeIDList != newadeIDList:
        for ID in newadeIDList:
            if ID not in adeIDList:
                adeIDList.append(ID)
                runADE[ID] = False
        aux = adeIDList.copy()
        for ID in aux:
            if ID not in newadeIDList:
                adeIDList.remove(ID)
                runADE.pop(ID,None)

                        #runADE.remove(ID)
    
    return jsonify(adeIDList)

@app.route('/runADE',methods=['GET'])
def GetrunADE():
    """
    Get Run ADE Flag
    """
    return jsonify(runADE)

@app.route('/runADE',methods=['PUT'])
def PutrunADE():
    """
    Put ADE ID List
    """
    response = json.loads(request.data)
    global runADE
    for ade in adeIDList:
        runADE[ade] = response
    return jsonify(runADE)

@app.route('/runADE/<int:ID>',methods=['GET'])
def GetrunADEID(ID):
    """
    Get Run ADE Flag
    """
    return jsonify(runADE[ID])

@app.route('/runADE/<int:ID>',methods=['PUT'])
def PutrunADEID(ID):
    """
    Put ADE ID List
    """
    response = request.data
    global runADE
    runADE[ID] = json.loads(response)
    return jsonify(runADE)

@app.route('/MovementCommands', methods=['GET'])
def GetMovementCommands():
    """
    GET MovementCommands
    """
    returndata = deepcopy(movementCommands)
    returndata.data = returndata.data[-1,:]
    return returndata.ToJson()

@app.route('/MovementCommands', methods=['PUT'])
def PutMovementCommands():
    """
    PUT MovementCommands
    """
    response = request.data
    global movementCommands
    movementCommands.data =  df.NumpyFromJson(response)
    return response 

@app.route('/MoveList', methods=['GET'])
def GetMoveList():
    """
    GET MoveList
    """
    returndata = deepcopy(adeMoveList)
    return returndata.ToJson()

@app.route('/MoveList', methods=['PUT'])
def PutMoveList():
    """
    PUT MoveList
    """
    response = request.data
    global adeMoveList
    adeMoveList.data =  df.NumpyFromJson(response)
    return response 

@app.route('/Lines', methods=['GET'])
def GetLines():
    """
    GET Lines
    """
    return json.dumps(currentLine)

@app.route('/Lines', methods=['PUT'])
def PutLines():
    """
    PUT Lines
    """
    response = request.data
    lines = json.loads(response)

    global currentLine
    global movementCommands
    global adeMoveList

    if type(lines) == int:
        currentLine= lines
        try:
            movementCommands.data = adeMoveList.data[currentLine,:]
        except:
            return json.dumps('Line '+str(currentLine)+' out of range')
    elif type(lines) == list:
        startLine = lines[0]
        endLine = lines[1]
        for currentLine in range(startLine,endLine):
            try :
                movementCommands.data = adeMoveList.data[currentLine,:]
                time.sleep(config.webserver.pauseBetweenMovementCommands-time.time()%config.webserver.pauseBetweenMovementCommands)
            except:
                return json.dumps('Line '+str(currentLine)+' out of range')
    return response 

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Measurement Module API:

@app.route('/MeasuredCords', methods=['GET'])
def GetMeasuredCords():
    """
    GET Measured Coordinates
    """
    return df.NumpyToJson(measuredCords)

@app.route('/MeasuredCords', methods=['PUT'])
def PutMeasuredCords():
    """
    PUT Measured Coordinates
    """
    response = request.data
    global measuredCords
    measuredCords = df.NumpyFromJson(response)
    return response

@app.route('/MeasuredImage', methods=['GET'])
def GetMeasuredImage():
    """
    GET Measured Image
    """
    return df.ImageToJson(measuredImage)

@app.route('/MeasuredImage', methods=['PUT'])
def PutMeasuredImage():
    """
    PUT Measured Image
    """
    response = request.data
    global measuredImage
    measuredImage = df.ImageFromJson(response)
    cv2.imwrite('html/img/detectedCircles.jpg',measuredImage)
    return response

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Gesture Module API:

@app.route('/PoseID', methods=['GET'])
def GetPoseID():
    """
    GET Pose ID
    """
    return json.dumps(poseID)

@app.route('/PoseID', methods=['PUT'])
def PutPoseID():
    """
    PUT Pose ID
    """
    response = request.data
    global poseID
    poseID = df.json.loads(response)
    #if PoseID != 'x':
    #    global gestureMode
    #    gestureMode = None
    return response

@app.route('/GestureMode', methods=['GET'])
def GetGestureMode():
    """
    GET Gesture Mode
    """
    return json.dumps(gestureMode)

@app.route('/GestureMode', methods=['PUT'])
def PutGestureMode():
    """
    PUT Gesture Mode
    """
    response = request.data
    global gestureMode
    gestureMode = df.json.loads(response)
    return response

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Simulation Module API:

@app.route('/StartSimulation', methods=['GET'])
def GetStartSimulation():
    """
    GET Gesture Mode
    """
    return json.dumps(startSimulation)

@app.route('/StartSimulation', methods=['PUT'])
def PutStartSimulation():
    """
    PUT Gesture Mode
    """
    response = request.data
    global startSimulation
    startSimulation = df.json.loads(response)
    return response

@app.route('/adeNodes',methods=['GET'])
def GetAdeNodes():
    """
    Get ADE Nodes
    """
    return pickle.dumps(adeNodes)

@app.route('/adeNodes',methods=['PUT'])
def PutAdeNodes():
    """
    Put ADE Nodes
    """
    global adeNodes
    adeNodes = pickle.loads(request.data)

    return pickle.dumps(adeNodes)

@app.route('/SensorValues', methods=['GET'])
def GetSensorValues():
    """
    GET Gesture Mode
    """
    return pickle.dumps(sensorValues)

@app.route('/SensorValues', methods=['PUT'])
def PutSensorValues():
    """
    PUT Gesture Mode
    """
    response = request.data
    global sensorValues
    sensorValues = pickle.loads(response)
    return response

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Control Module API:

@app.route('/StartControl', methods=['GET'])
def GetStartControl():
    """
    GET Gesture Mode
    """
    return json.dumps(startControl)

@app.route('/StartControl', methods=['PUT'])
def PutStartControl():
    """
    PUT Gesture Mode
    """
    response = request.data
    global startControl
    startControl = df.json.loads(response)
    return response

@app.route('/MeshPoints', methods=['GET'])
def GetMeshPoints():
    """
    GET Gesture Mode
    """
    return pickle.dumps(meshPoints)

@app.route('/MeshPoints', methods=['PUT'])
def PutMeshPoints():
    """
    PUT Gesture Mode
    """
    response = request.data
    global meshPoints
    meshPoints = pickle.loads(response)
    return response




if __name__ == "__main__":
    logging.info('Startet Webserver Module')



    
    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # System APIs:
    endProgram = False

    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Comunication API:
    
    adeIDList = []
    calibrateADE = None
    batteryStatus = {}
    runADE = {}
    for ade in adeIDList:
        batteryStatus[ade] = 100
        runADE[ade] = False
    adeMoveList = df.adeMoveList()
    movementCommands = df.adeMoveList()
    currentLine = 0

    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Measurement Module API:
    measuredCords = np.array([])
    measuredImage = []
    adeNodes = df.ElementDefinition()

    
    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Gesture Module API:
    
    poseID = 0
    gestureMode = None

    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Simulation Module API:

    startSimulation = False
    sensorValues = {}
    
    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Control Module API:
    
    meshPoints = {}
    startControl = False

    app.run()
