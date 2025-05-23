#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 10:13:16 2025

@author: fynnheidenreich
"""

import os
import sys

import csv


if __name__ == "__main__":
    if os.getcwd().split('/')[-1].__contains__('exampleScripts'):
        os.chdir('..')
    if sys.path[0].split('/')[-1].__contains__('exampleScripts'):
        sys.path.append(os.getcwd())

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, 'output')
    

solutionFilePathRel="./solution/coordinatesSolution.txt"
solutionFilePath = os.path.join(output_dir, os.path.basename(solutionFilePathRel))
if os.path.isfile(solutionFilePath):
    os.remove(solutionFilePath)
    
import config
config.init()


from functions.simulation_functions import*

from functions import com_functions as cf
from functions import data_functions as df
#from functions import marker_functions as mf
from functions import measurement_functions as mef
from functions import webserver_functions as wf
from functions import trajectory_functions as tf

import exudyn as exu
from exudyn.utilities import*
from exudyn.interactive import SolutionViewer

import numpy as np
import matplotlib.pyplot as plt
from numpy import absolute, linalg as LA
import math
import logging
from copy import deepcopy
import time
import pickle
import json


from functions import visualization_functions as vf
from functions.control_functions import *
import matplotlib.ticker as ticker


def PreStepUserFunction(mbs, t):
    """
    Prestep User Function 
    """
    #print("mbs=", mbs2)q
    LrefExt = mbs.variables['LrefExt']
    Actors = mbs.variables['springDamperDriveObjects']
    Connectors = mbs.variables['Connectors']
    Vertices = mbs.variables['VertexLoads']
    TimeStamp = mbs.variables['TimeStamp']
    line = mbs.variables['line']



    if config.simulation.massPoints2DIdealRevoluteJoint2D:
        refLength = (config.lengths.L0) 
    else:
        refLength = (config.lengths.L0-2*(config.lengths.L1+config.lengths.L2)) 
        
    SensorValuesTime[line][t]={}
    
    
    
    
    SensorValues = {} 
    for ID in adeIDs:
        SensorValues[ID] = np.asmatrix([mbs.GetSensorValues(ADE[ID]['sensors'][0])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][1])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][2])[0:2]])   
    
    SensorValuesTime[line][t]=SensorValues

    ##+++++++++++++++++++++++++++++++++++++++
    ## Driving ADEs acording to MoveList if Activated
    if config.simulation.useMoveList:
        
        #nextTimeStamp = False
        for i in range(len(adeIDs)):
            index = adeMoveList.GetIndexOfID(adeIDs[i])
            if index != None:
                # Drive Actors:
                for j in range(3):
                    oSpringDamper = mbs.variables['springDamperDriveObjects'][i*3+j]
                    # u1 = mbs.GetObjectParameter(oSpringDamper,'referenceLength')-refLength
                    u1 = adeMoveList.data[cnt,index+j+1]/10000 #-adeMoveList.data[0,index+j+1]/10000
                    #u1 = LrefExt[i*3+j]
                    u2 = adeMoveList.data[cnt+1,index+j+1]/10000#-adeMoveList.data[0,index+j+1]/10000
                    oSpringDamper = mbs.variables['springDamperDriveObjects'][i*3+j]
                    driveLen = UserFunctionDriveRefLen1(t, u1,u2)
                    mbs.SetObjectParameter(oSpringDamper,'referenceLength',driveLen+refLength)       
       
                    
    if config.simulation.useInverseKinematic:
        for i in range(len(mbs.variables['springDamperDriveObjects'])):

            u1=sortedActorStrokesList[cnt][i]/10000 
            u2=sortedActorStrokesList[cnt+1][i]/10000
            
            # if i == 11 and t == 0.0332: 
            #     print('cnt={}: move actor {} from {} to {} '.format(cnt, i, u1, u2), 'correction' * mbs.variables['correction'])
            oSpringDamper = mbs.variables['springDamperDriveObjects'][i]
            driveLen = UserFunctionDriveRefLen1(t, u1,u2)
            
            # mbs.SetObjectParameter(oSpringDamper,'referenceLength',driveLen+refLength)  
            mbs.SetObjectParameter(oSpringDamper,'referenceLength',driveLen+refLength)#LrefExt[i])
        
        
    mbsV.SendRedrawSignal()
    
    return True #succeeded


def setConfigFileKinModel():
    config.simulation.useRevoluteJoint2DTower=True
    #Options for six-bar linkages, if all False RevoluteJoint2D are used for the six-bar linkages
    config.simulation.simplifiedLinksIdealRevoluteJoint2D = False #simplified model using only revoluteJoints at edges
    config.simulation.simplifiedLinksIdealCartesianSpringDamper = False
    config.simulation.massPoints2DIdealRevoluteJoint2D = True
    
    config.simulation.useCompliantJoints=False          #False for ideal
    config.simulation.numberOfElementsBeam=8
    config.simulation.numberOfElements=16

    config.simulation.useCrossSpringPivot = False #use CrossSpringPivot

    config.simulation.cartesianSpringDamperActive = False #setTrue ASME2020CI
    config.simulation.connectorRigidBodySpringDamper = False 
    
    config.simulation.rigidBodySpringDamperNonlinearStiffness = False #use nonlinear compliance matrix then choose circularHinge and crossSpringPivots
    config.simulation.rigidBodySpringDamperNonlinearStiffnessCircularHinge=False
    config.simulation.rigidBodySpringDamperNonlinearStiffnessCrossSpringPivot=False 
    config.simulation.rigidBodySpringDamperNeuralNetworkCircularHinge=False
    config.simulation.rigidBodySpringDamperNeuralNetworkCrossSprings=False
    config.simulation.neuralnetworkJIT = True
    
    if config.simulation.useCompliantJoints or config.simulation.useCrossSpringPivot:
        config.simulation.graphics = 'graphicsSTL'
    else:
        config.simulation.graphics = 'graphicsStandard'    




def setConfigFileSurrogateModel():
    config.simulation.useRevoluteJoint2DTower=True
    
    #Options for six-bar linkages, if all False RevoluteJoint2D are used for the six-bar linkages
    config.simulation.simplifiedLinksIdealRevoluteJoint2D = False #simplified model using only revoluteJoints at edges
    config.simulation.simplifiedLinksIdealCartesianSpringDamper = False
    config.simulation.massPoints2DIdealRevoluteJoint2D = False     
    
    config.simulation.useCompliantJoints=False   #False for ideal
    config.simulation.numberOfElementsBeam=8
    config.simulation.numberOfElements=16

    config.simulation.useCrossSpringPivot = True #use CrossSpringPivot

    config.simulation.cartesianSpringDamperActive = False #setTrue ASME2020CI
    config.simulation.connectorRigidBodySpringDamper = False 
    
    config.simulation.rigidBodySpringDamperNonlinearStiffness = False #use nonlinear compliance matrix then choose circularHinge and crossSpringPivots
    config.simulation.rigidBodySpringDamperNonlinearStiffnessCircularHinge=False
    config.simulation.rigidBodySpringDamperNonlinearStiffnessCrossSpringPivot=False 
    config.simulation.rigidBodySpringDamperNeuralNetworkCircularHinge=False  
    config.simulation.rigidBodySpringDamperNeuralNetworkCrossSprings=False 
    config.simulation.neuralnetworkJIT = True
    

    if config.simulation.useCompliantJoints or config.simulation.useCrossSpringPivot:
        config.simulation.graphics = 'graphicsSTL'
    else:
        config.simulation.graphics = 'graphicsStandard'





def createKinematicModel(mbs, mbs2, adeMoveList=None):
    #create new simplified idealized system
    setConfigFileKinModel()   
        
    mbs2.variables['springDamperDriveObjects']=[] #all drives that need to be changed over time
    mbs2.variables['springDamperDriveObjectsFactor']=[] #factor for drives: depends on distance to neutral fiber
    mbs2.variables['LrefExt']=[]
    mbs2.variables['Connectors'] = []
    mbs2.variables['MarkersConnectors'] = []
    mbs2.variables['MarkersConnectorsLoads'] = []
    mbs2.variables['Vertices'] = []
    mbs2.variables['VertexLoads'] = []
    mbs2.variables['TimeStamp'] = 0
    mbs2.variables['nextTimeStamp'] = False
    mbs2.variables['Sensors'] = []
    mbs2.variables['activateMouseDrag'] = False
    mbs2.variables['driveToPoint'] = False
    mbs2.variables['line'] = 0
    mbs2.variables['xValue']=[]
    mbs2.variables['correction'] = False
    # mbs2.variables['counterForTesting'] = 0                
    mbs2.variables['xValue']=[]  
    mbs2.variables['sensorTCP']=[]


    if adeMoveList==None:
        ADE2, mbs2 = GenerateMesh(adeNodes,mbs2,adeMoveList = adeMoveList)     #Generating Mesh with Movelist
    else:           
        ADE2, mbs2 = GenerateMesh(adeNodes,mbs2)                               #Generating Mesh without Movelist
              
    sensorMbsTCPList=sensorList(nodeList, adeNodes, mbs2)
    mbs2.variables['sensorTCP']=sensorTCPList
    
    oGround2 = mbs2.AddObject(ObjectGround(referencePosition = [0,0,0]))
   
    mbs2.Assemble()
   
    if type(adeNodes) == dict:
        for ID in adeNodes[0].elements:
            for i in range(len(adeNodes[0].connectionList[ID])):
                if adeNodes[0].connectionList[ID][i][2] in ADE2:
                    ConnectADE(mbs2,ADE2,adeNodes[0].connectionList[ID][i])
    else:
        for ID in adeNodes.elements:
            for i in range(len(adeNodes.connectionList[ID])):
                if adeNodes.connectionList[ID][i][2] in ADE2:
                    ConnectADE(mbs2,ADE2,adeNodes.connectionList[ID][i])                
        
    ikObj = InverseKinematicsATE(mbs2, sensorTCPList, mbs, oGround2, ADE2)
                  
    return ikObj, mbs2, sensorTCPList




def createSurrogateModel(mbs, adeNodes):
    ##+++++++++++++++++++++++++++++++++++++++
    ## Checking if to end the Program
    ##+++++++++++++++++++++++++++++++++++++++
    ## Initialization of Variables
    mbs.variables['springDamperDriveObjects']=[] #all drives that need to be changed over time
    
    # mbs.variables['springDamperDriveObjectsStop']=[] #all drives that need to be changed over time
    
    mbs.variables['springDamperDriveObjectsFactor']=[] #factor for drives: depends on distance to neutral fiber
    mbs.variables['LrefExt']=[]
    mbs.variables['Connectors'] = []
    mbs.variables['MarkersConnectors'] = []
    mbs.variables['MarkersConnectorsLoads'] = []
    mbs.variables['Vertices'] = []
    mbs.variables['VertexLoads'] = []
    mbs.variables['TimeStamp'] = 0
    mbs.variables['nextTimeStamp'] = False
    mbs.variables['Sensors'] = []
    mbs.variables['activateMouseDrag'] = False
    mbs.variables['driveToPoint'] = False
    mbs.variables['line'] = 0
    mbs.variables['xValue']=[]
    mbs.variables['correction'] = False
    
    mbs.variables['counterForTesting'] = 0
    mbs.variables['rigidBodySpringDampers']=[]
    mbs.variables['predictedForces']=[]

    

    
    
    if config.simulation.activateWithKeyPress:
        mbs.variables['activateMouseDrag'] = False
    
    mbs.variables['xValue']=[]

    ##+++++++++++++++++++++++++++++++++++++++
    ## Getting Variables from Webserver
    # adeNodes =  pickle.loads(client.Get('adeNodes'))
    adeIDs = []

    if type(adeNodes) == dict:
        for ID in adeNodes[0].elements:
                adeIDs.append(ID)
    else:
        for ID in adeNodes.elements:
                adeIDs.append(ID)

    startSimulation = True

    # if config.simulation.useMoveList:
        # adeMoveList = df.adeMoveList()
        # adeMoveList.FromJson(client.Get('MoveList'))


    setConfigFileSurrogateModel()



    if startSimulation and (type(adeNodes) == df.ElementDefinition or type(adeNodes) == dict):
        if config.simulation.useMoveList: 
            ADE, mbs = GenerateMesh(adeNodes,mbs,adeMoveList = adeMoveList)         #Generating Mesh with Movelist
        else:
            ADE, mbs = GenerateMesh(adeNodes,mbs)                                   #Generating Mesh without Movelist
    
    return mbs, ADE, adeIDs




def CalculateActorLengths(meshPoints):
    lengths = {}
    for ID in meshPoints:
        A = LA.norm(meshPoints[ID][0] -meshPoints[ID][1])-config.lengths.L0
        C = LA.norm(meshPoints[ID][1] -meshPoints[ID][2])-config.lengths.L0
        B = LA.norm(meshPoints[ID][2] -meshPoints[ID][0])-config.lengths.L0

        lengths[ID] = [A,B,C]

    return lengths

def CalculateNextMovementCommandsFromSensor(actorLength,adeMoveList):
    nextLine = df.adeMoveList()
    nextLine.data = adeMoveList.data[0,:].copy()
    for ID in actorLength:
        index = nextLine.GetIndexOfID(ID)

        nextLine.data[0,index+1] += actorLength[ID][0]*10000
        nextLine.data[0,index+2] += actorLength[ID][1]*10000
        nextLine.data[0,index+3] += actorLength[ID][2]*10000


    return nextLine








##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Initialization of Module
config.init()
logging.info('Starte Simulations Module')

client = wf.WebServerClient()
SC = exu.SystemContainer()
mbs = SC.AddSystem()



timeTotal=True
addTimeStaticStep=False
numberOfSteps=0



config.simulation.constrainActorLength = True



##simulation options
displaySimulation=True

writeStatistics=False


simulateArm=True


my_dir = os.path.abspath(os.path.join(os.getcwd(), 'functions'))


if simulateArm:
    adeNodes = df.ElementDefinition()
    # adeNodes.AddElementNodes(ID= 1, Nodes = [3,1,2])
    # nodeList = [3]
    
    nx = 1
    ny = 4
    lengthInitX = 0.229*nx
    lengthInity = 0.229*ny
    GenerateGridSimulation(nx, ny, adeNodes)
    # adeNodes.AddElementNodes(ID = 11, Nodes = [11,12,13])
    
    existingNodeIDs = list(adeNodes.nodeDefinition.keys())
    maxNodeID = max(existingNodeIDs)
    
    yCoordinates = [coord[1] for coord in adeNodes.nodeDefinition.values()]
    yMaxArm = max(yCoordinates)
    
    nodeYList = [(nodeID, coord[1]) for nodeID, coord in adeNodes.nodeDefinition.items()]
    nodeYList.sort(key=lambda x: x[1], reverse=True)
    topNode1, topNode2 = nodeYList[0][0], nodeYList[1][0]
    print("IDs der zwei höchsten Knoten:", topNode1, topNode2)
    
    # load gripper seperately
    tempGripper = df.ElementDefinition()
    tempGripper.ReadMeshFile('data/experiments/gripper/initial9_gripper')
    
    # offset Nodes Coordinates on top of arm
    newNodeDefinition = {}
    nodeIDMap = {}
    
    for oldID, coord in tempGripper.nodeDefinition.items():
        newID = oldID + maxNodeID - 2
        newCoord = np.array([coord[0], coord[1] + yMaxArm])
        newNodeDefinition[newID] = newCoord
        nodeIDMap[oldID] = newID
    
    # add Nodes
    adeNodes.nodeDefinition.update(newNodeDefinition)
    
    # offset ADE IDs to come after the ADEs of the Arm
    for elemID, nodes in tempGripper.elements.items():
        newElemID = max(adeNodes.elements.keys()) + 1
        newNodes = [nodeIDMap[n] for n in nodes]
        adeNodes.AddElementNodes(ID=newElemID, Nodes=newNodes, setLists=False)
        
    # add connections to new Node IDs
    for connID, connList in tempGripper.connectionList.items():
        newConnList = []
        for conn in connList:
            newConn = conn.copy()
            newConn[1] = nodeIDMap.get(conn[1], conn[1])
            newConn[2] = nodeIDMap.get(conn[2], conn[2])
            newConn[3] = nodeIDMap.get(conn[3], conn[3])
            newConnList.append(newConn)
        
        newConnID = max(adeNodes.connectionList.keys()) + 1
        adeNodes.connectionList[newConnID] = newConnList
    
    # update lists
    adeNodes.SetConnectionList()
    adeNodes.SetListOfNodes()
    adeNodes.SetADEPointsOfNode()
    adeNodes.SetNodeNumbersToAde()
    
    # change this to something that works with different grippers too
    nodeList = [topNode2 + 2]
    print('TCP Node ID:', nodeList)
    
    # config.simulation.setMeshWithCoordinateConst = False
    config.simulation.setMeshWithCoordinateConst = True


if config.simulation.setMeshWithCoordinateConst:
    config.simulation.numberOfBoundaryPoints = 0 #change here number of boundary 


oGround = mbs.AddObject(ObjectGround(referencePosition = [0,0,0]))


computeDynamic=True
config.simulation.useMoveList = False
useInverseKinematic = True



config.simulation.boundaryConnectionInfoInMoveList = False




#Options for Visualization
config.simulation.frameList = True
config.simulation.showJointAxes=False

config.simulation.animation=True

config.simulation.showLabeling = True #shows nodeNumbers, ID and sideNumbers, simulationTime


visualizeTrajectory=True
visualizeSurfaces=False


def sensorList(nodeList, adeNodes, mbs):  
    sensorTCPList=[]
    for node in nodeList:
        getADEFromNodeNumber=next(iter(adeNodes.adePointsOfNode[node]))
        adeIndex012OfNodeFromADEID=adeNodes.nodeNumbersToAdeIndex[getADEFromNodeNumber][0]
        sensorTCP = mbs.AddSensor(SensorMarker(markerNumber=mbs.GetSensor(ADE[getADEFromNodeNumber]['sensors'][adeIndex012OfNodeFromADEID])['markerNumber'], outputVariableType=exu.OutputVariableType.Position, storeInternal=True))
        sensorTCPList +=[sensorTCP]
    return sensorTCPList



usePTP = True
useSyncPTP = False


# +++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++
## start program ++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++
# +++++++++++++++++++++++++++++++++++++++
timeTotal=0
numberOfSteps=0



startSimulation = False
while not startSimulation:
    try:
        mbs, ADE, adeIDs = createSurrogateModel(mbs, adeNodes)

        simulationSettings = exu.SimulationSettings()
        T=0.002
        SC.visualizationSettings.connectors.defaultSize = T
        SC.visualizationSettings.bodies.defaultSize = [T, T, T]
        SC.visualizationSettings.nodes.defaultSize = 0.0025
        SC.visualizationSettings.markers.defaultSize = 0.005
        SC.visualizationSettings.loads.defaultSize = 0.015
        SC.visualizationSettings.general.autoFitScene=True

        SC.visualizationSettings.nodes.show= False
        SC.visualizationSettings.markers.show= False
        SC.visualizationSettings.connectors.show= False

        SC.visualizationSettings.openGL.lineWidth=2
        SC.visualizationSettings.openGL.lineSmooth=True
        SC.visualizationSettings.openGL.multiSampling = 4
        SC.visualizationSettings.general.drawCoordinateSystem = True
        SC.visualizationSettings.window.renderWindowSize=[1600,1000]
        
        SC.visualizationSettings.contact.showSearchTree = True
        SC.visualizationSettings.contact.showSearchTreeCells = True
        SC.visualizationSettings.contact.showBoundingBoxes = True
        
        if visualizeTrajectory:
            SC.visualizationSettings.sensors.traces.showPositionTrace = True
            
        if config.simulation.activateWithKeyPress:
            SC.visualizationSettings.window.keyPressUserFunction = UserFunctionkeyPress
            
            if config.simulation.showJointAxes:
                SC.visualizationSettings.connectors.defaultSize = 0.002
                SC.visualizationSettings.connectors.showJointAxes = True
                SC.visualizationSettings.connectors.jointAxesLength = 0.0015
                SC.visualizationSettings.connectors.jointAxesRadius = 0.0008
        
        # create sensor for "MCP"   
        sensorTCPList=sensorList(nodeList, adeNodes, mbs)
        sensorTCP=sensorTCPList

        mbsV = SC.AddSystem()
        mbsV.variables['mbs']=mbs
        mbs.variables['mbsV']=mbsV
        
        if visualizeSurfaces:
            def UFgraphics2(mbs, objectNum):
                mbsV = mbs.variables['mbsV']
                mbsV.SendRedrawSignal()
                return []
            
            if visualizeTrajectory:
                ground2 = mbs.AddObject(ObjectGround(visualization=VObjectGround(graphicsDataUserFunction=UFgraphics2)))        
            
        ##+++++++++++++++++++++++++++++++++++++++
        ## Assemble Mesh 
        mbs.Assemble()
        sysStateList = mbs.systemData.GetSystemState()

        if displaySimulation:
            exu.StartRenderer()
            mbs.WaitForUserToContinue()
            
        #connect and disconnect ATCs
        if type(adeNodes) == dict:
            for ID in adeNodes[0].elements:
                for i in range(len(adeNodes[0].connectionList[ID])):
                    if adeNodes[0].connectionList[ID][i][2] in ADE:
                        ConnectADE(mbs,ADE,adeNodes[0].connectionList[ID][i])
        else:
            for ID in adeNodes.elements:
                for i in range(len(adeNodes.connectionList[ID])):
                    if adeNodes.connectionList[ID][i][2] in ADE:
                        ConnectADE(mbs,ADE,adeNodes.connectionList[ID][i])

        mbs.AssembleLTGLists()
        mbs.systemData.SetSystemState(sysStateList,configuration = exu.ConfigurationType.Initial)

        ##+++++++++++++++++++++++++++++++++++++++
        ## Setting um SensorValues and Sending initial Sensor Values (Mesh Node Coordinates)
        SensorValuesTime = {}
        SensorValuesTime[0] = {}
        SensorValues = {}
        SensorValues[0] = {}

        counterSensorPlot=0
        SensorValuesTime2 = {} 
        SensorValuesTime2[0] = {}  
        SensorValues2 = {} 
        SensorValues2[0] = {}
        
        for ID in adeIDs:
            SensorValues[0][ID] = np.asmatrix([mbs.GetSensorValues(ADE[ID]['sensors'][0])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][1])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][2])[0:2]])
        # client.Put('SensorValues',pickle.dumps(SensorValues))
        SensorValuesTime[0]=SensorValues.copy()
        ##+++++++++++++++++++++++++++++++++++++++
        for ID in adeIDs:
            SensorValues2[0][ID] = np.asmatrix([mbs.GetSensorValues(ADE[ID]['sensors'][0])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][1])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][2])[0:2]])
        # client.Put('SensorValues',pickle.dumps(SensorValues))
        SensorValuesTime2[0]=SensorValues2.copy()
        
        if visualizeTrajectory:
            #user function for moving graphics:
            def UFgraphics(mbsV, objectNum):
                gSurface=[]

                mbs = mbsV.variables['mbs']
                
                for ID in adeIDs:
                    A = mbs.GetSensorValues(ADE[ID]['sensors'][0],configuration=exu.ConfigurationType.Visualization) 
                    B = mbs.GetSensorValues(ADE[ID]['sensors'][1],configuration=exu.ConfigurationType.Visualization)
                    C = mbs.GetSensorValues(ADE[ID]['sensors'][2],configuration=exu.ConfigurationType.Visualization)
                    
                    points=[A,B,C]       
                    
                    triangles=[0,1,2]
                    gSurface+=[GraphicsDataFromPointsAndTrigs(points,triangles,color=[0.41,0.41,0.41,0.5])]
            
                return gSurface
            ground = mbsV.AddObject(ObjectGround(visualization=VObjectGround(graphicsDataUserFunction=UFgraphics)))
        
         
            
        mbsV.Assemble()


        
        simulationSettings.staticSolver.newton.numericalDifferentiation.relativeEpsilon = 1e-4
        newResidual = True
        if not(newResidual):
            simulationSettings.staticSolver.newton.relativeTolerance = 1e-10
            simulationSettings.staticSolver.newton.absoluteTolerance = 1e-12
        else:
            #new solver settings, work for all 2x7 configurations with eps=1e-4
            simulationSettings.staticSolver.newton.relativeTolerance = 1e-5
            simulationSettings.staticSolver.newton.absoluteTolerance = 1e-5
            simulationSettings.staticSolver.newton.newtonResidualMode = 1 #take Newton increment
        
        simulationSettings.staticSolver.stabilizerODE2term = 1
        
        simulationSettings.staticSolver.newton.numericalDifferentiation.relativeEpsilon = 1e-9
        simulationSettings.staticSolver.newton.maxIterations = 50
            
        simulationSettings.linearSolverType = exu.LinearSolverType.EigenSparse
        simulationSettings.staticSolver.newton.weightTolerancePerCoordinate = True   
        simulationSettings.solutionSettings.solutionInformation = "PARTS_StaticSolution"
        
        
        
        
        
        numberOfLoadSteps=1
        simulationSettings.staticSolver.numberOfLoadSteps = numberOfLoadSteps

        exu.SolveStatic(mbs, simulationSettings=simulationSettings)
        
        
        solODE2initial = mbs.systemData.GetODE2Coordinates(configuration=exu.ConfigurationType.Current)
        mbs.systemData.SetODE2Coordinates(solODE2initial, configuration=exu.ConfigurationType.Initial)

        mbs.WaitForUserToContinue()
        
        ##############################################################  

        simulationSettings.timeIntegration.numberOfSteps = int(config.simulation.nrSteps)
        simulationSettings.timeIntegration.endTime = config.simulation.endTime
        simulationSettings.timeIntegration.generalizedAlpha.computeInitialAccelerations = False

        if addTimeStaticStep:
            timeTotal=timeTotal+mbs.sys['staticSolver'].timer.total
            numberOfSteps=numberOfSteps+1

        ##############################################################
        # IMPORTANT!!!!!!!!!
        simulationSettings.timeIntegration.newton.useModifiedNewton = True #JG
        simulationSettings.displayStatistics = True
        simulationSettings.displayComputationTime = True

        if config.simulation.solutionViewer:
            SC.visualizationSettings.general.autoFitScene=False
            simulationSettings.solutionSettings.solutionWritePeriod = config.simulation.solutionWritePeriod
            simulationSettings.solutionSettings.writeSolutionToFile = True
            # simulationSettings.solutionSettings.coordinatesSolutionFileName = config.simulation.solutionViewerFile
            simulationSettings.solutionSettings.coordinatesSolutionFileName = solutionFilePath
            simulationSettings.solutionSettings.writeFileFooter = False
            simulationSettings.solutionSettings.writeFileHeader = True

            if os.path.isfile(solutionFilePath):
                os.remove(solutionFilePath)
            simulationSettings.solutionSettings.appendToFile = True
            simulationSettings.solutionSettings.sensorsAppendToFile =True

        mbs.SetPreStepUserFunction(PreStepUserFunction)
        
        
        
        if useInverseKinematic:
            if config.simulation.useMoveList==False:
                cnt=-1

            
            
            config.simulation.useInverseKinematic=True
            ##+++++++++++++++++++++++++++++++++++++++
            ## Starting Simulation without Movelist
            if config.simulation.useInverseKinematic == True:
    
                config.simulation.useMoveList = False
    
                mbs2 = SC.AddSystem()
                averageSensorValues=AverageMeshPoints(SensorValues[cnt+1],adeNodes,draw = False)
                
                for node in adeNodes.adePointsOfNode:
                    ADENr=next(iter(adeNodes.adePointsOfNode[node]))
                    index=adeNodes.adePointsOfNode[node][ADENr]
                    adeNodes.nodeDefinition[node] = np.array(averageSensorValues[ADENr][index])[0]
                                        
                adeNodes.connectionList={}
                adeNodes.SetListOfNodes()           
                adeNodes.SetConnectionList()
                adeNodes.SetNodeNumbersToAde()
                adeNodes.SetADEPointsOfNode()   
    
                ikObj, mbs2, sensorMbsTCP = createKinematicModel(mbs, mbs2)
                
                ##############################################################
                
                dxx = [[0.5, 0]]            
                
                ##############################################################
            
                cnt=0
                actorList = []
                sortedActorStrokesList=[] 
                 
                setConfigFileKinModel()  # get ActuatorLength from KinModel 
                l1 = ikObj.GetActuatorLength()
                
    
                if config.simulation.massPoints2DIdealRevoluteJoint2D:
                    refLength = (config.lengths.L0) 
                else:
                    refLength = (config.lengths.L0-2*(config.lengths.L1+config.lengths.L2))
                
                
                # map actuator length to SurrogateModel
                actorList = [((np.array(l1)-refLength)*10000).tolist()]
                # actorList = [(np.round(l1, 5)).tolist()]
                sortedActorStrokesList += actorList      
                
                # mbs.SetPreStepUserFunction(PreStepUserFunction)
                import matplotlib.colors as mColor
                mycolors = list(mColor.TABLEAU_COLORS)
    
    
                detaX=np.array([0,0])
                
                # set start and end point
                start = np.array([0, 0])
                target = np.array([0.5, 0])
                # set max velocity and acceleration
                vmax = 20
                amax = 10
                
                actorTraj = []
                
                # calculate actuator Lengths for start position
                ikObj.InverseKinematicsRelative(None, np.array(start))
                startLengths = ((np.array(ikObj.GetActuatorLength()) - refLength) * 10000)
                
                # calculate actuator Lengths for target position
                ikObj.InverseKinematicsRelative(None, np.array(target))
                targetLengths = ((np.array(ikObj.GetActuatorLength()) - refLength) * 10000)
                
                
                
                # calculate trajectory for every actuator
                if usePTP:
                    traj = tf.PTPTrajectory(startLengths, targetLengths, vmax, amax)
                elif useSyncPTP:
                    traj = tf.syncPTPTrajectory(startLengths, targetLengths, vmax, amax)
                
                t_sim = 0
                dt = 0.2


                colors = list(mColor.TABLEAU_COLORS.values())
                print('Number of Akuators:', traj.numActuators)

                
                # run loop for time the slowest actuator needs
                while t_sim < np.max(traj.t_total):
                    # get actuator lenghts for current time step
                    l1 = traj.evaluate(t_sim)
                    
                    for i in range(traj.numActuators):
                        plt.scatter(t_sim, l1[i], s=10, color=colors[i % len(colors)], label=f"Aktuator {i+1}")
                    
                        
                    # print('t_sim:', t_sim)
                    
    
                    ### toDO change here for more nodes the error!!!!
                    # iWantToGoToThisValue = copy.copy(mbs2.GetSensorValues(sensorMbsTCP[0])[0:2])
                    ### toDO change here for more nodes the error!!!!
    
                    # l1 = ikObj.GetActuatorLength()
    
                    if config.simulation.massPoints2DIdealRevoluteJoint2D:
                        refLength = (config.lengths.L0) 
                    else:
                        refLength = (config.lengths.L0-2*(config.lengths.L1+config.lengths.L2))
                        
                        
                    actorList = [np.array(l1).tolist()]
                    
                    sortedActorStrokesList += actorList 
    
                    mbs.variables['line']=cnt+1
                    SensorValuesTime[cnt+1]={} 
    
           
                    nextTimeStamp = False   
    
                    if computeDynamic:
                        
                        # setConfigFileKinModel()    
                        setConfigFileSurrogateModel()
                        
                        simulationSettings.timeIntegration.numberOfSteps = int(config.simulation.nrSteps)
                        # simulationSettings.timeIntegration.endTime = config.simulation.endTime
                
                        simulationSettings.displayComputationTime = False  
                        simulationSettings.timeIntegration.verboseMode = 0
                        simulationSettings.displayStatistics = False       
                        
                        
                        mbs.SetPreStepUserFunction(PreStepUserFunction)
                        
    
                        simulationSettings.staticSolver.verboseMode=0
    
                        exu.SolveStatic(mbs, simulationSettings,showHints=False) 
                        # exu.SolveDynamic(mbs, simulationSettings) 
                        
                        timeTotal=timeTotal+mbs.sys['staticSolver'].timer.total
                        numberOfSteps=numberOfSteps+1                            
                        
                        
                        ## Sending new Sensor Values to Webserver
                        SensorValuesTime[cnt+1][0]={}
                        SensorValues[cnt+1] = {}
                        for ID in adeIDs:
                            SensorValues[cnt+1][ID] = np.asmatrix([mbs.GetSensorValues(ADE[ID]['sensors'][0])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][1])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][2])[0:2]])
                        # client.Put('SensorValues',pickle.dumps(SensorValues))                           
                        SensorValuesTime[cnt+1][0]=SensorValues[cnt+1]    
                        
                        # for plotSensorMbsTCP in sensorMbsTCP:
                        #     plt.scatter(mbs2.GetSensorValues(plotSensorMbsTCP)[0], mbs2.GetSensorValues(plotSensorMbsTCP)[1], marker='d', color='red', s=48)
                        #     plt.scatter(iWantToGoToThisValue[0], iWantToGoToThisValue[1], marker='d', color='red', s=48)
                            
                            
                        # for plotSensorTCP in sensorTCP:
                        #     plt.scatter( mbs.GetSensorValues(plotSensorTCP)[0], mbs.GetSensorValues(plotSensorTCP)[1], marker='x', color='red', s=48)
                            
    
                        cnt=cnt+1
    
                        sysStateList = mbs.systemData.GetSystemState()
                        mbs.systemData.SetSystemState(sysStateList,configuration = exu.ConfigurationType.Initial)     
    
    
                                
                        counterSensorPlot=counterSensorPlot+1
                        
                        SensorValuesTime2[counterSensorPlot]={} 
                        SensorValuesTime2[counterSensorPlot][0]={}
                        SensorValues2[counterSensorPlot] = {}
                        for ID in adeIDs:
                            SensorValues2[counterSensorPlot][ID] = np.asmatrix([mbs.GetSensorValues(ADE[ID]['sensors'][0])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][1])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][2])[0:2]])                     
                        SensorValuesTime2[counterSensorPlot][0]=SensorValues2[counterSensorPlot]  
                        
    
                        if not mbs.GetRenderEngineStopFlag():
                            sysStateList = mbs.systemData.GetSystemState()
                            mbs.systemData.SetSystemState(sysStateList,configuration = exu.ConfigurationType.Initial)
                            
                    t_sim += dt
                    
                            
                            
                if displaySimulation:  
                    mbs.WaitForUserToContinue()      
                    exu.StopRenderer() #safely close rendering window!                
                    
    
                else:
                    if computeDynamic:
    
                            simulationSettings.solutionSettings.solutionInformation = "Exudyn ADE Simulation"
    
                            mbs.Assemble()
                            
                            if displaySimulation:
                                exu.StartRenderer()
                                mbs.WaitForUserToContinue()
    
                            exu.SolveStatic(mbs, simulationSettings = simulationSettings)
    
                            ##+++++++++++++++++++++++++++++++++++++++
                            ## Sending new Sensor Values to Webserver
                            SensorValues[1] = {}
                            for ID in adeIDs:
                                SensorValues[1][ID] = np.asmatrix([mbs.GetSensorValues(ADE[ID]['sensors'][0])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][1])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][2])[0:2]])
                            client.Put('SensorValues',pickle.dumps(SensorValues))
                            SensorValuesTime[cnt][0]=SensorValues[cnt+1]  
                        
                            # print("stop flag=", mbs.GetRenderEngineStopFlag())
                    
                        
                    if not mbs.GetRenderEngineStopFlag():
                        sysStateList = mbs.systemData.GetSystemState()
                        mbs.systemData.SetSystemState(sysStateList,configuration = exu.ConfigurationType.Initial)
                    else: 
                        break
        
        if displaySimulation and not mbs.GetRenderEngineStopFlag():  
            mbs.WaitForUserToContinue()      
            exu.StopRenderer() #safely close rendering window!
        try:
            if config.simulation.solutionViewer:
                  sol = LoadSolutionFile(config.simulation.solutionViewerFile, safeMode=True)#, maxRows=100) 
                  SolutionViewer(mbs,sol) #can also be entered in 
        except:
            logging.warning('Couldnt load Solution File')


        ##+++++++++++++++++++++++++++++++++++++++
        # hier pfade absolut machen, da program ordner wechselt
        sensorValuesPath = os.path.join(output_dir, os.path.basename(config.simulation.sensorValuesFile))
        sensorValuesTimePath = os.path.join(output_dir, os.path.basename(config.simulation.sensorValuesTimeFile))
        sensorValuesPath2 = os.path.join(output_dir, os.path.basename(config.simulation.sensorValuesFile2))
        sensorValuesTimePath2 = os.path.join(output_dir, os.path.basename(config.simulation.sensorValuesTimeFile2))
        
        ## Saving Sensor Data to File
        if config.simulation.saveSensorValues:
              # pickle.dump(  SensorValues,  open(config.simulation.sensorValuesFile,  'wb')  )
              # pickle.dump(  SensorValuesTime,  open(config.simulation.sensorValuesTimeFile,  'wb')  )
              
              # pickle.dump(  SensorValues2,  open(config.simulation.sensorValuesFile2,  'wb')  )
              # pickle.dump(  SensorValuesTime2,  open(config.simulation.sensorValuesTimeFile2,  'wb')  )
              
              pickle.dump(  SensorValues,  open(sensorValuesPath,  'wb')  )
              pickle.dump(SensorValuesTime, open(sensorValuesTimePath, 'wb'))
              pickle.dump(SensorValues2, open(sensorValuesPath2, 'wb'))
              pickle.dump(SensorValuesTime2, open(sensorValuesTimePath2, 'wb'))
             
        ##+++++++++++++++++++++++++++++++++++++++
        ## Resetting Simulation
        mbs.Reset()
        startSimulation = True

        break
    
    except Exception as ExeptionMessage: 
        logging.warning('Error ocured during Simulation')
        logging.warning(ExeptionMessage)
        startSimulation = False
        #client.Put('StartSimulation',json.dumps(False))
        mbs.Reset()
    time.sleep(config.simulation.looptime-time.time()%config.simulation.looptime)

        


print('time: ', timeTotal, 'numberOfSteps: ',numberOfSteps)
print('userFunction: ',mbs.variables['counterForTesting'])

        
        
 

