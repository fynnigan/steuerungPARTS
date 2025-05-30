﻿#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Data and information that support the findings of the article:
# M. Pieber and Z. Zhang and P. Manzl and J. Gerstmayr, SURROGATE MECHANICAL 
# MODEL FOR PROGRAMMABLE STRUCTURES
# Proceedings of the ASME 2024
#
# Details:             
# Circular hinge with beam elements and cross-spring pivot with beam elements
# Compute data for six-bar linkage with polynomial fitting and 
#
#
#
# Author:   M. Pieber
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
    if os.getcwd().split('/')[-1].__contains__('exampleScripts'):
        os.chdir('..')
    if sys.path[0].split('/')[-1].__contains__('exampleScripts'):
        sys.path.append(os.getcwd())


solutionFilePath="./solution/coordinatesSolution.txt"
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





def createKinematicModel(mbs, mbs2, adeMoveList2=None):
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
              
    sensorMbs2TCPList=sensorList(nodeList, adeNodes, mbs2)
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


simulate4ATCs=False
simulateGripper=True 
simulateTowerADEs=False
simulateRabbit=False
simulateDuckBridgeCraneTower=False
simulateReconfigurationGerbl=False
##++++++++++++++++++++++++++++++
##ANIMATIONS
##make images for animations (requires FFMPEG):
# my_dir = os.path.abspath('./functions')
my_dir = os.path.abspath(os.path.join(os.getcwd(), 'functions'))
# os.system('start /MIN cmd /C  python "'+os.path.join(my_dir, 'webserver_functions.py')+'"')    
# os.system('start /MIN cmd /C python "'+os.path.join(my_dir, 'measurement_functions.py')+'"')



if simulate4ATCs:
    adeNodes = df.ElementDefinition()
    adeNodes.AddElementNodes(ID= [4,1], Nodes = [[3,1,2],[3,2,4]])
    adeNodes.AddElementNodes(ID= 4, Nodes = [3,1,2])
    adeNodes.AddElementNodes(ID= 1, Nodes = [3,2,4])
    adeNodes.AddElementNodes(ID= 2, Nodes = [3,4,5])
    adeNodes.AddElementNodes(ID= 3, Nodes = [6,5,4])

    adeMoveList = df.adeMoveList('data/moveList/Fahrt014ADE.txt')
    # adeMoveList = df.adeMoveList('data/moveList/Fahrt014ADETest.txt')
    adeNodes.connectionList={}
    adeNodes.SetListOfNodes()           
    adeNodes.SetConnectionList()
    adeNodes.SetNodeNumbersToAde()
    adeNodes.SetADEPointsOfNode() 
    
    nodeList=[6] #nodes to constrain
    
    # adeMoveList.Split(maxStepsize=5, keepConnectionInfo=True)
    
    config.simulation.setMeshWithCoordinateConst = True
    
if simulateGripper:
    TimeStamp=0
    adeNodes = {}
    adeNodes = df.ElementDefinition()
    adeNodes.ReadMeshFile('data/experiments/gripper/initial9_gripper')
    adeMoveList = df.adeMoveList('data/experiments/gripper/configuration_gripper.txt')
    # adeNodes.SetNodeNumbersToAde()

    adeNodes.connectionList={}
    adeNodes.SetListOfNodes()           
    adeNodes.SetConnectionList()
    adeNodes.SetNodeNumbersToAde()
    adeNodes.SetADEPointsOfNode() 

    nodeList=[8,11] #nodes to constrain
    # nodeList=[8] #nodes to constrain
    
    config.simulation.setMeshWithCoordinateConst = True
    

if simulateTowerADEs:
    adeMoveList=df.adeMoveList()
    adeNodes = df.ElementDefinition()
    nx=10
    ny=2
    lengthInitX=0.229*nx
    lengthInitY=0.229*ny
    GenerateGridSimulation(nx,ny-1,adeNodes)


    nodeList=[]
    tol=0.003
    P1=[lengthInitX,0]
    P2=[lengthInitX,lengthInitY]
    nodesInRectangle=findNodesInRectangle(adeNodes, P1 = np.array(P1), P2 = np.array(P2))       
    nodeList+=nodesInRectangle
    # nodeList=[63,42,21] #nodes to constrain
    
    config.simulation.setMeshWithCoordinateConst = False

if simulateRabbit:
    adeMoveList=df.adeMoveList()
    adeNodes = df.ElementDefinition()
    adeNodes.ReadAbaqusInputFile('data/meshFiles/Job-1_works.inp')
    adeIDs = []
    for ID in adeNodes.elements:
        adeIDs.append(ID)
    
    adeNodes.SetADEPointsOfNode()
    adeNodes.SetNodeNumbersToAde()

    nodeList=[]
    tol=0.003
    P1=[-3.418,-1.223]
    P2=[-2.581,-0.07105]
    nodesInRectangle=findNodesInRectangle(adeNodes, P1 = np.array(P1), P2 = np.array(P2))       
    nodeList+=nodesInRectangle

    P1=[3.5,-2.2]
    P2=[4.635,-1.405]
    nodesInRectangle=findNodesInRectangle(adeNodes, P1 = np.array(P1), P2 = np.array(P2))       
    nodeList+=nodesInRectangle    
    # nodeList=[18,19,20,17,8,7,46,91,45] #nodes to constrain
    

    config.simulation.setMeshWithCoordinateConst = False

    config.lengths.maxStroke = 0.2
    config.lengths.minStroke = -0.1

if simulateDuckBridgeCraneTower:
    adeNodes = {}
    
    # ADE Nodes at TimeStamp 0
    adeNodes[0] = df.ElementDefinition()
    adeNodes[0].AddElementNodes(ID= 6, Nodes = [1,2,3])
    adeNodes[0].AddElementNodes(ID= 5, Nodes = [4,3,2])
    adeNodes[0].AddElementNodes(ID= 3, Nodes = [1,3,5])
    adeNodes[0].AddElementNodes(ID= 4, Nodes = [6,5,3])
    adeNodes[0].AddElementNodes(ID= 1, Nodes = [6,8,5])
    adeNodes[0].AddElementNodes(ID= 2, Nodes = [7,5,8])
    # adeNodes[0].AddElementNodes(ID= 7, Nodes = [7,8,9])
    # adeNodes[0].AddElementNodes(ID= 8, Nodes = [7,9,10])
    #ADE Nodes at TimeStamp 1
    adeNodes[1] = df.ElementDefinition()
    adeNodes[1].AddElementNodes(ID= 6, Nodes = [1,2,3])
    adeNodes[1].AddElementNodes(ID= 5, Nodes = [4,3,2])
    adeNodes[1].AddElementNodes(ID= 4, Nodes = [4,5,3])
    adeNodes[1].AddElementNodes(ID= 3, Nodes = [9,3,5])
    adeNodes[1].AddElementNodes(ID= 1, Nodes = [4,8,5])
    adeNodes[1].AddElementNodes(ID= 2, Nodes = [7,5,8])
    # adeNodes[1].AddElementNodes(ID= 7, Nodes = [7,8,9])
    # adeNodes[1].AddElementNodes(ID= 8, Nodes = [7,9,10])
    #ADE Nodes at TimeStamp 2
    adeNodes[2] = df.ElementDefinition()
    adeNodes[2].AddElementNodes(ID= 6, Nodes = [1,2,3])
    adeNodes[2].AddElementNodes(ID= 5, Nodes = [4,3,2])
    adeNodes[2].AddElementNodes(ID= 4, Nodes = [4,5,3])
    adeNodes[2].AddElementNodes(ID= 3, Nodes = [7,3,5])
    adeNodes[2].AddElementNodes(ID= 1, Nodes = [11,8,5])
    adeNodes[2].AddElementNodes(ID= 2, Nodes = [7,5,8])
    # adeNodes[2].AddElementNodes(ID= 7, Nodes = [7,8,9])
    # adeNodes[2].AddElementNodes(ID= 8, Nodes = [7,9,10])
    

    adeMoveList = df.adeMoveList('data/moveList/Fahrt016ADE.txt')

    # adeMoveList.Split(maxStepsize=200, keepConnectionInfo=True)

    nodeList=[7]



if simulateReconfigurationGerbl:
    TimeStamp = 0

    adeNodes = {}
    adeNodes[TimeStamp] = df.ElementDefinition()
    adeNodes[TimeStamp].ReadMeshFile('data/MeshFiles/initial62exuATC2.txt')

    
    #adeNodes[0].nodeDefinition = {}
    adeMoveList = df.adeMoveList('data/moveList/configurationGerbl_full.txt')
    
    currentLine = df.adeMoveList()
    adeIDs = []
    nextTimeStamp = False
    for ID in adeNodes[TimeStamp].elements:
        adeIDs.append(ID)


    for i in range(1,adeMoveList.data.shape[0]):
        for j in range(len(adeIDs)):
            index = adeMoveList.GetIndexOfID(adeIDs[j])
            if index != None:
                if  adeMoveList.data[i,index+4] >= 3 or adeMoveList.data[i,index+5] >= 3 or adeMoveList.data[i,index+6] >= 3:
                    if adeMoveList.data[i,index+4] != adeMoveList.data[i-1,index+4] or adeMoveList.data[i,index+5] != adeMoveList.data[i-1,index+5] or adeMoveList.data[i,index+6] != adeMoveList.data[i-1,index+6]:
                        nextTimeStamp = True
        if nextTimeStamp == True:
            nextTimeStamp = False
            TimeStamp += 1
            currentLine.data = adeMoveList.data[i,:]
            adeNodes[TimeStamp] = df.ElementDefinition()
            adeNodes[TimeStamp].ConnectionListFromMoveListLine(adeIDs,currentLine)


    ### test?
    adeNodes[0].connectionList[1][1][2]=1

    config.simulation.setMeshWithCoordinateConst = True
    config.simulation.connectionInfoInMoveList = True
    
    
    
    nodeList=[]
    
    
    
    
if config.simulation.setMeshWithCoordinateConst:
    config.simulation.numberOfBoundaryPoints = 0 #change here number of boundary 


oGround = mbs.AddObject(ObjectGround(referencePosition = [0,0,0]))


computeDynamic=True
config.simulation.useMoveList = False
useInverseKinematic = True



config.simulation.boundaryConnectionInfoInMoveList = False




#Options for Visualization
config.simulation.frameList = False
config.simulation.showJointAxes=False

config.simulation.animation=False

config.simulation.showLabeling = False #shows nodeNumbers, ID and sideNumbers, simulationTime


visualizeTrajectory=False
visualizeSurfaces=False


def sensorList(nodeList, adeNodes, mbs):  
    sensorTCPList=[]
    for node in nodeList:
        getADEFromNodeNumber=next(iter(adeNodes.adePointsOfNode[node]))
        adeIndex012OfNodeFromADEID=adeNodes.nodeNumbersToAdeIndex[getADEFromNodeNumber][0]
        sensorTCP = mbs.AddSensor(SensorMarker(markerNumber=mbs.GetSensor(ADE[getADEFromNodeNumber]['sensors'][adeIndex012OfNodeFromADEID])['markerNumber'], outputVariableType=exu.OutputVariableType.Position, storeInternal=True))
        sensorTCPList +=[sensorTCP]
    return sensorTCPList







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


            # if simulate4ATCs:
            #     tol=0.003
            #     nodesInRectangle=findNodesInRectangle(adeNodes, P1 =np.array([0-tol,0-tol]), P2 = np.array([100+tol,0+tol]))       
            #     addBoundaryWithNodesInRectangle(adeNodes,nodesInRectangle)

            # if simulateTowerADEs:
            #     tol=0.003
            #     nodesInRectangle=findNodesInRectangle(adeNodes, P1 =np.array([0-tol,0-tol]), P2 = np.array([lengthInitX+tol,0+tol]))       
            #     addBoundaryWithNodesInRectangle(adeNodes,nodesInRectangle)
                            
            
            ##+++++++++++++++++++++++++++++++++++++++
            ## Simulation Settings
            simulationSettings = exu.SimulationSettings() #takes currently set values or default values
            # simulationSettings.timeIntegration.preStepPyExecute = 'MBSUserFunction()'
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
            
            SC.visualizationSettings.openGL.lineWidth=2 #maximum
            SC.visualizationSettings.openGL.lineSmooth=True
            SC.visualizationSettings.openGL.multiSampling = 4
            SC.visualizationSettings.general.drawCoordinateSystem = False
            #SC.visualizationSettings.window.renderWindowSize=[1600,1024]
            SC.visualizationSettings.window.renderWindowSize=[1600,1000]
            

            SC.visualizationSettings.contact.showSearchTree = True
            SC.visualizationSettings.contact.showSearchTreeCells = True
            SC.visualizationSettings.contact.showBoundingBoxes = True
            
            
            
            if visualizeTrajectory:
                SC.visualizationSettings.sensors.traces.showPositionTrace = True
            
            # simulationSettings.displayComputationTime= True
            ##++++++++++++++++++++++++++++++
            ##ANIMATIONS
            ##make images for animations (requires FFMPEG):
            ##requires a subfolder 'images'
            # if config.simulation.animation:
                # simulationSettings.solutionSettings.recordImagesInterval=0.04 #simulation.endTime/200
            
            
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
                # program switches file path to other folder
                
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
            
            
            
            ###################################################################
            ###################################################################
            ###################################################################
            # simulationSettings.timeIntegration.newton.useNewtonSolver = True
            # simulationSettings.timeIntegration.newton.useModifiedNewton = True
            #######################################################################
            # simulationSettings.staticSolver.stabilizerODE2term = 0.2
            # simulationSettings.linearSolverType = exu.LinearSolverType.EigenSparse #use for overconstrained systems
            # simulationSettings.linearSolverType = exu.LinearSolverType.EigenDense #use for overconstrained systems
            # simulationSettings.linearSolverSettings.ignoreSingularJacobian = True #use for overconstrained systems
            # simulationSettings.timeIntegration.adaptiveStep=False
            # simulationSettings.staticSolver.newton.numericalDifferentiation.jacobianConnectorDerivative=False
            
            # simulationSettings.timeIntegration.simulateInRealtime=True
            #######################################################################
            # simulationSettings.staticSolver.newton.relativeTolerance = 1e0
            # simulationSettings.staticSolver.newton.absoluteTolerance = 1e0
            # simulationSettings.displayStatistics = True       
            # simulationSettings.timeIntegration.newton.modifiedNewtonJacUpdatePerStep=False
            ###################################################################
            ###################################################################
            ###################################################################
            # simulationSettings.timeIntegration.verboseMode=0
            
            numberOfLoadSteps=1
            simulationSettings.staticSolver.numberOfLoadSteps = numberOfLoadSteps
            
            exu.SolveStatic(mbs, simulationSettings=simulationSettings)
            # exu.SolveDynamic(mbs, simulationSettings=simulationSettings)
                        

            solODE2initial = mbs.systemData.GetODE2Coordinates(configuration=exu.ConfigurationType.Current)
            mbs.systemData.SetODE2Coordinates(solODE2initial, configuration=exu.ConfigurationType.Initial)  #set old solution as initial value for new solution
             
            mbs.WaitForUserToContinue()    
            
            

            ##############################################################          
            
            # mbs.SetPreStepUserFunction(PreStepUserFunction)

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
            # simulationSettings.linearSolverType = exu.LinearSolverType.EigenSparse #sparse solver !!!!!!!!!!!!!!!
            ##############################################################
            simulationSettings.displayStatistics = False     
            simulationSettings.displayComputationTime = False  


            if config.simulation.solutionViewer: #use this to reload the solution and use SolutionViewer
                SC.visualizationSettings.general.autoFitScene=False #if reloaded view settings
                simulationSettings.solutionSettings.solutionWritePeriod = config.simulation.solutionWritePeriod
                simulationSettings.solutionSettings.writeSolutionToFile = True
                simulationSettings.solutionSettings.coordinatesSolutionFileName = config.simulation.solutionViewerFile
                simulationSettings.solutionSettings.writeFileFooter = False
                simulationSettings.solutionSettings.writeFileHeader = True
                
                
                if os.path.isfile(solutionFilePath):
                    os.remove(solutionFilePath)
                simulationSettings.solutionSettings.appendToFile = True
                simulationSettings.solutionSettings.sensorsAppendToFile =True
                


            
  
            mbs.SetPreStepUserFunction(PreStepUserFunction)

            ##+++++++++++++++++++++++++++++++++++++++
            ## Starting Simulation using Movelist           
            if config.simulation.useMoveList:
                config.simulation.useInverseKinematic=False
                ##+++++++++++++++++++++++++++++++++++++++
                ## Checking and Setting Connections of ADEs
                TimeStamp = 0

                for i in range(len(adeMoveList.data)-1):
                    mbs.variables['line']=i+1
                    SensorValuesTime[i+1]={}


                    nextTimeStamp = False
                    cnt=i

                    simulationSettings.timeIntegration.numberOfSteps = int(config.simulation.nrSteps)
                    endTime=cf.CalcPause(adeMoveList,cnt+1,minPauseVal = 0.2)
                    # config.simulation.endTime = endTime
                    simulationSettings.timeIntegration.endTime = endTime
                    
                    if cnt > 0:
                        ##+++++++++++++++++++++++++++++++++++++++
                        ## Checking if Connections changed with the MoveList
                        for j in range(len(adeIDs)):
                            index = adeMoveList.GetIndexOfID(adeIDs[j])
                            if index != None:
                                if config.simulation.connectionInfoInMoveList:
                                    if adeMoveList.data[cnt,index+4] != adeMoveList.data[cnt-1,index+4] or adeMoveList.data[cnt,index+5] != adeMoveList.data[cnt-1,index+5] or adeMoveList.data[cnt,index+6] != adeMoveList.data[cnt-1,index+6]:
                                        nextTimeStamp = True
                                else:
                                    if  adeMoveList.data[cnt,index+4] != 0 or adeMoveList.data[cnt,index+5] != 0 or adeMoveList.data[cnt,index+6] != 0: #for 6ADEs
                                        nextTimeStamp = True
                                
                        if nextTimeStamp == True:
                            nextTimeStamp = False
                            TimeStamp += 1

                            if TimeStamp > 0:
                                IDList = []
                                for ID in adeNodes[0].elements:
                                    IDList += [ID]
                                    ##+++++++++++++++++++++++++++++++++++++++
                                    ## Disconnecting Connections that are no longer Needed
                                    for k in range(len(adeNodes[TimeStamp-1].connectionList[ID])):
                                        if adeNodes[TimeStamp-1].connectionList[ID][k] not in adeNodes[TimeStamp].connectionList[ID] and (adeNodes[TimeStamp-1].connectionList[ID][k][2] in IDList or adeNodes[TimeStamp-1].connectionList[ID][k][2] <= -1):
                                            DisconnectADE(mbs,ADE,adeNodes[TimeStamp-1].connectionList[ID][k])
                                            if config.simulation.debugConnectionInfo:
                                                logging.info("Disconnected" + str(adeNodes[TimeStamp-1].connectionList[ID][k]))
                                    ##+++++++++++++++++++++++++++++++++++++++
                                    ## Connecting new Connections of ADEs
                                    for k in range(len(adeNodes[TimeStamp].connectionList[ID])):
                                        if adeNodes[TimeStamp].connectionList[ID][k] not in adeNodes[TimeStamp-1].connectionList[ID] and (adeNodes[TimeStamp].connectionList[ID][k][2] in IDList or adeNodes[TimeStamp].connectionList[ID][k][2] <= -1):
                                            ConnectADE(mbs,ADE,adeNodes[TimeStamp].connectionList[ID][k])
                                            if config.simulation.debugConnectionInfo:
                                                logging.info("Connected" + str(adeNodes[TimeStamp].connectionList[ID][k]))

                                mbs.AssembleLTGLists()
                                mbs.systemData.SetSystemState(sysStateList,configuration = exu.ConfigurationType.Initial)

                    ##+++++++++++++++++++++++++++++++++++++++
                    ## Simulation of Mesh
                    if computeDynamic:
                        simulationSettings.solutionSettings.solutionInformation = "MoveList Line " +str(i+2)
                        
                        if not config.simulation.SolveDynamic:                       
                            ###################################################################
                            ###################################################################
                            ###################################################################
                            # simulationSettings.timeIntegration.newton.useModifiedNewton = True
                            # simulationSettings.staticSolver.newton.relativeTolerance = 1e0
                            # simulationSettings.staticSolver.newton.absoluteTolerance = 1e0
                            # simulationSettings.displayStatistics = True  
                            # simulationSettings.timeIntegration.newton.modifiedNewtonJacUpdatePerStep=True
                            ###################################################################
                            ###################################################################
                            ###################################################################                         
                            numberOfLoadSteps=50
                            simulationSettings.staticSolver.numberOfLoadSteps = numberOfLoadSteps
                            exu.SolveStatic(mbs, simulationSettings = simulationSettings,showHints=False)
                            
                            timeTotal=timeTotal+mbs.sys['staticSolver'].timer.total
                            numberOfSteps=numberOfSteps+1

                        else:
                            # simulationSettings.timeIntegration.newton.useModifiedNewton = True
                            # simulationSettings.staticSolver.newton.relativeTolerance = 1e0
                            # simulationSettings.staticSolver.newton.absoluteTolerance = 1e0
                            # simulationSettings.displayStatistics = True   
                            
                            exu.SolveDynamic(mbs, simulationSettings = simulationSettings,showHints=False)
                            
                            timeTotal=timeTotal+mbs.sys['dynamicSolver'].timer.total
                            numberOfSteps=numberOfSteps+1
                        
                        
                        ## Sending new Sensor Values to Webserver
                        SensorValuesTime[cnt+1][0]={}
                        SensorValues[cnt+1] = {}
                        for ID in adeIDs:
                            SensorValues[cnt+1][ID] = np.asmatrix([mbs.GetSensorValues(ADE[ID]['sensors'][0])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][1])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][2])[0:2]])
                            
                            # sensorNumbers=np.asmatrix([ADE[ID]['sensors'][0],ADE[ID]['sensors'][1],ADE[ID]['sensors'][2]])
                            # mbs.GetSensor(ADE[ID]['sensors'][0])['markerNumber']
                                                     
                        # client.Put('SensorValues',pickle.dumps(SensorValues))
                        # SensorValuesTime[cnt+1][0]=SensorValues[cnt+1]

                    
                    # if not mbs.GetRenderEngineStopFlag():
                        sysStateList = mbs.systemData.GetSystemState()
                        mbs.systemData.SetSystemState(sysStateList,configuration = exu.ConfigurationType.Initial)

                    # else:
                        
                        # ##+++++++++++++++++++++++++++++++++++++++
                        # ## Resetting Simulation
                        # break


            if not useInverseKinematic:
                if displaySimulation and not mbs.GetRenderEngineStopFlag():  
                    mbs.WaitForUserToContinue()      
                    exu.StopRenderer() #safely close rendering window!
                try:
                    if config.simulation.solutionViewer:
                          sol = LoadSolutionFile(config.simulation.solutionViewerFile, safeMode=True)#, maxRows=100) 
                          SolutionViewer(mbs,sol) #can also be entered in 
                except:
                    logging.warning('Couldnt load Solution File')                
                    
            
            
            
            
            
            
            
            
            if useInverseKinematic:
                if config.simulation.useMoveList==False:
                    cnt=-1
    
                
                
                config.simulation.useInverseKinematic=True
                
                
                # if not config.simulation.useInverseKinematic:
                #     if displaySimulation and not mbs.GetRenderEngineStopFlag():  
                #         mbs.WaitForUserToContinue()      
                #         exu.StopRenderer() #safely close rendering window!
                #     try:
                #         # if config.simulation.solutionViewer:
                              # sol = LoadSolutionFile(config.simulation.solutionViewerFile, safeMode=True)#, maxRows=100) 
                              # SolutionViewer(mbs,sol) #can also be entered in 
                #     except:
                #         logging.warning('Couldnt load Solution File')
                        
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
    
                    ikObj, mbs2, sensorMbs2TCP = createKinematicModel(mbs, mbs2)
    
    
    
                    # test circle
                    r = 0.05
                    if simulateTowerADEs:
                        r=0.2
                    if simulateRabbit:
                        r=0.2
                    if simulateGripper:
                        r=0.1
                        phi = np.linspace(0, 2*np.pi, 10)
                    else:
                        phi = np.linspace(0, 2*np.pi, 10) 
                    
                    if simulateRabbit:
                        phi = np.linspace(0, 4*np.pi, 20) 
                        dxx=[]
                        for j in range(len(phi)-1): 
                            xOld = r* np.array([cos(phi[j]), np.sin(phi[j])])
                            xNew = r* np.array([cos(phi[j+1]), np.sin(phi[j+1])])
                            dxx += [xNew - xOld]   
                        
                    else:
                        dxx=[]
                        for j in range(len(phi)-1): 
                            xOld = r* np.array([-cos(phi[j]), np.sin(phi[j])])
                            xNew = r* np.array([-cos(phi[j+1]), np.sin(phi[j+1])])
                            dxx += [xNew - xOld]     
                    
                    
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
    
                        

                    detaX=np.array([0,0,0,0])
    
                    for k in range(len(dxx)):
                        dx=[]
                        for i in range(len(nodeList)):
                            if simulateGripper:
                                if i ==1:
                                    dxx[k][0]=-dxx[k][0]
                            dx+=list(dxx[k])#+list(dxx[k]) #+list(dxx[k]) #<- change value here to go
                        
                        dx1=dx[0:2] - detaX[0:2]
                        dx2=dx[2:] - detaX[2:]
                        
                        dx=list(dx1)+list(dx2)
                            
    
    
                        setConfigFileKinModel()  # get ActuatorLength from KinModel 
                        
                        ikObj.InverseKinematicsRelative(None, np.array(list(dx)))
                        
    
                        ### toDO change here for more nodes the error!!!!
                        iWantToGoToThisValue = copy.copy(mbs2.GetSensorValues(sensorMbs2TCP[0])[0:2])
                        iWantToGoToThisValue2 = copy.copy(mbs2.GetSensorValues(sensorMbs2TCP[1])[0:2])
                        ### toDO change here for more nodes the error!!!!
     
                        l1 = ikObj.GetActuatorLength()
    
                        if config.simulation.massPoints2DIdealRevoluteJoint2D:
                            refLength = (config.lengths.L0) 
                        else:
                            refLength = (config.lengths.L0-2*(config.lengths.L1+config.lengths.L2))
    
                        # map actuator length to SurrogateModel
                        actorList = [((np.array(l1)-refLength)*10000).tolist()]
                        # actorList = [(np.round(l1, 5)).tolist()]
                        
                        sortedActorStrokesList += actorList 

                        # for i in range(len(sortedActorStrokesList)-1):
                        mbs.variables['line']=cnt+1
                        SensorValuesTime[cnt+1]={} 

                        # endTime = 0.83
                        # actorTime = 0.83
                        # nrSteps = 5
               
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
                            # exu.SolveDynamic(mbs, simulationSettings,showHints=False) 
                            
                            timeTotal=timeTotal+mbs.sys['staticSolver'].timer.total
                            numberOfSteps=numberOfSteps+1                            
                            
                            
                            ## Sending new Sensor Values to Webserver
                            SensorValuesTime[cnt+1][0]={}
                            SensorValues[cnt+1] = {}
                            for ID in adeIDs:
                                SensorValues[cnt+1][ID] = np.asmatrix([mbs.GetSensorValues(ADE[ID]['sensors'][0])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][1])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][2])[0:2]])
                            # client.Put('SensorValues',pickle.dumps(SensorValues))                           
                            SensorValuesTime[cnt+1][0]=SensorValues[cnt+1]    
                            
                            for plotSensorMbs2TCP in sensorMbs2TCP:
                                plt.scatter(mbs2.GetSensorValues(plotSensorMbs2TCP)[0], mbs2.GetSensorValues(plotSensorMbs2TCP)[1], marker='d', color='red', s=48)
                                plt.scatter(iWantToGoToThisValue[0], iWantToGoToThisValue[1], marker='d', color='red', s=48)
                                
                                
                            for plotSensorTCP in sensorTCP:
                                plt.scatter( mbs.GetSensorValues(plotSensorTCP)[0], mbs.GetSensorValues(plotSensorTCP)[1] , marker='x', color='red', s=48)
                                

                            cnt=cnt+1

                            sysStateList = mbs.systemData.GetSystemState()
                            mbs.systemData.SetSystemState(sysStateList,configuration = exu.ConfigurationType.Initial)     
                            
                            ePosDiff=[0,0,0,0]
                            # ePosDiff+=iWantToGoToThisValue - mbs.GetSensorValues(sensorTCP[0])[0:2]]

                            correctLastStep=True
                            detaX=np.array([0,0,0,0])
                            if correctLastStep:
                                for i in range(2):
                                    
                                    mbs3 = SC.AddSystem()
                                    
                                    # averageSensorValues=AverageMeshPoints(SensorValues[cnt],adeNodes,draw = False)
                                   
                                    # for node in adeNodes.adePointsOfNode:
                                    #     ADENr=next(iter(adeNodes.adePointsOfNode[node]))
                                    #     index=adeNodes.adePointsOfNode[node][ADENr]
                                    #     adeNodes.nodeDefinition[node] = np.array(averageSensorValues[ADENr][index])[0]
                                    
                                    # adeNodes.SetListOfNodes()
                                    # adeNodes.SetConnectionList()
                                    # adeNodes.SetNodeNumbersToAde()
                                    # adeNodes.SetADEPointsOfNode()   
                                    
                                    
                                    # adeNodes.SetBaseNodes(adeNodes.elements[1][0:2])

                                    ikObj3, mbs3, sensorMbs3TCP = createKinematicModel(mbs, mbs3)   
                                    
                                    

                                    ### toDO change here for more nodes the error!!!!
                                    # dx = (iWantToGoToThisValue-mbs.GetSensorValues(sensorTCP[0])[0:2]) *2
                                    
                                    ePosDiff[0:2] = iWantToGoToThisValue - mbs.GetSensorValues(sensorTCP[0])[0:2]
                                    
                                    # dx= dx/np.linalg.norm(dx) *np.linalg.norm(dx) *2
                                    
                                    # dx2 = (iWantToGoToThisValue2-mbs.GetSensorValues(sensorTCP[1])[0:2]) *2
                                    ePosDiff[2:]  = iWantToGoToThisValue2 - mbs.GetSensorValues(sensorTCP[1])[0:2]
                                    # print('desired: ', iWantToGoToThisValue)
                                    # print('surrogate: ', mbs.GetSensorValues(sensorTCP[0])[0:2])
                                    # print('eposDiff ikine model:', ePosDiff[0:2] )
                                    
                                    # dx2= dx2/np.linalg.norm(dx2) *np.linalg.norm(dx2) *2
                                    
                                    detaX=detaX+np.array(ePosDiff)
                                    # dxList=[]
                                    # for i in range(len(nodeList)):
                                    #     dxList+=list(dx)#+list(dxx[k]) #+list(dxx[k]) #<- change value here to go
                                    
                                    # dxList=[list(dx1)+list(dx2)][0]
                                    # dxList=[list(dx)][0]


                                    # print('offset dx:',dx)
                                    ikObj.InverseKinematicsRelative(None, np.array(ePosDiff))
                                    ### toDO change here for more nodes the error!!!!
                                    
                                    
                                    l1 = ikObj.GetActuatorLength()
                                    
                                    
                                    if config.simulation.massPoints2DIdealRevoluteJoint2D:
                                        refLength = (config.lengths.L0) 
                                    else:
                                        refLength = (config.lengths.L0-2*(config.lengths.L1+config.lengths.L2))
                                        
                                    actorList = [((np.array(l1)-refLength)*10000).tolist()]
                                    # actorList = [(np.round(l1, 5)).tolist()]
                                    
                                    # sortedActorStrokesList += actorList 
                                    # actorList= np.array(sortedActorStrokesList[-1]) + (np.array(sortedActorStrokesList[-1])-np.array(actorList)) 
                                    

                                    
                                    
                                    # print(actorList)
                                    
                                    sortedActorStrokesList += [list(actorList[0])]
                                    # sortedActorStrokesList[cnt] = actorList[0] 

                                    # actorList = [(np.round(l1, 5)*10000).tolist()]
                                    # sortedActorStrokesList[cnt] = actorList[0] 
                                    # mbs3.Reset()
                                    
                                    
                                    mbs.variables['line']=cnt+1
                                    SensorValuesTime[cnt+1]={} 

                                    # l1 = ikObj3.GetActuatorLength()
                                    # actorList = [(np.round(l1, 5)*10000).tolist()]
                                    # sortedActorStrokesList[cnt+1] = actorList[0] 
                                    # mbs3.Reset()
                                    
                                    sysStateList = mbs.systemData.GetSystemState()
                                    mbs.systemData.SetSystemState(sysStateList,configuration = exu.ConfigurationType.Initial)
                                    
                                    setConfigFileSurrogateModel() 
                                    numberOfLoadSteps=100
                                    simulationSettings.staticSolver.numberOfLoadSteps = numberOfLoadSteps
                                    
                                    # simulationSettings.staticSolver.newton.relativeTolerance = 1e-4
                                    # simulationSettings.staticSolver.newton.absoluteTolerance = 1e-4
                                    # simulationSettings.staticSolver.newton.newtonResidualMode = 10 #take Newton increment
                                    
                                    
                                    # mbs.SetPreStepUserFunction(PreStepUserFunction)
                                    exu.SolveStatic(mbs, simulationSettings,showHints=False) 
                                    # exu.SolveDynamic(mbs, simulationSettings = simulationSettings,showHints=False)
            
                                    timeTotal=timeTotal+mbs.sys['staticSolver'].timer.total
                                    numberOfSteps=numberOfSteps+1
                                    
                                    # ## Sending new Sensor Values to Webserver
                                    # SensorValuesTime[cnt+1][0]={}
                                    # SensorValues[cnt+1] = {}
                                    # for ID in adeIDs:
                                    #     SensorValues[cnt+1][ID] = np.asmatrix([mbs.GetSensorValues(ADE[ID]['sensors'][0])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][1])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][2])[0:2]])
                                    # # client.Put('SensorValues',pickle.dumps(SensorValues))                           
                                    # SensorValuesTime[cnt+1][0]=SensorValues[cnt+1]     
                                    SensorValuesTime[cnt][0]={}
                                    SensorValues[cnt] = {}
                                    for ID in adeIDs:
                                        SensorValues[cnt][ID] = np.asmatrix([mbs.GetSensorValues(ADE[ID]['sensors'][0])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][1])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][2])[0:2]])
                                    # client.Put('SensorValues',pickle.dumps(SensorValues))                           
                                    SensorValuesTime[cnt][0]=SensorValues[cnt]  
                                    
                                    
                                    # for plotSensorMbs3TCP in sensorMbs3TCP:
                                        # plt.scatter( mbs3.GetSensorValues(plotSensorMbs3TCP)[0], mbs3.GetSensorValues(plotSensorMbs3TCP)[1], marker='d', color=mycolors[i] )
                                        
                                    for plotSensorTCP in sensorTCP:
                                        plt.scatter( mbs.GetSensorValues(plotSensorTCP)[0], mbs.GetSensorValues(plotSensorTCP)[1] , marker='x', color=mycolors[i], s=48)
                                        # plt.show()
                                    mbs3.Reset()
                                    cnt=cnt+1

                            
                            
                            
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
                                
                        # plt.scatter( mbs2.GetSensorValues(sensorMbs2TCP)[0], mbs2.GetSensorValues(sensorMbs2TCP)[1], marker='o' )
                        # plt.scatter( mbs.GetSensorValues(sensorTCP)[0], mbs.GetSensorValues(sensorTCP)[1] , marker='x')
                        # averageSensorValues=AverageMeshPoints(SensorValues[cnt-1],adeNodes,draw = True)
                        
                        
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
            ## Saving Sensor Data to File
            if config.simulation.saveSensorValues:
                  pickle.dump(  SensorValues,  open(config.simulation.sensorValuesFile,  'wb')  )
                  pickle.dump(  SensorValuesTime,  open(config.simulation.sensorValuesTimeFile,  'wb')  )
                  
                  pickle.dump(  SensorValues2,  open(config.simulation.sensorValuesFile2,  'wb')  )
                  pickle.dump(  SensorValuesTime2,  open(config.simulation.sensorValuesTimeFile2,  'wb')  )
                 
            ##+++++++++++++++++++++++++++++++++++++++
            ## Resetting Simulation
            mbs.Reset()
            startSimulation = False

            break
    
    except Exception as ExeptionMessage: 
        logging.warning('Error ocured during Simulation')
        logging.warning(ExeptionMessage)
        startSimulation = False
        #client.Put('StartSimulation',json.dumps(False))
        mbs.Reset()
    time.sleep(config.simulation.looptime-time.time()%config.simulation.looptime)

        


print('time: ', timeTotal, 'numberOfSteps: ',numberOfSteps)

# try:
#     client.Put('EndProgram',json.dumps(True))
# except:
#     os._exit(0)

print('userFunction: ',mbs.variables['counterForTesting'])


# mbs.variables['counterForTesting']=0

#%% Ploten von zuvor gespeicherten Sensorwerten:
# plt.close('all')

# from functions import visualization_functions as vf
# from functions.control_functions import *
# import matplotlib.ticker as ticker
# # ### 

# SensorValues4ATCRBSD = pickle.load( open('output/SensorValuesTime.pickle',  'rb') ) # Laden von Sensor Werten
# SensorValues4ATCRBSDTime = SensorValues4ATCRBSD

# accumulatedTime=0
# for line in SensorValues4ATCRBSD:
#     for time in SensorValues4ATCRBSD[line]:
        
#         dT=SensorValues4ATCRBSD[line][time]-accumulatedTime
#         accumulatedTime+=dT
        
        
#         SensorValues4ATCRBSDTime[line][time]=accumulatedTime
    
    


# #%% Ploten von zuvor gespeicherten Sensorwerten:
# plt.close('all')

# from functions import visualization_functions as vf
# from functions.control_functions import *
# # ### 
# SensorValues4ATCcrossSprings = pickle.load( open('output/SensorValues4ATCcrossSprings.pickle',  'rb') ) # Laden von Sensor Werten
# SensorValues4ATCcircularHinge = pickle.load( open('output/SensorValues4ATCcircularHinge.pickle',  'rb') ) # Laden von Sensor Werten

# # SensorValues4ATCRBSD = pickle.load( open('output/SensorValues4ATCRBSD.pickle',  'rb') ) # Laden von Sensor Werten
# SensorValues4ATCRBSD = pickle.load( open('output/SensorValues.pickle',  'rb') ) # Laden von Sensor Werten

# # vf.PlotSensorValues(SensorValues4ATCcrossSprings,combinePlots=True,figureName='Fig1')           # Ploten von allen Werten


# linestyleList=['-','--','-.',':','.',',','o','v','^','<','>','1','2','3','4','s','p','*','h','H','+','x','D','d','_']
# markerstyleList=['.',',','o','v','^','<','>','1','2','3','4','8','s','p','P','*','h','H','+','x','X','d','D']
# colorlineList=['b','g','r','c','m','y','k','b','g','r','c','m','y','k']


# SensorValues4ATCRBSD2 = pickle.load( open('output/SensorValuesTime.pickle',  'rb') ) # Laden von Sensor Werten

# #%% sortTime
# SensorValues4ATCRBSDTime = {}
# accumulatedTime=0
# for line in SensorValues4ATCRBSD2:
#     SensorValues4ATCRBSDTime[line]={}
#     lastTime=0
#     for newTime in SensorValues4ATCRBSD2[line]:
#         dT=newTime-lastTime    
#         accumulatedTime+=dT        
#         SensorValues4ATCRBSDTime[line][accumulatedTime]=SensorValues4ATCRBSD2[line][newTime]  
#         lastTime=newTime
        
        
# for line in SensorValues4ATCRBSDTime:
#     for time in SensorValues4ATCRBSDTime[line]:
#         SensorValues4ATCRBSDTime[line][time] = AverageMeshPoints(SensorValues4ATCRBSDTime[line][time],adeNodes,draw=False)












# adeIDList = []
# for ID in adeNodes.elements:
#         adeIDList.append(ID)

# idealMesh = {}
# for line in SensorValues4ATCcrossSprings:     
#     currentLine = df.adeMoveList()
#     currentLine.data=adeMoveList.data[line,:]
#     mef.Stroke2desiredMesh(adeIDList,boundary = np.matrix([[0,-config.lengths.boundaryYdistance,0,0.01,-config.lengths.boundaryYdistance,0]]),strokeLength= currentLine,adeNodes = adeNodes,draw = False)

#     idealMesh[line] = {}
#     for ID in adeNodes.elements:
#         idealMesh[line][ID] = [adeNodes.nodeDefinition[adeNodes.elements[ID][0]],adeNodes.nodeDefinition[adeNodes.elements[ID][1]],adeNodes.nodeDefinition[adeNodes.elements[ID][2]]]


# import matplotlib.ticker as ticker
# # 'data//experiments//supportStructure//Image//7//MeshPoints.txt'
# # 'data//experiments//4ADEs#//uncorrected//MeshPoints.txt'
# meshPoints = {}
# with open('data//experiments//4ADEs#//uncorrected//MeshPoints.txt') as f:
#     lines = f.readlines()
#     for i,line in enumerate(lines):
#         meshPoints[i] = {}
#         line = line.split()
#         for j in range(0,len(line),7):
#             meshPoints[i][int(line[j])] = {}
#             for k in range(3):
#                 meshPoints[i][int(line[j])][k] = np.matrix(line[j+k*2+1:j+k*2+3],dtype=float)/1000
   
                
# meshPointsVicon = {}
# with open('data//experiments//4ADEs#//uncorrected//MeshPointsVicon.txt') as f:
#     lines = f.readlines()
#     for i,line in enumerate(lines):
#         meshPointsVicon[i] = {}
#         line = line.split()
#         for j in range(0,len(line),7):
#             meshPointsVicon[i][int(line[j])] = {}
#             for k in range(3):
#                 meshPointsVicon[i][int(line[j])][k] = np.matrix(line[j+k*2+1:j+k*2+3],dtype=float)



# for line in idealMesh:
#     SensorValues4ATCcircularHinge[line] = AverageMeshPoints(SensorValues4ATCcircularHinge[line],adeNodes,draw=False)
#     SensorValues4ATCcrossSprings[line] = AverageMeshPoints(SensorValues4ATCcrossSprings[line],adeNodes,draw=False)
#     SensorValues4ATCRBSD[line] = AverageMeshPoints(SensorValues4ATCRBSD[line],adeNodes,draw=False)
    
#     idealMesh[line] = AverageMeshPoints(idealMesh[line],adeNodes,draw=False)
    
#     meshPoints[line] = AverageMeshPoints(meshPoints[line],adeNodes,draw=False)
#     meshPointsVicon[line] = AverageMeshPoints(meshPointsVicon[line],adeNodes,draw=False)


# def MoveToOrigin(sensorValues,adeNodes,ID,Side):
#         Nodes = adeNodes.GetNodeOfElementSide(ID,Side)
#         index1 = adeNodes.elements[ID].index(Nodes[0])
#         index2 = adeNodes.elements[ID].index(Nodes[1])
#         P1=sensorValues[0][ID][index1]
#         P2=sensorValues[0][ID][index2]
        
#         if LA.norm(P2) < LA.norm(P1):
#             P1,P2 = P2,P1

#         Bound1=P1.T
#         Bound2=P2.T#P1+(P2-P1)/LA.norm(P2-P1)*(B1-29)
#         an = (P2-P1)/LA.norm(P2-P1) 
        
#         ex = np.array([[1],[0]])
#         bound = P2-P1
#         alpha = -math.acos(an[0,0])

#         P1=P1.T
        
#         for line in sensorValues:
#             for ID in sensorValues[line]:
#                 for i in range(3):
#                     sensorValues[line][ID][i] = np.asarray(np.matmul(mef.RotAz(alpha),np.matrix(sensorValues[line][ID][i].T-P1))).T

# #moveMeshPoints to origin
# MoveToOrigin(meshPoints,adeNodes,5,3)
# MoveToOrigin(meshPointsVicon,adeNodes,5,3)


# startingNodes=adeNodes.nodeDefinition


# idealMesh2=idealMesh[0]
# nodes2=adeNodes.nodeDefinition



        
# errorModelIdealList=[]
# errorExperimentIdealList=[]
# errorExperimentModelList=[]


# Fig2=plt.figure('Trajectory')
# ax=Fig2.gca() # get current axes
# for ID in adeNodes.elements:
#     for Point in range(3):
#         for line in SensorValues4ATCRBSDTime:
#             SensorValues4ATCRBSDLine=SensorValues4ATCRBSDTime[line]
#             vf.PlotSensorValuePoint(SensorValues4ATCRBSDLine,ID,Point,linecolor = colorlineList[ID] ,markercolor = colorlineList[ID],marker=markerstyleList[Point], figureName = 'Trajectory',label='',combinePlots=True,plotMarker=False)


# #%% Figure 2 Plot trajectory of Point
# for pointOfADEToPlot in adeNodes.nodeNumbersToAdeIndex:
#     for pointToPlot in adeNodes.nodeNumbersToAdeIndex[pointOfADEToPlot]:           
        
#         substractOffset=False
#         lineToStart=0 #define line/step = 0 if substractOffset = True
        
#         if substractOffset:     
#             offsetIdeal=vf.calculateOffset(idealMesh,idealMesh,lineToStart,pointOfADEToPlot,pointToPlot) # == 0
            
#             offsetRBSD=vf.calculateOffset(idealMesh,SensorValues4ATCRBSD,lineToStart,pointOfADEToPlot,pointToPlot)
#             offsetCircularHinge=vf.calculateOffset(idealMesh,SensorValues4ATCcircularHinge,lineToStart,pointOfADEToPlot,pointToPlot)
#             offsetCrossSprings=vf.calculateOffset(idealMesh,SensorValues4ATCcrossSprings,lineToStart,pointOfADEToPlot,pointToPlot)
            
#             offsetExperiment=vf.calculateOffset(idealMesh,meshPoints,lineToStart,pointOfADEToPlot,pointToPlot)
#             offsetExperimentVicon=vf.calculateOffset(idealMesh,meshPointsVicon,lineToStart,pointOfADEToPlot,pointToPlot)
#         else:
#             offsetIdeal=np.matrix([[0, 0]])
            
#             offsetRBSD=np.matrix([[0, 0]])
#             offsetCircularHinge=np.matrix([[0, 0]])
#             offsetCrossSprings=np.matrix([[0, 0]])
            
#             offsetExperiment=np.matrix([[0, 0]]) 
#             offsetExperimentVicon=np.matrix([[0, 0]]) 
        

#         vf.substractOffset(SensorValues4ATCRBSD,pointOfADEToPlot,pointToPlot,offsetRBSD,combinePlots=True,figureName='Trajectory',linecolor = 'orange',markercolor='orange',marker='x',label='RBSD') 
#         # vf.substractOffset(SensorValues4ATCcircularHinge,pointOfADEToPlot,pointToPlot,offsetCircularHinge,combinePlots=True,figureName='Trajectory',linecolor = 'orange' ,markercolor='orange',marker='x',label='circularHinges') 
#         # vf.substractOffset(meshPointsVicon,pointOfADEToPlot,pointToPlot,offsetExperimentVicon,combinePlots=True,figureName='Trajectory',linecolor = 'blue' ,marker='o',markercolor='blue',label='experiment circularHinges')
        
#         vf.substractOffset(idealMesh,pointOfADEToPlot,pointToPlot,offsetIdeal,combinePlots=True,figureName='Trajectory',linecolor = 'red' ,marker='^',markercolor='red',label='ideal')
#         # vf.substractOffset(SensorValues4ATCcrossSprings,pointOfADEToPlot,pointToPlot,offsetCrossSprings,combinePlots=True,figureName='Trajectory',linecolor = 'orange' ,markercolor='orange',marker='x',label='crossSprings') 
#         vf.substractOffset(meshPoints,pointOfADEToPlot,pointToPlot,offsetExperiment,combinePlots=True,figureName='Trajectory',linecolor = 'blue' ,marker='o',markercolor='blue',label='experiment crossSprings')
        
#         plt.legend()
#         plt.grid()
#         ax.grid(True, 'major', 'both')
#         ax.xaxis.set_major_locator(ticker.MaxNLocator(10)) 
#         ax.yaxis.set_major_locator(ticker.MaxNLocator(10)) 
        
#         ax.set_ylabel(r'measuring points y in m')
#         ax.set_xlabel(r'measuring points x in m')
#         plt.show()
        
        
                
#         # %% Figure 3 Plot error
#         ##plot error Point P6 in mm
#         figName='ADE'+str(pointOfADEToPlot)+'point'+str(pointToPlot)+'nodeNumber:'+str(adeNodes.elements[pointOfADEToPlot][pointToPlot])
#         Fig3=plt.figure(figName)
#         ax=Fig3.gca() # get current axes
        
#         errorModelIdeal=vf.plotError(SensorValues4ATCRBSD,idealMesh,pointOfADEToPlot,pointToPlot,offsetRBSD,offsetIdeal,combinePlots=True,figureName=figName,linecolor = 'blue' ,marker='o',markercolor='blue',label='model-ideal')
#         # vf.plotError(meshPointsVicon,idealMesh,pointOfADEToPlot,pointToPlot,offsetExperimentVicon,offsetIdeal,combinePlots=True,figureName='Fig3',linecolor = 'red' ,marker='^',markercolor='red',label='experiment-ideal - circularHinges')
#         # vf.plotError(meshPointsVicon,SensorValues4ATCRBSD,pointOfADEToPlot,pointToPlot,offsetExperimentVicon,offsetRBSD,combinePlots=True,figureName='Fig3',linecolor = 'orange' ,marker='x',markercolor='orange',label='experiment-model circularHinges')
        
#         errorExperimentIdeal=vf.plotError(meshPoints,idealMesh,pointOfADEToPlot,pointToPlot,offsetExperiment,offsetIdeal,combinePlots=True,figureName=figName,linecolor = 'red' ,marker='^',markercolor='red',label='experiment-ideal')
#         errorExperimentModel=vf.plotError(meshPoints,SensorValues4ATCRBSD,pointOfADEToPlot,pointToPlot,offsetExperiment,offsetRBSD,combinePlots=True,figureName=figName,linecolor = 'orange' ,marker='x',markercolor='orange',label='experiment-model')
        
#         print('nodeNumber:'+str(adeNodes.elements[pointOfADEToPlot][pointToPlot]))
#         print('errorModelIdeal=',errorModelIdeal,'m')
#         print('errorExperimentIdeal=',errorExperimentIdeal,'m')
#         print('errorExperimentModel=',errorExperimentModel,'m')
        
#         # plt.figure('Fig3')
#         # plt.legend(["experiment-ideal","experiment-model","model-ideal"])
#         plt.legend()
#         plt.grid()
#         ax.grid(True, 'major', 'both')
#         ax.xaxis.set_major_locator(ticker.MaxNLocator(10)) 
#         ax.yaxis.set_major_locator(ticker.MaxNLocator(10)) 
        
#         plt.xlim(-1, len(SensorValues4ATCRBSD))
        
#         ax.set_ylabel(r'point P6 error in m')
#         ax.set_xlabel(r'steps')
#         plt.show()
        
#         errorModelIdealList+=[errorModelIdeal]
#         errorExperimentIdealList+=[errorExperimentIdeal]
#         errorExperimentModelList+=[errorExperimentModel]


# meanErrorModelIdealList=np.mean(errorModelIdealList)
# meanErrorExperimentIdealList=np.mean(errorExperimentIdealList)
# meanErrorExperimentModelList=np.mean(errorExperimentModelList)
# print('meanError:')
# print('meanErrorModelIdeal=',meanErrorModelIdealList,'m')
# print('meanErrorExperimentIdeal=',meanErrorExperimentIdealList,'m')
# print('meanErrorExperimentModel=',meanErrorExperimentModelList,'m')


        
# #%% Figure 4 Plot mesh of step (line)    
# length = len(meshPointsVicon)
# length = 1 

# for line in range(length):        
#     line=1 #line to plot
#     ##+++++++++++++++++++++++++++++++++++++++
#     ## Plotting Mesh over Measured Image
#     logging.info('Line '+str(line)+':  Plotting Mesh over Measured Image')
    
#     currentAvgMeshPoints = {}
#     currentAvgMeshPoints[0] = meshPoints[line]
    
#     currentAvgMeshPointsVicon = {}
#     currentAvgMeshPointsVicon[0] = meshPointsVicon[line]
    
#     currentAvgSensorValuesRBSD = {}
#     currentAvgSensorValuesRBSD[0] = SensorValues4ATCRBSD[line]
    
#     currentAvgSensorValuesCircularHinge = {}
#     currentAvgSensorValuesCircularHinge[0] = SensorValues4ATCcircularHinge[line]
    
#     currentAvgSensorValuesCrossSprings = {}
#     currentAvgSensorValuesCrossSprings[0] = SensorValues4ATCcrossSprings[line]
#     ##Section 1
    
#     ##plot mesh
#     figName='step: '+str(line)
#     Fig4=plt.figure(figName)
#     ax=Fig4.gca() # get current axes
    
    
#     currentIdealMesh = {}
#     currentIdealMesh[0] = idealMesh[line]
    
    
#     vf.PlotSensorValues(currentAvgSensorValuesRBSD,markercolor = 'orange',marker = 'x',figureName = figName,label='RBSD',combinePlots=True)
#     # vf.PlotSensorValues(currentAvgSensorValuesCircularHinge,markercolor = 'orange',marker = 'x',figureName = 'Fig4',label='circularHinge',combinePlots=True)
    
#     vf.PlotSensorValues(currentIdealMesh,markercolor = 'red',marker = '^',figureName = figName,label='ideal',combinePlots=True)
    
#     # vf.PlotSensorValues(currentAvgMeshPointsVicon,markercolor = 'blue',marker = 'o',figureName = 'Fig4',label='experiment',combinePlots=True)
    
#     # vf.PlotSensorValues(currentAvgSensorValuesCrossSprings,markercolor = 'orange',marker = 'x',figureName = 'Fig4',label='crossSprings',combinePlots=True)
#     vf.PlotSensorValues(currentAvgMeshPoints,markercolor = 'blue',marker = 'o',figureName = figName,label='experiment',combinePlots=True)
    
    
#     plt.legend()
#     plt.grid()
#     plt.axis('equal')
#     ax.grid(True, 'major', 'both')
#     ax.xaxis.set_major_locator(ticker.MaxNLocator(10)) 
#     ax.yaxis.set_major_locator(ticker.MaxNLocator(10)) 
    
#     ax.set_ylabel(r'nodes y in m')
#     ax.set_xlabel(r'nodes x in m')
#     plt.show()
        
 
