#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if __name__ == "__main__":
    if os.getcwd().split('\\')[-1].__contains__('exampleScripts'):
        os.chdir('..')
    if sys.path[0].split('\\')[-1].__contains__('exampleScripts'):
        sys.path.append(os.getcwd())


solutionFilePath="./solution/coordinatesSolution.txt"
if os.path.isfile(solutionFilePath):
    os.remove(solutionFilePath)

solutionFilePath="./solution/coordinatesSolutionIKIN.txt"
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
import functions.trajectory_functions as tf

import exudyn as exu
from exudyn.utilities import*
from exudyn.interactive import SolutionViewer

import numpy as np
import matplotlib
matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["STIX Two Text"]
})
import matplotlib.pyplot as plt
from numpy import absolute, linalg as LA
import math
import logging
from copy import deepcopy
import time
import pickle
import json
from enum import Enum, auto
import csv


from functions import visualization_functions as vf
from functions.control_functions import *
import matplotlib.ticker as ticker

# check if newly added side result in a closed loop:
def checkClosedLoops(oldListOfADEs,newADEID,connect2ADEID):    
    newList=oldListOfADEs
    first_key=newADEID
    first_value=connect2ADEID

    closedLoop=False
    
    visitedADEID=[]
    for i in first_value:
        searchList=newList.copy()
        visitedADEID+=[i]
        if i in searchList:
            
            for j in searchList[i]:                
                
                if j in searchList:
                    visitedADEID+=[j]
                    if i in searchList:
                        
                        del searchList[i]
                    # print('index j:', j)
                    # print('serchList:', searchList)
                    
                    for k in visitedADEID:
                        if len(visitedADEID)!=2:
                            
                            if k==first_key:
                                # print()
                                # print('oldlistOfADEs:',oldListOfADEs)
                                # print('newADEID:',connect2ADEID,'connect2ADEID:',newADEID)
                                closedLoop=True
                                # print('visitedADE ID:',visitedADEID)
                                # print('closed loooop!!!1111')

    return closedLoop

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
            
            if i == 11 and t == 0.0332: 
                print('cnt={}: move actor {} from {} to {} '.format(cnt, i, u1, u2), 'correction' * mbs.variables['correction'])
            oSpringDamper = mbs.variables['springDamperDriveObjects'][i]
            driveLen = UserFunctionDriveRefLen1(t, u1,u2)
            
            # mbs.SetObjectParameter(oSpringDamper,'referenceLength',driveLen+refLength)  
            mbs.SetObjectParameter(oSpringDamper,'referenceLength',driveLen+refLength)#LrefExt[i])
        
        
    mbsV.SendRedrawSignal()
    
    return True #succeeded



def setConfigFileKinModel():
    config.simulation.useRevoluteJoint2DTower = True
    config.simulation.setMeshWithCoordinateConst = True
    
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

    config.lengths.stiffnessPrismaticJoints = 1
    config.lengths.dampingPrismaticJoints = 5e3

    if config.simulation.useCompliantJoints or config.simulation.useCrossSpringPivot:
        config.simulation.graphics = 'graphicsSTL'
    else:
        config.simulation.graphics = 'graphicsStandard'    




def setConfigFileSurrogateModel():
    
    config.simulation.useRevoluteJoint2DTower = True #revoluteJoints are used to connect triangles with other triangles and to ground, only works if boundary is set with coordinateConstraints <--- toDo change to GenericJoints!!!
    config.simulation.setMeshWithCoordinateConst = True #<--- toDo combine with useRevoluteJoint2DTower to GenericJoints!!!
    
    #choose one option to create six-bar linkages, if all set to False, 
    #RevoluteJoint2D are used for joints for the six-bar linkages
    config.simulation.simplifiedLinksIdealRevoluteJoint2D = False #simplified model using rigidBodies for the six-bar linkages and revoluteJoints at edges
    config.simulation.simplifiedLinksIdealCartesianSpringDamper = False
    config.simulation.massPoints2DIdealRevoluteJoint2D = False #simplified model using only massPoints and revoluteJoints at edges
    
    
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

    if config.simulation.useCompliantJoints or config.simulation.useCrossSpringPivot:
        config.simulation.graphics = 'graphicsSTL'
    else:
        config.simulation.graphics = 'graphicsStandard'

    config.simulation.SolveDynamic = True



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
    mbs2.variables['sensorTCP']=sensorMbs2TCPList
    
    oGround2 = mbs2.AddObject(ObjectGround(referencePosition = [0,0,0]))
   
    mbs2.Assemble()
   
    # if type(adeNodes) == dict:
    #     for ID in adeNodes[0].elements:
    #         for i in range(len(adeNodes[0].connectionList[ID])):
    #             if adeNodes[0].connectionList[ID][i][2] in ADE2:
    #                 ConnectADE(mbs2,ADE2,adeNodes[0].connectionList[ID][i])
    # else:
    #     for ID in adeNodes.elements:
    #         for i in range(len(adeNodes.connectionList[ID])):
    #             if adeNodes.connectionList[ID][i][2] in ADE2:
    #                 ConnectADE(mbs2,ADE2,adeNodes.connectionList[ID][i])                
    
    #########################################################################   
    #########################################################################  
    shortConList={}
    for items in adeNodes.connectionList:
        shortConList[items]=[]
        for conList in adeNodes.connectionList[items]: 
            shortConList[items].append(conList[2])

    newADEID=[]
    connect2ADEID=[]
    for items in shortConList:
        for adeIDss in shortConList[items]:
            newADEID+=[adeIDss]
            connect2ADEID+=[items]
    newADEID+=[0]       
    connect2ADEID+=[0]   
      
    oldList={}
    cnt=0
    for items in shortConList:
        oldList[items]=[]
        for adeIDss in shortConList[items]:
            oldList[items]+=[adeIDss]
            cnt+=1
            # print(oldList)
            isLoop=checkClosedLoops(oldList,connect2ADEID[cnt],[newADEID[cnt]])  
            
            if connect2ADEID[cnt]!=0:
                currentConnectionList=[]
                for i in adeNodes.connectionList[connect2ADEID[cnt]]:
                    if i[2]==newADEID[cnt]:
                        currentConnectionList=i
                        
                if isLoop:
                    print('surrogate loop!')
                    ConnectADELoop(mbs2,ADE2,currentConnectionList)         
                else:
                    ConnectADE(mbs2,ADE2,currentConnectionList)
    ######################################################################### 
    #########################################################################    

    
    
    ikObj = InverseKinematicsATE(mbs2, sensorMbs2TCPList, mbs, oGround2, ADE2)
                  
    return ikObj, mbs2, sensorMbs2TCPList




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


    mbs.Assemble()

    shortConList={}
    for items in adeNodes.connectionList:
        shortConList[items]=[]
        for conList in adeNodes.connectionList[items]: 
            shortConList[items].append(conList[2])

    newADEID=[]
    connect2ADEID=[]
    for items in shortConList:
        for adeIDss in shortConList[items]:
            newADEID+=[adeIDss]
            connect2ADEID+=[items]
    newADEID+=[0]       
    connect2ADEID+=[0]   
      
    oldList={}
    cnt=0
    for items in shortConList:
        oldList[items]=[]
        for adeIDss in shortConList[items]:
            oldList[items]+=[adeIDss]
            cnt+=1
            # print(oldList)
            isLoop=checkClosedLoops(oldList,connect2ADEID[cnt],[newADEID[cnt]])  
            
            if connect2ADEID[cnt]!=0:
                currentConnectionList=[]
                for i in adeNodes.connectionList[connect2ADEID[cnt]]:
                    if i[2]==newADEID[cnt]:
                        currentConnectionList=i
                        
                if isLoop:
                    
                    if config.simulation.useRevoluteJoint2DTower:
                        print('surrogate loop!')
                        ConnectADELoop(mbs,ADE,currentConnectionList) 
                    else:
                        ConnectADE(mbs,ADE,currentConnectionList)
                else:
                    ConnectADE(mbs,ADE,currentConnectionList)

    
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



timeTotal=False
numberOfSteps=0



config.simulation.constrainActorLength = True



##simulation options
displaySimulation=True

simulate4ATCs=False
simulateGripper=True 
simulateTowerADEs=False
simulateRabbit=False
simulateDuckBridgeCraneTower=False
simulateReconfigurationGerbl=False
##++++++++++++++++++++++++++++++
##ANIMATIONS
##make images for animations (requires FFMPEG):
my_dir = os.path.abspath('./functions')
# os.system('start /MIN cmd /C  python "'+os.path.join(my_dir, 'webserver_functions.py')+'"')    
# os.system('start /MIN cmd /C python "'+os.path.join(my_dir, 'measurement_functions.py')+'"')

contactON = True


if simulateGripper:
    if contactON:
        ##++++++++++++++++++++++++++++++
        gContact = mbs.AddGeneralContact()
        gContact.verboseMode = 1
        mu = 0.2*0      #dry friction
    
        gContact.frictionProportionalZone = 1e-3
        # gContact.excludeDuplicatedTrigSphereContactPoints = False
        fricMat = mu*np.eye(1)
        
        
        gContact.SetFrictionPairings(fricMat)
        gContact.SetSearchTreeBox(pMin=[-0.5,0,-0.2],pMax=[0.5,1.5,0.2])
        gContact.SetSearchTreeCellSize(numberOfCells=[5,5,1]) 
        
        sizeOfCylinder=0.08
        sizeOfSpheres=0.05
        
        frictionMaterialIndex=0
        
        #stiffness of spheres
        contactSteel = 0.001 #N/m (linear) normal contact stiffness
        dSteel=contactSteel*0.1
             
        #stiffness of wood/wall
        kFloor = 0.0001
        dFloor = kFloor*0.1
        
        
        sizeOfWall=[0.002,0.002,0.002]
        rhoWood=680 #kg/m^2 680
        # passt genau für 0.229, 0.1 und dann greifen bei gripper 2 hoch
        # p0 = np.array([0.229*3/2,0.229*4+0.229/2,0]) 
        # passt genau für greifen bei gripper 2 hoch
        # p0 = np.array([0.229/2,0.229*3+0.229/2,0])
        p0 = np.array([1.2,1.075,0]) 
        
        
        # color4wall = [0.9,0.9,0.7,0.25] # grau
        color4wall = [0.0, 0.0, 1.0, 0.8]  # blau
        gWall = GraphicsDataCylinder(pAxis=[0,0,-0.1], vAxis=[0,0,0.2], radius=sizeOfCylinder, color=color4wall)
    
        pRef = p0
        rot0=RotationMatrixZ(0) @ RotationMatrixX(0)
        RBinertia = InertiaCuboid(density=rhoWood,sideLengths=sizeOfWall)
        v0=[0,0,0]    
            
    
    
    
    config.simulation.useMoveList = False
    TimeStamp=0
    adeNodes = {}
    adeNodes = df.ElementDefinition()
    
    # nx=2
    # ny=2
    # lengthInitX=0.229*nx
    # lengthInitY=0.229*ny
    # GenerateGridSimulation(nx,ny-1,adeNodes)
    
    adeNodes.ReadMeshFile('data//experiments//gripper//initial_gripper_parallel')
    adeMoveList = df.adeMoveList('data//experiments//gripper//configuration_gripper.txt')
    # adeNodes.SetNodeNumbersToAde()

    adeNodes.connectionList={}
    adeNodes.SetListOfNodes()           
    adeNodes.SetConnectionList()
    adeNodes.SetNodeNumbersToAde()
    adeNodes.SetADEPointsOfNode() 

    # nodeList=[7,8,9,15,16]
    nodeList=[12,11,13,15,16,19,20] #nodes to constrain
    
    TCP = nodeList[0]
    
    TCPcoords = []
    for nodeID in nodeList:
        if nodeID == TCP:
            continue
        coord = adeNodes.nodeDefinition[nodeID]
        TCPcoords.append([coord[0], coord[1]])
    
    # Koordinaten des Referenz TCPs    
    TCPRefCoord = np.array(adeNodes.nodeDefinition[TCP])

    # Offset aller TCPs zu Referenz TCP
    TCPOffsets = []
    for coord in TCPcoords:
        offset = np.array(coord) - TCPRefCoord
        TCPOffsets.append(offset)
    
    config.simulation.setMeshWithCoordinateConst = True

    
    
    
# if config.simulation.setMeshWithCoordinateConst:
#     config.simulation.numberOfBoundaryPoints = 0 #change here number of boundary 


oGround = mbs.AddObject(ObjectGround(referencePosition = [0,0,0]))



computeDynamic = True
# config.simulation.useMoveList = True
useInverseKinematic = True


# config.simulation.boundaryConnectionInfoInMoveList = False


#Options for Visualization
config.simulation.frameList = False
config.simulation.showJointAxes = False

config.simulation.animation = False

config.simulation.showLabeling = False #shows nodeNumbers, ID and sideNumbers, simulationTime


visualizeTrajectory = True
visualizeSurfaces = False


CPresolution = 30
LinResolution = 30


def sensorList(nodeList, adeNodes, mbs):  
    sensorTCPList=[]
    for node in nodeList:
        getADEFromNodeNumber=next(iter(adeNodes.adePointsOfNode[node]))
        adeIndex012OfNodeFromADEID=adeNodes.nodeNumbersToAdeIndex[getADEFromNodeNumber][0]
        sensorTCP = mbs.AddSensor(SensorMarker(markerNumber=mbs.GetSensor(ADE[getADEFromNodeNumber]['sensors'][adeIndex012OfNodeFromADEID])['markerNumber'], outputVariableType=exu.OutputVariableType.Position, storeInternal=True))
        sensorTCPList +=[sensorTCP]
    return sensorTCPList


# set max velocity and acceleration
# fast:
# vMax = 0.0075
# aMax = 0.005
# slow:
vMax = 0.0075
aMax = 0.0025

# store node out of nodeList to use for ikine calculation                    
ikineNodeList = [i for i in range(len(nodeList))]


def MCPsRelativeToTCP(sensorTCP):
    
    MCPcoords = []
    for s in sensorTCP[1:]:
        MCPcoords.append(np.array(mbs.GetSensorValues(s)[0:2]))
    
    # Koordinaten des TCPs    
    TCPcoords = np.array(mbs.GetSensorValues(sensorTCP[0])[0:2])

    # Offset aller MCPs zu TCP
    MCPtoTCPoffsetList = []
    for coord in MCPcoords:
        offset = np.array(coord) - TCPcoords
        MCPtoTCPoffsetList.append(offset)
    
    return MCPtoTCPoffsetList


def targetToNodeListCoords(pos, angle, sensorTCP):
    offsetList = MCPsRelativeToTCP(sensorTCP)

    # Rotationsmatrix
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    
    # Rotierte Zielkoordinaten berechnen relativ zu hauptTCP
    rotatedTargets = [pos + R @ offset for offset in offsetList]

    # neue Koordinaten relativ zu ihrer vorherigen Position
    relativeTargetOffsets = np.array(rotatedTargets) - np.array(offsetList)

    # alle zielpunkte anfuegen
    target = [pos[0], pos[1]]
    for i in relativeTargetOffsets:
        target.append(i[0])
        target.append(i[1])
        
    return target


def OpenCloseGripper(currentLen, ikObj, phi, close=True, parallel=True, d=0.2): 
    c, s = np.cos(phi), np.sin(phi)
    sign = 1 if close else -1
        
    target = [
         sign*d*c,  sign*d*s,
        -sign*d*c, -sign*d*s,
         sign*d*c,  sign*d*s,
        -sign*d*c, -sign*d*s,
    ]
    target = np.concatenate((np.zeros(6), target))

    ikObj.InverseKinematicsRelative(None, target)
    targetLen = np.array(ikObj.GetActuatorLength())
    
    return tf.PTPTrajectory(currentLen, targetLen, vMax, aMax, sync=True)
    

def calcTrajPTP(currentLen, targetPose, ikObj, sensorTCP, sync=True):
    targetPos, targetAng = targetPose
    
    if config.simulation.massPoints2DIdealRevoluteJoint2D:
        refLength = (config.lengths.L0) 
    else:
        refLength = (config.lengths.L0-2*(config.lengths.L1+config.lengths.L2))

    posVec = targetToNodeListCoords(targetPos, targetAng, sensorTCP)
    
    realPositions = np.array([mbs.GetSensorValues(s)[0:2] for s in sensorTCP]) # in simulation
    ikSensorIDs = list(range(51, 58))
    internalPositions = np.array([ikObj.mbsIK.GetSensorValues(i)[0:2] for i in ikSensorIDs]) # in ikObj
    positionDelta = (realPositions - internalPositions).flatten()
    posVec += positionDelta
    
    ikObj.InverseKinematicsRelative(None, np.array(posVec))
    # targetLen = ((np.array(ikObj.GetActuatorLength()) - refLength) * 10000)/100
    targetLen = np.array(ikObj.GetActuatorLength())
    
    # realLength = refLength + targetLen / 10000
    realLength = targetLen
    if min(realLength) < 0.2285 or max(realLength) > 0.3295:
        print(targetPos, ': liegt außerhalb des Konfigurationsraumes mit max:', max(realLength), 'und min: ', min(realLength))
    
    traj = tf.PTPTrajectory(currentLen, targetLen, vMax, aMax, sync)
    return traj


def calcTrajCP(currentLen, poseVec, ikObj, method='spline'):
    if config.simulation.massPoints2DIdealRevoluteJoint2D:
        refLength = (config.lengths.L0) 
    else:
        refLength = (config.lengths.L0-2*(config.lengths.L1+config.lengths.L2))
    
    traj = tf.CPTrajectory(poseVec, CPresolution, ikObj, ikineNodeList, TCPOffsets, refLength, method)
    return traj


class gripperStates(Enum):
    IDLE = auto()
    APPROACH_OBJECT = auto()
    POSITION_GRIPPER = auto()
    CLOSE_GRIPPER = auto()
    MOVE_OBJECT = auto()
    OPEN_GRIPPER = auto()
    RETRACT_GRIPPER = auto()
    TEST_PTP = auto()
    TEST_CP = auto()
    TEST_LIN = auto()
    
    
def calcPositions(myDict):
    posObj = myDict['posObj']
    targetPosObj = myDict['targetPosObj']
    myDict['gripperAngle'] = 0
    TCPcoords = adeNodes.nodeDefinition[TCP]
    distToObj = 0.36
    
    relPosVec = []

    alpha1 = np.arctan2(posObj[1], posObj[0])
    alpha1 = -np.pi / 16 * 7 + np.pi/2
    distObj = np.hypot(posObj[0], posObj[1])

    # Schritt 1: Annäherung
    distTCP1 = distObj - 1.65 * distToObj
    tcp1 = np.array([distTCP1 * np.cos(alpha1)-0.4, distTCP1 * np.sin(alpha1)+0.75])
    relPosVec.append((tcp1 - TCPcoords, alpha1 - np.pi / 2))

    # Schritt 2: aufs Objekt zubewegen (einfache Distanz zum Objekt)
    distTCP2 = distObj - distToObj
    tcp2 = np.array([distTCP2 * np.cos(alpha1)-0.4, distTCP2 * np.sin(alpha1)+0.75])
    relPosVec.append((tcp2 - tcp1, 0))
    
    # Schritt 3: Greifen
    
    # Schritt 4: Wegbewegen (einfache Distanz zum Objekt)
    alpha2 = np.arctan2(targetPosObj[1], targetPosObj[0])
    distTarget = np.hypot(targetPosObj[0], targetPosObj[1])
    
    distTCP3 = distTarget - distToObj
    tcp3 = np.array([distTCP3 * np.cos(alpha2), distTCP3 * np.sin(alpha2)])
    relPosVec.append((tcp3 - tcp2, alpha2 - alpha1))
    
    # Schritt 5: loslassen

    # Schritt 6: Greifer entfernen (doppelte Distanz zum Objekt)
    distTCP4 = distTarget - 1.5 * distToObj
    tcp4 = np.array([distTCP4 * np.cos(alpha2), distTCP4 * np.sin(alpha2)])
    relPosVec.append((tcp4 - tcp3, alpha2 - np.pi / 2))
    
    myDict['relPosVec'] = relPosVec
    
    return myDict
    


def functionStateMachine(t, l0, ikObj, myState, myDict):
    posObj = myDict['posObj']
    sContactForceList = myDict['sContactForceList']
    sensorTCP = myDict['sensorTCP']
    relPosVec = myDict['relPosVec']
    TCPcoords = adeNodes.nodeDefinition[TCP]
    TCPtoObj = posObj - TCPcoords
    
    # traj = calcTrajPTP(l0, relPosVec[0], ikObj, sensorTCP, sync=True)
    # traj = calcTrajPTP(l0, relPosVec[1], ikObj, sensorTCP, sync=True)
    # traj = OpenCloseGripper(l0, ikObj, myDict['gripperAngle'], close=True)
    # traj = calcTrajPTP(l0, relPosVec[2], ikObj, sensorTCP, sync=True)
    # traj = OpenCloseGripper(l0, ikObj, myDict['gripperAngle'], close=False)
    # traj = calcTrajPTP(l0, relPosVec[3], ikObj, sensorTCP, sync=True)
    # poseVec = [([0, 0], 0), targetPose]
    # traj = calcTrajCP(l0, poseVec, ikObj, method='linear')
    
    
    
    l1 = np.zeros(len(l0))
    if 'tLocal' in myDict:
        tLocal = myDict['tLocal']
    
    match myState:
        case gripperStates.IDLE:
            print('IDLE')
        case gripperStates.APPROACH_OBJECT:
            if 'trajApproach' not in myDict:
                targetPose = relPosVec[0]
                myDict['gripperAngle'] += targetPose[1]
                
                # poseVec = [
                #     ([0, 0], 0),    
                #     ([targetPose[0][0]/2, 0], targetPose[1]/2),    
                #     ([targetPose[0][0]/2, 0], targetPose[1]/2),      
                #     ([0, targetPose[0][1]], targetPose[1]),    
                # ]
                
                traj = calcTrajPTP(l0, targetPose, ikObj, sensorTCP, sync=True)
                # traj = calcTrajCP(l0, poseVec, ikObj, method='spline')
                
                myDict['trajApproach'] = traj
                tLocal = 0
                myDict['tLocal'] = tLocal
            else:
                traj = myDict['trajApproach']
            
            tTraj = traj.t_total
            l1 = traj.evaluate(tLocal)
            
            if tLocal+1 > tTraj:
                myState = gripperStates.POSITION_GRIPPER
        
        case gripperStates.POSITION_GRIPPER:
            if 'trajPositionGripper' not in myDict:
                targetPose = relPosVec[1]
                myDict['gripperAngle'] += targetPose[1]
                
                traj = calcTrajPTP(l0, targetPose, ikObj, sensorTCP, sync=True)
                
                myDict['trajPositionGripper'] = traj
                tLocal = 0
                myDict['tLocal'] = tLocal
            else:
                traj = myDict['trajPositionGripper']
            
            tTraj = traj.t_total
            l1 = traj.evaluate(tLocal)
                        
            if tLocal+1 > tTraj:       
                myState = gripperStates.CLOSE_GRIPPER
    
        case gripperStates.CLOSE_GRIPPER:
            if 'trajCloseGripper' not in myDict:
                traj = OpenCloseGripper(l0, ikObj, myDict['gripperAngle'], close=True)

                myDict['trajCloseGripper'] = traj
                tLocal = 0
                myDict['tLocal'] = tLocal
            else:
                traj = myDict['trajCloseGripper']
            
            tTraj = traj.t_total
            l1 = traj.evaluate(tLocal)
            
            # check if contact force is higher than schwellwert
            threshold = 1e-6
            
            for s in sContactForceList:
                data = mbs.GetSensorStoredData(s)
                forceNorm = np.linalg.norm(data[:, 1:4], axis=1)
                print('max kraft an Kontakt:', max(forceNorm))
                if np.any(forceNorm > threshold):
                    print('kraft auf Kontakt hat schwellwert überschritten')
                    myDict['contactThresholdTime'] = tLocal
                    myState = gripperStates.MOVE_OBJECT
                    l1 = np.zeros(len(l1))
                    break
                                        
            if tLocal+1 > tTraj: 
                myState = gripperStates.MOVE_OBJECT
    
        case gripperStates.MOVE_OBJECT:
            if 'trajMoveObject' not in myDict:
                targetPose = relPosVec[2]
                myDict['gripperAngle'] += targetPose[1]
                
                traj = calcTrajPTP(l0, targetPose, ikObj, sensorTCP, sync=True)
                # poseVec = [([0, 0], 0), (targetPose[0], 0)]
                # traj = calcTrajCP(l0, poseVec, ikObj, method='linear')
                
                
                myDict['trajMoveObject'] = traj
                tLocal = 0
                myDict['tLocal'] = tLocal
            else:
                traj = myDict['trajMoveObject']
            
            tTraj = traj.t_total
            l1 = traj.evaluate(tLocal)
            
            if tLocal+1 > tTraj:
                myState = gripperStates.OPEN_GRIPPER
    
        case gripperStates.OPEN_GRIPPER:
            if 'trajOpenGripper' not in myDict:
                traj = OpenCloseGripper(l0, ikObj, myDict['gripperAngle'], close=False)
                
                myDict['trajOpenGripper'] = traj
                tLocal = 0
                myDict['tLocal'] = tLocal
            else:
                traj = myDict['trajOpenGripper']
            
            tTraj = traj.t_total
            l1 = traj.evaluate(tLocal)
            
            if tLocal >= myDict['contactThresholdTime']:
                myState = gripperStates.RETRACT_GRIPPER
                                        
            if tLocal+1 > tTraj:       
                myState = gripperStates.RETRACT_GRIPPER
    
        case gripperStates.RETRACT_GRIPPER:
            if 'trajRetractGripper' not in myDict:
                targetPose = relPosVec[3]
                myDict['gripperAngle'] += targetPose[1]
                
                traj = calcTrajPTP(l0, targetPose, ikObj, sensorTCP, sync=True)
                # poseVec = [([0, 0], 0), targetPose]
                # traj = calcTrajCP(l0, poseVec, ikObj, method='linear')
                
                myDict['trajRetractGripper'] = traj
                tLocal = 0
                myDict['tLocal'] = tLocal
            else:
                traj = myDict['trajRetractGripper']
            
            tTraj = traj.t_total
            l1 = traj.evaluate(tLocal)
                        
            if tLocal+1 > tTraj:       
                myState = gripperStates.IDLE
                
        case gripperStates.TEST_CP:
            if 'trajCPTest' not in myDict:
                # poseVec = [
                #     ([0, 0], 0),
                #     ([0.229, 0], 0),
                #     ([0, 0.229], 0),    
                #     ([-0.229*2, 0], 0),    
                #     ([0, 0.229], 0),    
                #     ([0.229, 0], 0),    
                # ]
                poseVec = [
                    ([0, 0], 0),    
                    # ([0.01, 0], 0),    
                    # ([0.229, 0], 0),
                    # ([0, 0.165], 0),
                    # ([0.05, 0.17], -np.arctan(TCPtoObj[0]/TCPtoObj[1])),
                    ([0.4, 0.2], -np.pi/5),
                    ([0.6, -0.4], -2*np.pi/5)
                ]
                # methods: 'spline', 'linear, 'bezier2', 'bezier3', 'bezier5'
                traj = calcTrajCP(l0, poseVec, ikObj, method='bezier3')
                
                myDict['trajCPTest'] = traj
                tLocal = 0
                myDict['tLocal'] = tLocal
            else:
                traj = myDict['trajCPTest']
            
            tTraj = traj.t_total
            l1 = traj.evaluate(tLocal/CPresolution)
            
            if tLocal+1 > tTraj:
                print('test durchgeführt')
        
        case gripperStates.TEST_PTP:    
            if 'trajPTPTest' not in myDict:
                # vecRel = [posObj[0] - targetPosObj[0], posObj[1] - targetPosObj[1]]
                # targetPose = (vecRel, 0)
                targetPose = ([0.229, 0.165], -np.pi/8)
                targetPose = ([1, -0.2], -2*np.pi/5)
                traj = calcTrajPTP(l0, targetPose, ikObj, sensorTCP)
                               
                myDict['trajPTPTest'] = traj
                tLocal = 0
                myDict['tLocal'] = tLocal
            else:
                traj = myDict['trajPTPTest']
            
            tTraj = traj.t_total
            l1 = traj.evaluate(tLocal)
        
            if tLocal+1 > tTraj:
                print('test durchgeführt')
                
    myDict['tLocal'] = tLocal+1
            
    if np.all(l1==0):
        print('no new value was calculated in fsm')
        return l0, myState, myDict
    return l1, myState, myDict 



def plotContactForceNorms(mbs, sContactForceList):
    """
    Plotte die Norm der Kontaktkräfte aller Sensoren in sContactForceList.

    Args:
        mbs: Exudyn MultibodySystem
        sContactForceList: Liste von Sensor-IDs (jeweils für ein ObjectContactSphereSphere)
    """
    plt.figure(figsize=(10,6))
    
    for s in sContactForceList:
        try:
            data = mbs.GetSensorStoredData(s)  # Spalten: t, Fx, Fy, Fz
            t = data[:, 0]
            tCorrected = t.copy()
            offset = 0
            
            for i in range(1, len(t)):
                if tCorrected[i] < tCorrected[i-1]:
                    # Zeitsprung erkannt → neuen Offset berechnen
                    offset += tCorrected[i-1]
                tCorrected[i] += offset
                
            forceNorm = np.linalg.norm(data[:, 1:4], axis=1)  # Norm über Fx, Fy, Fz
            plt.plot(tCorrected, forceNorm, label=f'Sensor {s}')
        except:
            print(f"Fehler beim Zugriff auf Sensor {s}")
    
    plt.xlabel('Zeit [s]')
    plt.ylabel('Kraftnorm [N]')
    plt.title('Norm der Kontaktkräfte (ObjectContactSphereSphere)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def appendContactForcesToCSV(mbs, sContactForceList, fileName='contactForces.csv'):
    """
    Fügt Kontaktkraft-Daten aus Exudyn-Sensoren fortlaufend in eine CSV-Datei im übergeordneten Verzeichnis von exampleScripts ein.
    
    - Die Zeitwerte werden NICHT verändert oder aufaddiert.
    - Es wird automatisch ein Header erstellt, wenn die Datei neu ist.

    Args:
        mbs: Exudyn MultibodySystem
        sContactForceList: Liste von Sensor-IDs (z. B. von ObjectContactSphereSphere)
        fileName: Name der CSV-Datei
    """
    # Speicherort relativ zum Skriptverzeichnis
    baseDir = os.path.dirname(__file__)
    csvFilePath = os.path.join(baseDir, '..', fileName)

    # Öffne CSV im Anhang-Modus
    with open(csvFilePath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Schreibe Header falls Datei neu
        if os.path.getsize(csvFilePath) == 0:
            writer.writerow(['time', 'sensorID', 'Fx', 'Fy', 'Fz', 'Fnorm'])

        # Sensor-Daten einfügen
        for sID in sContactForceList:
            try:
                data = mbs.GetSensorStoredData(sID)
                for row in data:
                    t = row[0]
                    fx, fy, fz = row[1:4]
                    fnorm = np.linalg.norm([fx, fy, fz])
                    writer.writerow([t, sID, fx, fy, fz, fnorm])
            except:
                print(f"Fehler beim Zugriff auf Sensor {sID}")
                
def writeToCSV(data, fileName):
    
    # Speicherort relativ zum Skriptverzeichnis
    baseDir = os.path.dirname(__file__)
    csvFilePath = os.path.join(baseDir, '..', fileName)

    # Falls nur eine einzelne Zeile übergeben wurde → in Liste verpacken
    if isinstance(data[0], (int, float)):
        data = [data]

    numColumns = len(data[0])
    fileExists = os.path.exists(csvFilePath)
    fileIsEmpty = not fileExists or os.path.getsize(csvFilePath) == 0

    with open(csvFilePath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if fileIsEmpty:
            headers = [f"col{i}" for i in range(numColumns)]
            writer.writerow(headers)

        for row in data:
            if len(row) != numColumns:
                raise ValueError("Alle Zeilen müssen die gleiche Anzahl an Spalten haben")
            writer.writerow(row)
    
    


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
            
            if contactON:
                #+++++++++++++++++++++++++++++++++++++++ contact
                # p0 = np.array([0,0,0])
                # color4wall = [0.9,0.9,0.7,0.25]
                # gFloor = GraphicsDataOrthoCubePoint(p0,[1.,0.01,0.02],color4steelblue)
    
                # [meshPoints, meshTrigs] = GraphicsData2PointsAndTrigs(gFloor)
                
                # gDataList = [gFloor]                
                
                # oGround=mbs.AddObject(ObjectGround(referencePosition= [0.5,-0.007,0],
                #                                     visualization=VObjectGround(graphicsData=gDataList)))
                # mGround = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround))
                                
                # gContact.verboseMode = 1                
                # gContact.AddTrianglesRigidBodyBased(rigidBodyMarkerIndex=mGround, contactStiffness=kFloor, contactDamping=dFloor, frictionMaterialIndex=frictionMaterialIndex,
                #     pointList=meshPoints,  triangleList=meshTrigs)
                #+++++++++++++++++++++++++++++++++++++++ contact
            

            
                [nMass, oMass] = AddRigidBody(mainSys=mbs, inertia=RBinertia, 
                                      position=pRef, velocity=v0,
                                      rotationMatrix=rot0,
                                      gravity=[0.,0.,0.],
                                      graphicsDataList=[gWall],
                                      )
    
                [meshPointsCube, meshTrigsCube] = GraphicsData2PointsAndTrigs(gWall)
                
                mCube = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oMass, localPosition=[0,0,0]))
    
                # gContact.AddTrianglesRigidBodyBased(rigidBodyMarkerIndex=mCube, contactStiffness=kFloor, contactDamping=dFloor, frictionMaterialIndex=frictionMaterialIndex,
                                                                # pointList=meshPointsCube,  triangleList=meshTrigsCube)
                
                oGround=mbs.AddObject(ObjectGround(referencePosition= p0))
                mGround = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround))
                mZylinder = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oMass, localPosition=[0, 0, 0]))
                
                mbs.AddObject(GenericJoint(markerNumbers=[mGround, mCube], 
                                            constrainedAxes=[0,0,1,0,0,0],
                                            visualization=VObjectJointGeneric(axesRadius=0.5e-4, axesLength=0.5e-3)))
                
                
                mbs.AddSensor(SensorMarker(markerNumber=mCube, fileName='solution/sensorTest.txt', 
                                            outputVariableType=exu.OutputVariableType.Position))
            
                # connectionEntry = [12,3,13,2,8,3,9,2]
                connectionEntry = [12,3,13,2,16,3,17,2]
                
                # addContactSpheres(gContact,ADE,connectionEntry,sizeOfSpheres,contactSteel,dSteel,frictionMaterialIndex)                


                sContactForceList = []
                
                j = 0
                for i in connectionEntry[::2]:
                    # greiferseitige Marker (2 Marker pro Seite)
                    markerPair = ADE[i].get('cATC').markersForConnectorsATC[(connectionEntry[1::2][j] - 1) * 2 : (connectionEntry[1::2][j] - 1) * 2 + 2]
                    j += 1
                
                    for mGrip in markerPair:
                        # Dataknoten für Kontaktmodell
                        nData = mbs.AddNode(NodeGenericData(initialCoordinates=[0, 0, 0, 0], numberOfDataCoordinates=4))
                
                        # Kontaktobjekt
                        oSSC = mbs.AddObject(ObjectContactSphereSphere(
                            markerNumbers=[mGrip, mZylinder],
                            nodeNumber=nData,
                            spheresRadii=[sizeOfSpheres, sizeOfCylinder],
                            contactStiffness=contactSteel,
                            contactDamping=dSteel,
                            impactModel=0,
                            visualization=VObjectContactSphereSphere(show=True)
                        ))
                
                        # Sensor zur Kraftauslese
                        sContactForce = mbs.AddSensor(SensorObject(
                            objectNumber=oSSC,
                            outputVariableType=exu.OutputVariableType.Force,
                            storeInternal=True,
                            writeToFile=False
                        ))
                        sContactForceList.append(sContactForce)
                            
            
            
            
            ##+++++++++++++++++++++++++++++++++++++++   
            # forceVal=400
            # mbs.SetLoadParameter(mbs.variables['VertexLoads'][-2],'loadVector',[0,-forceVal,0])
            ##+++++++++++++++++++++++++++++++++++++++
            
            
            
            ##+++++++++++++++++++++++++++++++++++++++
            ## Simulation Settings
            simulationSettings = exu.SimulationSettings() #takes currently set values or default values
            # simulationSettings.timeIntegration.preStepPyExecute = 'MBSUserFunction()'
            T=0.002
            SC.visualizationSettings.connectors.defaultSize = T
            
            SC.visualizationSettings.connectors.showJointAxes=True
            SC.visualizationSettings.connectors.jointAxesLength=0.02
            SC.visualizationSettings.connectors.jointAxesRadius=0.005
            
            SC.visualizationSettings.bodies.defaultSize = [T, T, T]
            SC.visualizationSettings.nodes.defaultSize = 0.0025
            SC.visualizationSettings.markers.defaultSize = 0.005
            SC.visualizationSettings.loads.defaultSize = 0.015
            SC.visualizationSettings.general.autoFitScene=True
            
            SC.visualizationSettings.nodes.show = False
            SC.visualizationSettings.markers.show = False
            SC.visualizationSettings.connectors.show= False
            SC.visualizationSettings.sensors.show = False
            
            SC.visualizationSettings.openGL.lineWidth=2 #maximum
            SC.visualizationSettings.openGL.lineSmooth=True
            SC.visualizationSettings.openGL.multiSampling = 4
            SC.visualizationSettings.general.drawCoordinateSystem = False
            #SC.visualizationSettings.window.renderWindowSize=[1600,1024]
            SC.visualizationSettings.window.renderWindowSize=[1600,1000]

            SC.visualizationSettings.contact.showSearchTree = False
            SC.visualizationSettings.contact.showSearchTreeCells = False
            SC.visualizationSettings.contact.showBoundingBoxes = False
            SC.visualizationSettings.contact.showSpheres = True
            
            
            
            SC.visualizationSettings.nodes.showNumbers = False
            
            if visualizeTrajectory:
                SC.visualizationSettings.sensors.traces.showPositionTrace = False
            
            # simulationSettings.displayComputationTime= True
            ##++++++++++++++++++++++++++++++
            ##ANIMATIONS
            ##make images for animations (requires FFMPEG):
            ##requires a subfolder 'images'
            if config.simulation.animation:
                simulationSettings.solutionSettings.recordImagesInterval=0.04 #simulation.endTime/200
            
            
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
                
            
            
            
            
            ##+++++++++++++++++++++++++++++++++++++++
            ##just visualization stuff
            mbsV = SC.AddSystem()
            mbsV.variables['mbs']=mbs
            mbs.variables['mbsV']=mbsV
            
            if visualizeSurfaces:
                def UFgraphics2(mbs, objectNum):
                    mbsV = mbs.variables['mbsV']
                    mbsV.SendRedrawSignal()
                    return []
                                
                ground2 = mbs.AddObject(ObjectGround(visualization=VObjectGround(graphicsDataUserFunction=UFgraphics2)))                 
            ##+++++++++++++++++++++++++++++++++++++++
                        
            
            
            ## Assemble Mesh 
            mbs.Assemble()
            sysStateList = mbs.systemData.GetSystemState()

            
            if displaySimulation:
                SC.renderer.Start()
                SC.renderer.DoIdleTasks()


            # # connect and disconnect ATCs
            # if type(adeNodes) == dict:
            #     for ID in adeNodes[0].elements:
            #         for i in range(len(adeNodes[0].connectionList[ID])):
            #             if adeNodes[0].connectionList[ID][i][2] in ADE:
            #                 ConnectADE(mbs,ADE,adeNodes[0].connectionList[ID][i])
            # else:
            #     for ID in adeNodes.elements:
            #         for i in range(len(adeNodes.connectionList[ID])):
            #             if adeNodes.connectionList[ID][i][2] in ADE:
            #                 ConnectADE(mbs,ADE,adeNodes.connectionList[ID][i])
            #########################################################################   
            #########################################################################  
            

            mbs.AssembleLTGLists()
            mbs.systemData.SetSystemState(sysStateList,configuration = exu.ConfigurationType.Initial)

            mbs.Assemble()
            ##+++++++++++++++++++++++++++++++++++++++
            ## Setting um SensorValues and Sending initial Sensor Values (Mesh Node Coordinates)
            SensorValuesTime = {} 
            SensorValuesTime[0] = {}  
            SensorValues = {} 
            SensorValues[0] = {}
            for ID in adeIDs:
                SensorValues[0][ID] = np.asmatrix([mbs.GetSensorValues(ADE[ID]['sensors'][0])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][1])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][2])[0:2]])
            # client.Put('SensorValues',pickle.dumps(SensorValues))
            SensorValuesTime[0]=SensorValues.copy()
            ##+++++++++++++++++++++++++++++++++++++++



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
                    gSurface+=[GraphicsDataFromPointsAndTrigs(points,triangles,color=[0.41,0.41,0.41,0.0])]

                return gSurface
            
            if visualizeTrajectory:
                ground = mbsV.AddObject(ObjectGround(visualization=VObjectGround(graphicsDataUserFunction=UFgraphics)))

            mbsV.Assemble()

            

            newResidual = True
            if not(newResidual):
                simulationSettings.staticSolver.newton.relativeTolerance = 1e-10
                simulationSettings.staticSolver.newton.absoluteTolerance = 1e-12
            else:
                simulationSettings.staticSolver.newton.relativeTolerance = 1e-5
                simulationSettings.staticSolver.newton.absoluteTolerance = 1e-5
                simulationSettings.staticSolver.newton.newtonResidualMode = 1 #take Newton increment
            
            # simulationSettings.staticSolver.stabilizerODE2term = 1
            
            simulationSettings.staticSolver.newton.numericalDifferentiation.relativeEpsilon = 1e-9
            simulationSettings.staticSolver.newton.maxIterations = 250
            

            simulationSettings.linearSolverType = exu.LinearSolverType.EigenSparse
            # simulationSettings.linearSolverSettings.ignoreSingularJacobian=True
            
            simulationSettings.timeIntegration.newton.useModifiedNewton = False #for faster simulation
            
            
            simulationSettings.staticSolver.newton.weightTolerancePerCoordinate = True   
            simulationSettings.solutionSettings.solutionInformation = "PARTS_StaticSolution"
            simulationSettings.timeIntegration.generalizedAlpha.computeInitialAccelerations = False
            
            numberOfLoadSteps=10
            simulationSettings.staticSolver.numberOfLoadSteps = numberOfLoadSteps
            
            # print(mbs.ComputeSystemDegreeOfFreedom() )
            # exu.SolveStatic(mbs, simulationSettings=simulationSettings,showHints=True)
            exu.SolveDynamic(mbs, simulationSettings=simulationSettings,showHints=True)
            
            # print('numericalSolution:',mbs.GetSensorValues(nodeList[0]) )
            
            solODE2initial = mbs.systemData.GetODE2Coordinates(configuration=exu.ConfigurationType.Current)
            mbs.systemData.SetODE2Coordinates(solODE2initial, configuration=exu.ConfigurationType.Initial)  #set old solution as initial value for new solution
             
            # SC.renderer.DoIdleTasks()
            ##############################################################          
            
            # mbs.SetPreStepUserFunction(PreStepUserFunction)

            ##############################################################
            simulationSettings.timeIntegration.numberOfSteps = int(config.simulation.nrSteps)
            simulationSettings.timeIntegration.endTime = config.simulation.endTime
            
            
            timeTotal=timeTotal #+mbs.sys['staticSolver'].timer.total
            numberOfSteps=numberOfSteps+1

            # simulationSettings.linearSolverType = exu.LinearSolverType.EigenDense
            # simulationSettings.linearSolverSettings.ignoreSingularJacobian=True
            
            
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
                    config.simulation.endTime = endTime
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
                            numberOfLoadSteps=100
                            simulationSettings.staticSolver.numberOfLoadSteps = numberOfLoadSteps
                            
                            exu.SolveStatic(mbs, simulationSettings = simulationSettings,showHints=True)
                            
                            timeTotal=timeTotal#+mbs.sys['staticSolver'].timer.total
                            numberOfSteps=numberOfSteps+1

                        else:
                            exu.SolveDynamic(mbs, simulationSettings = simulationSettings,showHints=True)
                            
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

                        SensorValuesTime[cnt+1][0]=SensorValues[cnt+1]

                    # if not mbs.GetRenderEngineStopFlag():
                        sysStateList = mbs.systemData.GetSystemState()
                        mbs.systemData.SetSystemState(sysStateList,configuration = exu.ConfigurationType.Initial)

                    # else:
                        
                        # ##+++++++++++++++++++++++++++++++++++++++
                        # ## Resetting Simulation
                        # break


            if not useInverseKinematic:
                if displaySimulation and not mbs.GetRenderEngineStopFlag():  
                    SC.renderer.DoIdleTasks()   
                    exu.StopRenderer() #safely close rendering window!
                try:
                    if config.simulation.solutionViewer:
                          sol = LoadSolutionFile(config.simulation.solutionViewerFile, safeMode=True)#, maxRows=100) 
                          SolutionViewer(mbs,sol) #can also be entered in 
                except:
                    logging.warning('Couldnt load Solution File')                
                    
            
            
            
            
            ##START InverseKinematic ++++++++++++++++
            ##+++++++++++++++++++++++++++++++++++++++
            ##+++++++++++++++++++++++++++++++++++++++
            ##+++++++++++++++++++++++++++++++++++++++
            if useInverseKinematic:
                if config.simulation.useMoveList==False:
                    cnt=-1
    
                
                
                config.simulation.useInverseKinematic=True
                
                
                if not config.simulation.useInverseKinematic:
                    if displaySimulation and not mbs.GetRenderEngineStopFlag():  
                        SC.renderer.DoIdleTasks()      
                        exu.StopRenderer() #safely close rendering window!
                    try:
                        if config.simulation.solutionViewer:
                              sol = LoadSolutionFile(config.simulation.solutionViewerFile, safeMode=True)#, maxRows=100) 
                              SolutionViewer(mbs,sol) #can also be entered in 
                    except:
                        logging.warning('Couldnt load Solution File')
                        
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


                    ##+++++++++++++++++++++++++++++++++++++++
                    ##just visualization stuff
                    mbsV2 = SC.AddSystem()
                    mbsV2.variables['mbs']=mbs2
                    mbs2.variables['mbsV']=mbsV2
                    
                    # if visualizeSurfaces:
                    #     def UFgraphics2(mbs, objectNum):
                    #         mbsV2 = mbs2.variables['mbsV']
                    #         mbsV2.SendRedrawSignal()
                    #         return []
                                        
                    #     ground2 = mbs2.AddObject(ObjectGround(visualization=VObjectGround(graphicsDataUserFunction=UFgraphics2)))                 
                    # ##+++++++++++++++++++++++++++++++++++++++
                    # mbsV2.Assemble() 
    
                    ikObj, mbs2, sensorMbs2TCP = createKinematicModel(mbs, mbs2)
    



                    if False:
                        # test circle
                        r = 0.001
                        if simulateTowerADEs:
                            r=0.001
                        if simulateRabbit:
                            r=0.2
                        if simulateGripper:
                            r=0.15
                            phi = np.linspace(0, np.pi, 50)
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
        
                        
        
                        if simulateGripper:
                            dx2={}
                            for k in range(len(dxx)):
                                dx2[k]=[]
                                
                                for i in range(len(nodeList)):
                                    if simulateGripper:
                                        if i ==1:
                                            dxx[k][0]=-dxx[k][0]
                                    else:
                                        dxx[k][0]=dxx[k][0]
                                        
                                    dx2[k]+=list(dxx[k])#+list(dxx[k]) #+list(dxx[k]) #<- change value here to go
                                
    
                            r=0.5
                            phi = np.linspace(np.pi/2, np.pi, 50)
                            dxx=[]
                            for j in range(len(phi)-1): 
                                xOld = r* np.array([-cos(phi[j]), np.sin(phi[j])])
                                xNew = r* np.array([-cos(phi[j+1]), np.sin(phi[j+1])])
                                dxx += [xNew - xOld]     
    
    
                            for j in range(len(dxx)):
                                dx2[k+j+1]=[]
                                
                                for i in range(len(nodeList)):
                                    if i ==1:
                                        dxx[j][0]=dxx[j][0]
                                    else:
                                        dxx[j][0]=dxx[j][0]
                                        
                                    dx2[k+j+1]+=list(dxx[j])#+list(dxx[k]) #+list(dxx[k]) #<- change value here to go                         
                                
                             
                            
                         
                        # xVar=np.linspace(0,0.5,5)
                        # for i in range(len(xVar)-1):
                        #     dx2[k+i]=[]
                        #     xOld = np.array([xVar[i],xVar[i]*0])
                        #     xNew = np.array([xVar[i+1],xVar[i+1]*0])
                        #     dxx += [xNew - xOld]  
                                
                        #     for j in range(len(nodeList)):
                        #         # dxx[k+i][0]+=dxx[k+i][0]
                                
                        #         dx2[k+i]+=list(dxx[k+i])#+list(dxx[k]) #+list(dxx[k]) #<- change value here to go
                            
                            
                    # #user function for moving graphics:
                    # def UFgraphics(mbsV, objectNum):
                    #     gLine=[]
    
                    #     mbs = mbsV.variables['mbs']
                        
                    #     for sens in range(len(sensorMbs2TCP)):
                    #         A=mbs.GetSensorValues(sensorMbs2TCP[sens],configuration=exu.ConfigurationType.Visualization)
                            
                    #         for k in range(len(dxx)):
                    #             if simulateGripper:
                    #                 # if sens==0:
                    #                 dxx[k][0]=-dxx[k][0]
                                        
                    #             gLine+=[GraphicsDataLine([A,A+np.array(list(dxx[k])+[0])], color=color4blue)]
                    #             A=A+np.array(list(dxx[k])+[0])
                    
                    #     return gLine
                    
                    # if visualizeTrajectory:
                    #     ground = mbsV.AddObject(ObjectGround(visualization=VObjectGround(graphicsDataUserFunction=UFgraphics)))
                    
                     
                        
                    # mbsV.Assemble()
                        
                    
                    
                    
                    
                    
                    # # test straight line
                    # dxx=[]
                    # xVar=np.linspace(0,0.01,5)
                    # for i in range(len(xVar)-1):
                    #         xOld = np.array([xVar[i],xVar[i]])
                    #         xNew = np.array([xVar[i+1],xVar[i+1]])
                    #         dxx += [xNew - xOld]  
    
    
                    # test horizontal-line
                    # dxx=[]
                    # xVar=np.linspace(0,0.5,5)
                    # for i in range(len(xVar)-1):
                    #         xOld = np.array([xVar[i],0])
                    #         xNew = np.array([xVar[i+1],0])
                    #         dxx += [xNew - xOld]  
                    
                    
                    # # dxx=[]
                    # xVar=np.linspace(0,0.1,5)
                    # for i in range(len(xVar)-1):
                    #         xOld = np.array([0,xVar[i]])
                    #         xNew = np.array([0,xVar[i+1]])
                    #         dxx += [xNew - xOld]   
    
    
                    # xVar=np.linspace(0,0.5,5)
                    # for i in range(len(xVar)-1):
                    #         xOld = np.array([-xVar[i],0])
                    #         xNew = np.array([-xVar[i+1],0])
                    #         dxx += [xNew - xOld]  
                            
                    
                    # xVar=np.linspace(0,0.2,5)
                    # for i in range(len(xVar)-1):
                    #         xOld = np.array([0,-xVar[i]])
                    #         xNew = np.array([0,-xVar[i+1]])
                    #         dxx += [xNew - xOld]                         
    
    
                    # xVar=np.linspace(0,0.5,5)
                    # for i in range(len(xVar)-1):
                    #         xOld = np.array([-xVar[i],0])
                    #         xNew = np.array([-xVar[i+1],0])
                    #         dxx += [xNew - xOld]                          
                            
                    
                    # test with two points
                    # dxx=[np.array([0.08,0]),np.array([0.08,0])]
                    # dxx=[np.array([0.,0.2*0])]
                    
                    
                    cnt=0
                    actorList = []
                    sortedActorStrokesList=[] 
                     
                    setConfigFileKinModel()  # get ActuatorLength from KinModel 
                    l1 = ikObj.GetActuatorLength()
                    l0 = l1
                    
     
                    if config.simulation.massPoints2DIdealRevoluteJoint2D:
                        refLength = (config.lengths.L0) 
                    else:
                        refLength = (config.lengths.L0-2*(config.lengths.L1+config.lengths.L2))
                    
                    
                    # map actuator length to SurrogateModel
                    actorList = [((np.array(l1)-refLength)*10000).tolist()]
                    # actorList = [(np.round(l1, 5)).tolist()]
                    sortedActorStrokesList += actorList      
                    
                    # mbs.SetPreStepUserFunction(PreStepUserFunction)
                    
                        
                    # for k in range(len(dxx)):
                    #     dx=[]
                    
                    #     for i in range(len(nodeList)):
                    #         if simulateGripper:
                    #             if i ==1:
                    #                 dxx[k][0]=-dxx[k][0]
                    #         else:
                    #             dxx[k][0]=dxx[k][0]
                                
                    #         dx+=list(dxx[k])#+list(dxx[k]) #+list(dxx[k]) #<- change value here to go
                    
                    t = 0
                    tTotal = 100
                    
                    mbs.variables['state'] = gripperStates.APPROACH_OBJECT
                    mbs.variables['myDict'] = {
                        'posObj': [p0[0], p0[1]],
                        # 'targetPosObj': [p0[0], p0[1]],
                        'targetPosObj': [0.229/2,0.229*7],
                        'TCPcoords': adeNodes.nodeDefinition[TCP],
                        'sContactForceList': sContactForceList,
                        'sensorTCP': sensorTCP,
                    }
                    
                    mbs.variables['myDict'] = calcPositions(mbs.variables['myDict'])
    
        
                    for i in range(tTotal):
                        print('Time:', i)
                        setConfigFileKinModel()
                        
                        def UFgraphicsIK(mbsV2, objectNum):
                            actuatorSideNumber=[]
                            for oSpringDamper in mbs2.variables['springDamperDriveObjects']: 
                                
                                stiffness = mbs2.GetObjectParameter(oSpringDamper,'stiffness')
                                
                                
                                markerNumbers = mbs2.GetObjectParameter(oSpringDamper,'markerNumbers')                                  
                                positionActor1 = mbs2.GetMarkerOutput(markerNumbers[0],exu.OutputVariableType.Position, configuration = exu.ConfigurationType.Visualization)
                                positionActor2 = mbs2.GetMarkerOutput(markerNumbers[1],exu.OutputVariableType.Position, configuration = exu.ConfigurationType.Visualization)
                                
                                positionActor = positionActor1+(positionActor2-positionActor1)/2
                                actuatorSideNumber += [GraphicsDataText(point= positionActor, 
                                                                      text= str(round(stiffness,2)), 
                                                                      color= [0.,0.,0.,1.]) ]
                            return actuatorSideNumber
                        
                        # ground = mbsV2.AddObject(ObjectGround(visualization=VObjectGround(graphicsDataUserFunction=UFgraphicsIK)))
                        mbsV2.Assemble() 
                        
                        
                        # plotContactForceNorms(mbs, sContactForceList)  # alle Richtungen
                        appendContactForcesToCSV(mbs, sContactForceList, 'force_log.csv')
        
                        l1, mbs.variables['state'], mbs.variables['myDict'] = functionStateMachine(t, l0, ikObj, mbs.variables['state'], mbs.variables['myDict'])

                                                    
                        if config.simulation.massPoints2DIdealRevoluteJoint2D:
                            refLength = (config.lengths.L0) 
                        else:
                            refLength = (config.lengths.L0-2*(config.lengths.L1+config.lengths.L2))
    
                        # map actuator length to SurrogateModel
                        actorList = [((np.array(l1)-refLength)*10000).tolist()]
                        # actorList = [(np.round(l1, 5)).tolist()]
                        sortedActorStrokesList += actorList 
                        
                        for i in range(1):
                            # endTime = 0.83
                            actorTime = 0.83
                            # nrSteps = 5
                            mbs.variables['line']=cnt+1
                            SensorValuesTime[cnt+1]={}                    
                            
                            nextTimeStamp = False   
        
                            mbs.variables['cnt'] = cnt
                            if computeDynamic:
                                
                                # setConfigFileKinModel()    
                                setConfigFileSurrogateModel()
                                
                                simulationSettings.timeIntegration.numberOfSteps = int(config.simulation.nrSteps)
                                simulationSettings.timeIntegration.endTime = config.simulation.endTime
                        
                                simulationSettings.displayComputationTime = False  
                                simulationSettings.timeIntegration.verboseMode = 0
                                simulationSettings.displayStatistics = False       
                                
                                
                                #######################################
                                #######################################
                                simulationSettings.timeIntegration.newton.absoluteTolerance = 1e3
                                simulationSettings.timeIntegration.newton.numericalDifferentiation.minimumCoordinateSize = 1
                            
                                simulationSettings.timeIntegration.verboseMode = 0
                                #simulationSettings.timeIntegration.newton.useNumericalDifferentiation = True
                                #simulationSettings.timeIntegration.newton.numericalDifferentiation.doSystemWideDifferentiation = True
                                simulationSettings.timeIntegration.generalizedAlpha.spectralRadius = 0.4
                                simulationSettings.timeIntegration.adaptiveStep = True #disable adaptive step reduction   
                                #######################################
                                #######################################
                                
                                mbs.SetPreStepUserFunction(PreStepUserFunction)
                                
    
                                simulationSettings.staticSolver.verboseMode=0
    
                                # exu.SolveStatic(mbs, simulationSettings,showHints=False) 
                                exu.SolveDynamic(mbs, simulationSettings,showHints=False) 
                                
                                timeTotal=timeTotal#+mbs.sys['staticSolver'].timer.total
                                numberOfSteps=numberOfSteps+1                            
                                
                                
                                ## Sending new Sensor Values to Webserver
                                SensorValuesTime[cnt+1][0]={}
                                SensorValues[cnt+1] = {}
                                for ID in adeIDs:
                                    SensorValues[cnt+1][ID] = np.asmatrix([mbs.GetSensorValues(ADE[ID]['sensors'][0])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][1])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][2])[0:2]])
                                # client.Put('SensorValues',pickle.dumps(SensorValues))                           
                                SensorValuesTime[cnt+1][0]=SensorValues[cnt+1]    
                                
                                for plotSensorMbs2TCP in sensorMbs2TCP:
                                    plt.scatter( mbs2.GetSensorValues(plotSensorMbs2TCP)[0], mbs2.GetSensorValues(plotSensorMbs2TCP)[1], marker='d', color='blue' )
                                    
                                for plotSensorTCP in sensorTCP:
                                    plt.scatter( mbs.GetSensorValues(plotSensorTCP)[0], mbs.GetSensorValues(plotSensorTCP)[1] , marker='x', color='blue')
                                    
                                    # sensorValueError=np.array([mbs.GetSensorValues(sensorTCP)[0],mbs.GetSensorValues(sensorTCP)[1]])
                                    # gapWithoutCorr= np.linalg.norm(sensorValueError-iWantToGoToThisValue)
                                    # print('gap without correction:',gapWithoutCorr*1000,'mm')                            
                                
                                
                                
                                
    
    
                    
    
                                ## Sending new Sensor Values to Webserver
                                SensorValuesTime[cnt+1][0]={}
                                SensorValues[cnt+1] = {}
                                for ID in adeIDs:
                                    SensorValues[cnt+1][ID] = np.asmatrix([mbs.GetSensorValues(ADE[ID]['sensors'][0])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][1])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][2])[0:2]])
                                # client.Put('SensorValues',pickle.dumps(SensorValues))                           
                                SensorValuesTime[cnt+1][0]=SensorValues[cnt+1]    
                                
                                for plotSensorMbs2TCP in sensorMbs2TCP:
                                    plt.scatter( mbs2.GetSensorValues(plotSensorMbs2TCP)[0], mbs2.GetSensorValues(plotSensorMbs2TCP)[1], marker='d', color='blue' )
                                    
                                for plotSensorTCP in sensorTCP:
                                    plt.scatter( mbs.GetSensorValues(plotSensorTCP)[0], mbs.GetSensorValues(plotSensorTCP)[1] , marker='x', color='blue')
                                    
                                    
                                    
                                cnt=cnt+1
    
                                sysStateList = mbs.systemData.GetSystemState()
                                mbs.systemData.SetSystemState(sysStateList,configuration = exu.ConfigurationType.Initial)                        
                            
                            if not mbs.GetRenderEngineStopFlag():
                                sysStateList = mbs.systemData.GetSystemState()
                                mbs.systemData.SetSystemState(sysStateList,configuration = exu.ConfigurationType.Initial)
                                
                        # plt.scatter( mbs2.GetSensorValues(sensorMbs2TCP)[0], mbs2.GetSensorValues(sensorMbs2TCP)[1], marker='o' )
                        # plt.scatter( mbs.GetSensorValues(sensorTCP)[0], mbs.GetSensorValues(sensorTCP)[1] , marker='x')
                        # averageSensorValues=AverageMeshPoints(SensorValues[cnt-1],adeNodes,draw = True)
                        
                        
                        l0 = l1
                        t = t+1
                    
                        
                    if displaySimulation:  
                        print('simulation finished')
                        SC.renderer.DoIdleTasks()   
                        exu.StopRenderer() #safely close rendering window!                
                        
    
                    else:
                        if computeDynamic:
        
                                simulationSettings.solutionSettings.solutionInformation = "Exudyn ADE Simulation"
        
                                mbs.Assemble()
                                
                                if displaySimulation:
                                    SC.renderer.Start()
                                    SC.renderer.DoIdleTasks()
        
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
        
            ## END InverseKinematic +++++++++++++++++
            ##+++++++++++++++++++++++++++++++++++++++
            ##+++++++++++++++++++++++++++++++++++++++
            ##+++++++++++++++++++++++++++++++++++++++
            ##+++++++++++++++++++++++++++++++++++++++
        
        
        
        
        
        
            # if displaySimulation and not mbs.GetRenderEngineStopFlag():  
            SC.renderer.DoIdleTasks()    
            exu.StopRenderer() #safely close rendering window!
            
            
            try:
                if config.simulation.solutionViewer:
                       sol = LoadSolutionFile(config.simulation.solutionViewerFile, safeMode=True)#, maxRows=100) 
                       SolutionViewer(mbs,sol) #can also be entered in 
                      
                      
                      # sol2 = LoadSolutionFile(config.simulation.solutionViewerFileIKIN, safeMode=True)#, maxRows=100) 
                      # SolutionViewer(mbs2,sol2) #can also be entered in 
            except:
                logging.warning('Couldnt load Solution File')


            ##+++++++++++++++++++++++++++++++++++++++
            ## Saving Sensor Data to File
            if config.simulation.saveSensorValues:
                  pickle.dump(  SensorValues,  open(config.simulation.sensorValuesFile,  'wb')  )
                  pickle.dump(  SensorValuesTime,  open(config.simulation.sensorValuesTimeFile,  'wb')  )
                 
            ##+++++++++++++++++++++++++++++++++++++++
            ## Resetting Simulation
            mbs.Reset()
            startSimulation = False
            break
    
    except Exception as ExeptionMessage: 
        logging.warning('Error ocured during Simulation')
        logging.warning(ExeptionMessage)
        startSimulation = True
        exu.StopRenderer()
        #client.Put('StartSimulation',json.dumps(False))
        mbs.Reset()
    time.sleep(config.simulation.looptime-time.time()%config.simulation.looptime)

        
# print('time: ', timeTotal, 'numberOfSteps: ',numberOfSteps)

# try:
#     client.Put('EndProgram',json.dumps(True))
# except:
#     os._exit(0)

# print('userFunction: ',mbs.variables['counterForTesting'])

# print('time: ', , 'numberOfSteps: ',numberOfSteps)
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


from exudyn.plot import PlotSensor
PlotSensor(mbs, sensorNumbers=['solution/sensorTest.txt'], components=[-1])








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
        
 
