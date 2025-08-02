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
    
script_dir = os.path.dirname(__file__)
programm_dir = os.path.dirname(script_dir)

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
simulateGripper=False 
simulateTowerADEs=False
simulateRabbit=False
simulateDuckBridgeCraneTower=False
simulateReconfigurationGerbl=False
simulateGripperArm=True
##++++++++++++++++++++++++++++++
##ANIMATIONS
##make images for animations (requires FFMPEG):
my_dir = os.path.abspath('./functions')
# os.system('start /MIN cmd /C  python "'+os.path.join(my_dir, 'webserver_functions.py')+'"')    
# os.system('start /MIN cmd /C python "'+os.path.join(my_dir, 'measurement_functions.py')+'"')

contactON = True


if simulateGripper or simulateGripperArm:
    if contactON:
        ##++++++++++++++++++++++++++++++
        gContact = mbs.AddGeneralContact()
        gContact.verboseMode = 1
        mu = 0.2*0      #dry friction
    
        gContact.frictionProportionalZone = 1e-3
        # gContact.excludeDuplicatedTrigSphereContactPoints = False
        fricMat = mu*np.eye(1)
        
        
        gContact.SetFrictionPairings(fricMat)
        gContact.SetSearchTreeBox(pMin=[-0.5,0,-0.2],pMax=[0.5,3,0.3])
        gContact.SetSearchTreeCellSize(numberOfCells=[5,10,1]) 
        
        sizeOfCylinder=0.05
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
        p0 = np.array([0.229/2,0.229*9+0.229/2,0])
        
        color4wall = [0.9,0.9,0.7,0.25]
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
    
    adeNodes.ReadMeshFile('data//experiments//gripper//initial9_gripper')
    adeMoveList = df.adeMoveList('data//experiments//gripper//configuration_gripper.txt')
    # adeNodes.SetNodeNumbersToAde()

    adeNodes.connectionList={}
    adeNodes.SetListOfNodes()           
    adeNodes.SetConnectionList()
    adeNodes.SetNodeNumbersToAde()
    adeNodes.SetADEPointsOfNode() 

    nodeList=[15,16] #nodes to constrain
    
    
    config.simulation.setMeshWithCoordinateConst = True
    
if simulateGripperArm:
    nx = 1
    ny = 4
    gripperFilePath = os.path.join(programm_dir, "data", "experiments", "gripper", "initial9_gripper")
    adeMoveList = df.adeMoveList('data//experiments//gripper//configuration_gripper.txt')
    
    adeNodes = df.ElementDefinition()
    
    lengthInitX = 0.229*nx
    lengthInitY = 0.229*ny
    GenerateGridSimulation(nx, ny, adeNodes)
    
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
    meshPath = os.path.join(programm_dir, "data", "experiments", "gripper", "initial9_gripper")
    tempGripper.ReadMeshFile(meshPath)
    
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
    nodeList.append(topNode2 + 1)
    nodeList.append(topNode2 + 3)
    # nodeList = [topNode2 + 1, topNode2 + 2, topNode2 + 3] # 13, 14, 15 also die drei in einer linie vom TCP 14
    print('TCP Node ID:', nodeList[0])
    TCPID = nodeList[0]
    
    
    # sort all nodes for y coordinates to find two highest points on the gripper
    nodeYList = [(nodeID, coord[1]) for nodeID, coord in adeNodes.nodeDefinition.items()]
    nodeYList.sort(key=lambda x: x[1], reverse=True)
    topNode1, topNode2 = nodeYList[0][0], nodeYList[1][0] # zwei Spitzen des Greifers


    nodeList.append(topNode1)
    nodeList.append(topNode2)
    
    # print('GripperTopNode1:', topNode1, 'GripperTopNode2:', topNode2)
    
    # calculate offset of TCPs to TCPID
    TCPcoords = []
    for nodeID in nodeList:
        if nodeID == TCPID:
            continue
        coord = adeNodes.nodeDefinition[nodeID]
        TCPcoords.append([coord[0], coord[1]])
        
    # Koordinaten des Referenz TCPs    
    TCPRefCoord = np.array(adeNodes.nodeDefinition[TCPID])

    # Offset aller TCPs zu Referenz TCP
    TCPOffsets = []
    for coord in TCPcoords:
        offset = np.array(coord) - TCPRefCoord
        TCPOffsets.append(offset)
        
        
    
 
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

config.simulation.showLabeling = True #shows nodeNumbers, ID and sideNumbers, simulationTime


visualizeTrajectory = False
visualizeSurfaces = False

# set max velocity and acceleration
vMax = 20
aMax = 10

# store node out of nodeList to use for ikine calculation                    
ikineNodeList = [i for i in range(len(nodeList))]


def targetToNodeListCoords(pos, angle):

    # Rotationsmatrix
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    
    # Rotierte Zielkoordinaten berechnen relativ zu hauptTCP
    rotatedTargets = [pos + R @ offset for offset in TCPOffsets]

    # neue Koordinaten relativ zu ihrer vorherigen Position
    relativeTargetOffsets = np.array(rotatedTargets) - np.array(TCPOffsets)

    # alle zielpunkte anfuegen
    target = [pos[0], pos[1]]
    for i in relativeTargetOffsets:
        target.append(i[0])
        target.append(i[1])


def calcTrajPTP(currentLen, targetPose, ikObj):
    targetPos, targetAng = targetPose
    
    if config.simulation.massPoints2DIdealRevoluteJoint2D:
        refLength = (config.lengths.L0) 
    else:
        refLength = (config.lengths.L0-2*(config.lengths.L1+config.lengths.L2))

    posVec = targetToNodeListCoords(targetPos, targetAng)
    ikObj.InverseKinematicsRelative(None, np.array(posVec), ikineNodeList)
    targetLen = ((np.array(ikObj.GetActuatorLength()) - refLength) * 10000)
    
    realLength = refLength + targetLen / 10000
    if min(realLength) < 0.2285 or max(realLength) > 0.3295:
        print(targetPos, ': liegt außerhalb des Konfigurationsraumes mit max:', max(realLength), 'und min: ', min(realLength))
    
    traj = tf.PTPTrajectory(currentLen, targetLen, vMax, aMax, sync=True)
    return traj


def calcTrajCP(currentLen, poseVec, ikObj, method='spline'):
    if config.simulation.massPoints2DIdealRevoluteJoint2D:
        refLength = (config.lengths.L0) 
    else:
        refLength = (config.lengths.L0-2*(config.lengths.L1+config.lengths.L2))
    
    traj = tf.CPTrajectory(poseVec, 50, ikObj, ikineNodeList, context, refLength)
    return traj


def functionStateMachine(t, l0, ikObj):
    match myState:
        case gripperStates.IDLE:
            print('IDLE')
            myState = gripperStates.APPROACH_OBJECT
        case gripperStates.APPROACH_OBJECT:
            if 'trajApproach' not in mbs.variables:
                print("APPROACH_OBJECT")
                targetPose = ([0.229, 0.1], 0)
                traj = calcTrajPTP(l0, targetPose, ikObj)
                
                # plot_trajectory(traj, context['refLength'])
                
                mbs.variables['trajApproach'] = traj
                mbs.variables['tLocal'] = 0
            else:
                traj = mbs.variables['trajApproach']
            
            tTraj = traj.t_total
            l1 = traj.evaluate(t)
            
            if t+1 > tTraj:
                myState = gripperStates.POSITION_GRIPPER
            
    return l1



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
    
                gContact.AddTrianglesRigidBodyBased(rigidBodyMarkerIndex=mCube, contactStiffness=kFloor, contactDamping=dFloor, frictionMaterialIndex=frictionMaterialIndex,
                                                                pointList=meshPointsCube,  triangleList=meshTrigsCube)
                
                oGround=mbs.AddObject(ObjectGround(referencePosition= p0))
                mGround = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oGround))
                
                mbs.AddObject(GenericJoint(markerNumbers=[mGround, mCube], 
                                            constrainedAxes=[0,0,1,0,0,0],
                                            visualization=VObjectJointGeneric(axesRadius=0.5e-4, axesLength=0.5e-3)))
                
                
                mbs.AddSensor(SensorMarker(markerNumber=mCube, fileName='solution/sensorTest.txt', 
                                            outputVariableType=exu.OutputVariableType.Position))
            
                connectionEntry = [12,3,13,2,8,3,9,2]
                addContactSpheres(gContact,ADE,connectionEntry,sizeOfSpheres,contactSteel,dSteel,frictionMaterialIndex)                
                
            
            
            
            
            
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
                exu.StartRenderer()
                mbs.WaitForUserToContinue()


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
                    gSurface+=[GraphicsDataFromPointsAndTrigs(points,triangles,color=[0.41,0.41,0.41,0.5])]

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
             
            mbs.WaitForUserToContinue()
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
                        mbs.WaitForUserToContinue()      
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
                    
                    
                    
                    
                    t = 0
                    tTotal = 20
                    
                    for i in range(tTotal):
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
                        
                        ground = mbsV2.AddObject(ObjectGround(visualization=VObjectGround(graphicsDataUserFunction=UFgraphicsIK)))
                        mbsV2.Assemble() 
                        
                        
                        l1 = functionStateMachine(t)
                        
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
                        
                        
                        
                        t = t+1
                        
                        
                    #++++++++++++++ old programm ++++++++++++++
                    
                    # for k in range(len(dxx)):
                    #     dx=[]
                    
                    #     for i in range(len(nodeList)):
                    #         if simulateGripper:
                    #             if i ==1:
                    #                 dxx[k][0]=-dxx[k][0]
                    #         else:
                    #             dxx[k][0]=dxx[k][0]
                                
                    #         dx+=list(dxx[k])#+list(dxx[k]) #+list(dxx[k]) #<- change value here to go
                    if simulateGripper:
                        for k in dx2:        
                            dx=dx2[k]
                            
                            setConfigFileKinModel()  # get ActuatorLength from KinModel 
    
    
                            #user function for moving graphics:
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
                            
                            ground = mbsV2.AddObject(ObjectGround(visualization=VObjectGround(graphicsDataUserFunction=UFgraphicsIK)))
                            mbsV2.Assemble() 
                            
                            
                            ikObj.InverseKinematicsRelative(None, np.array(list(dx)))
                            
        
                            
    
                               
        
        
                            ### toDO change here for more nodes the error!!!!
                            iWantToGoToThisValue = mbs2.GetSensorValues(sensorMbs2TCP[0])[0:2]
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
        
            ## END InverseKinematic +++++++++++++++++
            ##+++++++++++++++++++++++++++++++++++++++
            ##+++++++++++++++++++++++++++++++++++++++
            ##+++++++++++++++++++++++++++++++++++++++
            ##+++++++++++++++++++++++++++++++++++++++
        
        
        
        
        
        
            # if displaySimulation and not mbs.GetRenderEngineStopFlag():  
            mbs.WaitForUserToContinue()      
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
        startSimulation = False
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
        
 
