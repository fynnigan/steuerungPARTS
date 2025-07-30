#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 08:25:35 2025

@author: fynnheidenreich
"""

import os
import sys

import csv
from enum import Enum, auto


if os.getcwd().split('/')[-1].__contains__('exampleScripts'):
    os.chdir('..')
if sys.path[0].split('/')[-1].__contains__('exampleScripts'):
    sys.path.append(os.getcwd())

# script_dir = os.path.dirname(os.path.abspath(__file__))
# output_dir = os.path.join(script_dir, 'output')
    

# solutionFilePathRel="./solution/coordinatesSolution.txt"
# solutionFilePath = os.path.join(output_dir, os.path.basename(solutionFilePathRel))
# if os.path.isfile(solutionFilePath):
#     os.remove(solutionFilePath)

script_dir = os.path.dirname(__file__)
solutionFilePath = os.path.join(script_dir, 'solution', 'coordinatesSolution.txt')

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
# from functions import trajectory_functions as tf
import functions.trajectory_functions as tf

import exudyn as exu
from exudyn.utilities import*
from exudyn.interactive import SolutionViewer
from exudyn.graphics import color

import numpy as np
import matplotlib.pyplot as plt
from numpy import absolute, linalg as LA
import math
import logging
from copy import deepcopy
import time
import pickle
import json
import importlib

importlib.reload(tf)


from functions import visualization_functions as vf
from functions.control_functions import *
import matplotlib.ticker as ticker

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
    
    
def targetToNodeListCoords(pos, angle, TCPOffsets):

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
        
    return target


def calcTrajPTP(currentLen, targetPose, context):
    ikObj = context['ikObj']
    ikineNodeList = context['ikineNodeList']
    refLength = context['refLength']
    vMax = context['vMax']
    aMax = context['aMax']
    
    targetPos, targetAng = targetPose

    posVec = targetToNodeListCoords(targetPos, targetAng, context['TCPOffsets'])
    ikObj.InverseKinematicsRelative(None, np.array(posVec), ikineNodeList)
    targetLen = ((np.array(ikObj.GetActuatorLength()) - refLength) * 10000)
    
    realLength = refLength + targetLen / 10000
    if min(realLength) < 0.2285 or max(realLength) > 0.3295:
        print(targetPos, ': liegt außerhalb des Konfigurationsraumes mit max:', max(realLength), 'und min: ', min(realLength))
    
    traj = tf.PTPTrajectory(currentLen, targetLen, vMax, aMax, sync=True)
    return traj

def calcTrajCP(currentLen, poseVec, context, method='spline'):
    ikObj = context['ikObj']
    ikineNodeList = context['ikineNodeList']
    refLength = context['refLength']
    # vMax = context['vMax']
    # aMax = context['aMax']
    
    traj = tf.CPTrajectory(poseVec, 50, ikObj, ikineNodeList, context, refLength)
    return traj


def functionStateMachine(t, actorLengths, posObj, myState, context):
    
    currentLen = context['currentLen']
    tprev = context['tprev']
    
    # FSM Logik
    match myState:
        case gripperStates.IDLE:
            print("IDLE")
            myState = gripperStates.APPROACH_OBJECT
        case gripperStates.TEST_PTP:
            if 'trajPTPTest' not in context:
                targetPose = ([0.229, 0.1], 0)
                traj = calcTrajPTP(currentLen, targetPose, context)
                
                context['trajPTPTest'] = traj
                context['tLocal'] = 0
            else:
                traj = context['trajPTPTest']
                
            tTraj = traj.t_total
            u1 = traj.evaluate(tprev)
            u2 = traj.evaluate(t)
            
            if t+1 > tTraj:
                sys.exit('test durchgeführt')
                
        case gripperStates.TEST_CP:
            if 'trajCPTest' not in context:
                poseVec = [
                    ([0, 0], 0),
                    ([0.229, 0], 0),
                    ([0, 0.229], 0),    
                    ([-0.229*2, 0], 0),    
                    ([0, 0.229], 0)    
                    ([0.229, 0], 0),    
                ]
                # methods: 'spline', 'linear, 'bezier2', 'bezier3', 'bezier5'
                traj = calcTrajCP(currentLen, poseVec, context, method='bezier3')
                
                context['trajCPTest'] = traj
                context['tLocal'] = 0
            else:
                traj = context['trajCPTest']
            
            tTraj = traj.t_total
            u1 = traj.evaluate(tprev)
            u2 = traj.evaluate(t)
            
            if t+1 > tTraj:
                sys.exit('test durchgeführt')
        
    
        case gripperStates.APPROACH_OBJECT:
            
            if 'trajApproach' not in context:
                targetPose = ([0.229, 0.1], 0)
                traj = calcTrajPTP(currentLen, targetPose, context)
                
                # plot_trajectory(traj, context['refLength'])
                
                context['trajApproach'] = traj
                context['tLocal'] = 0
            else:
                traj = context['trajApproach']
                
            tTraj = traj.t_total
            
            u1 = traj.evaluate(tprev)
            u2 = traj.evaluate(t)
            
            print("APPROACH_OBJECT")
            
            if t+1 > tTraj:
                myState = gripperStates.POSITION_GRIPPER
            
        case gripperStates.POSITION_GRIPPER:
            
            
            print("POSITION_GRIPPER")
                        
            if t+1 > tTraj:            
                myState = gripperStates.CLOSE_GRIPPER
    
        case gripperStates.CLOSE_GRIPPER:
            print("CLOSE_GRIPPER")
            myState = gripperStates.MOVE_OBJECT
    
        case gripperStates.MOVE_OBJECT:
            print("MOVE_OBJECT")
            myState = gripperStates.OPEN_GRIPPER
    
        case gripperStates.OPEN_GRIPPER:
            print("OPEN_GRIPPER")
            myState = gripperStates.RETRACT_GRIPPER
    
        case gripperStates.RETRACT_GRIPPER:
            print("RETRACT_GRIPPER")
            myState = gripperStates.IDLE
        
    context["state"] = myState

    if u1 is None or not isinstance(u1, (list, np.ndarray)) or len(u1) == 0:
        print('t > tTraj: error')
        return 0
    else:
        return u1, u2


def PreStepUserFunction(mbs, t):
    """
    Prestep User Function 
    """
    line = mbs.variables['line']
    context = mbs.variables['context']
    
    SensorValuesTime = context['SensorValuesTime']
    adeIDs = context['adeIDs']
    ADE = context['ADE']
    mbsV = context['mbsV']
    numberOfLoadSteps = context['numberOfLoadSteps']
    # state = context['state']

    # for testign
    state = gripperStates.TEST_CP

    if config.simulation.massPoints2DIdealRevoluteJoint2D:
        refLength = (config.lengths.L0) 
    else:
        refLength = (config.lengths.L0-2*(config.lengths.L1+config.lengths.L2)) 
        
    SensorValuesTime[line][t]={}
    
    SensorValues = {} 
    for ID in adeIDs:
        SensorValues[ID] = np.asmatrix([mbs.GetSensorValues(ADE[ID]['sensors'][0])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][1])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][2])[0:2]])   
    
    SensorValuesTime[line][t]=SensorValues
    

    # get current and next displacements, scale t to real time
    u1Vec, u2Vec = functionStateMachine(t*numberOfLoadSteps, 0, 0, state, context)

    for i in range(len(mbs.variables['springDamperDriveObjects'])):
        oSpringDamper = mbs.variables['springDamperDriveObjects'][i]
        # driveLen = absLengths[i]
        u1 = u1Vec[i]/10000
        u2 = u2Vec[i]/10000
        driveLen = UserFunctionDriveRefLen1(t, u1, u2)
        mbs.SetObjectParameter(oSpringDamper,'referenceLength',driveLen+refLength)  
        
        
    mbsV.SendRedrawSignal()
    print('t:', t)
    
    context.update({
        'SensorValuesTime': SensorValuesTime,
        'tprev': t*numberOfLoadSteps,
    })
    
    mbs.variables.update({
        'context': context,
    })
    
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





def createKinematicModel(mbs, mbs2, context):
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
    
    adeNodes = context['adeNodes']
    nodeList = context['nodeList']    
    sensorTCPList = context['sensorTCPList']    

       
    ADE2, mbs2 = GenerateMesh(adeNodes,mbs2)                               #Generating Mesh without Movelist
    
    sensorMbsTCPList=sensorList(nodeList, adeNodes, mbs2, context)
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
    
    context.update({
        'sensorTCPList': sensorTCPList,
    })
                  
    return ikObj, mbs2




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


    setConfigFileSurrogateModel()



    if startSimulation and (type(adeNodes) == df.ElementDefinition or type(adeNodes) == dict):
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

def sensorList(nodeList, adeNodes, mbs, context):  
    ADE = context['ADE']
    sensorTCPList=[]
    for node in nodeList:
        getADEFromNodeNumber=next(iter(adeNodes.adePointsOfNode[node]))
        adeIndex012OfNodeFromADEID=adeNodes.nodeNumbersToAdeIndex[getADEFromNodeNumber][0]
        sensorTCP = mbs.AddSensor(SensorMarker(markerNumber=mbs.GetSensor(ADE[getADEFromNodeNumber]['sensors'][adeIndex012OfNodeFromADEID])['markerNumber'], outputVariableType=exu.OutputVariableType.Position, storeInternal=True))
        sensorTCPList +=[sensorTCP]
    return sensorTCPList


# bekommt breite, höhe und datei pfad des Greifers
def createArm(nx, ny, gipperFilePath, context):
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
    meshPath = os.path.join(script_dir, "data", "experiments", "gripper", "initial9_gripper")
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

    
    # save everything into context
    context.update({
        "adeNodes": adeNodes,
        "nodeList": nodeList,
        "TCPID": TCPID,
        "TCPcoords": TCPcoords,
        "TCPRefCoord": TCPRefCoord,
        "TCPOffsets": TCPOffsets,
    })
    return True
    
    
def initializeSimulation(context):
    print('initing sim')
    SC = context['SC']
    
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
    SC.visualizationSettings.connectors.show= True

    SC.visualizationSettings.openGL.lineWidth=2
    SC.visualizationSettings.openGL.lineSmooth=True
    SC.visualizationSettings.openGL.multiSampling = 4
    SC.visualizationSettings.general.drawCoordinateSystem = True
    SC.visualizationSettings.window.renderWindowSize=[1600,1000]
    
    SC.visualizationSettings.contact.showSearchTree = True
    SC.visualizationSettings.contact.showSearchTreeCells = True
    SC.visualizationSettings.contact.showBoundingBoxes = True
    
    if context['visualizeTrajectory']:
        SC.visualizationSettings.sensors.traces.showPositionTrace = True
        
    if config.simulation.activateWithKeyPress:
        SC.visualizationSettings.window.keyPressUserFunction = UserFunctionkeyPress
        
        if config.simulation.showJointAxes:
            SC.visualizationSettings.connectors.defaultSize = 0.002
            SC.visualizationSettings.connectors.showJointAxes = True
            SC.visualizationSettings.connectors.jointAxesLength = 0.0015
            SC.visualizationSettings.connectors.jointAxesRadius = 0.0008
    
    
    context.update({
        'simulationSettings': simulationSettings,
        'SC': SC,
    })


def plot_trajectory(traj, refLength, every_n=1, label_prefix="Actuator"):
    time_steps = 100  # feinere Auflösung
    t_vals = np.linspace(0, traj.t_total, time_steps)
    lengths = np.array([refLength + traj.evaluate(t)/10000 for t in t_vals])

    num_actuators = lengths.shape[1]

    plt.figure(figsize=(10, 6))
    for i in range(0, num_actuators, every_n):
        if i == 0:
            plt.plot(t_vals, lengths[:, i], linewidth=0.3, label='Akuatoren', color='blue')
        else:
            plt.plot(t_vals, lengths[:, i], linewidth=0.3, color='blue')
            
    # Fixe y-Achse
    plt.ylim(0.22, 0.36)
    
    # Min-/Max-Hub als gestrichelte Linien
    plt.axhline(0.229, color='gray', linestyle='--', linewidth=1, label='Min-Hub (0.229 m)')
    plt.axhline(0.329, color='gray', linestyle='--', linewidth=1, label='Max-Hub (0.329 m)')

    
    plt.title(f"Zeitliche Längenänderungen der Aktuatoren")
    plt.xlabel("Zeit [s]")
    plt.ylabel("Länge [m]")
    plt.grid(True, alpha=0.5)
    plt.legend(loc='upper right')
    # plt.tight_layout()
    plt.show()


##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Initialization of Module


class gripperFSM:
    def __init__(self):
        
        config.init()
        logging.info('Starte Simulations Module')
        
        # client = wf.WebServerClient()
        SC = exu.SystemContainer()
        mbs = SC.AddSystem()
        
        state = gripperStates.APPROACH_OBJECT
        
        context = {}
        mbs.variables['context'] = context
        
        
        timeTotal=True
        addTimeStaticStep=False
        numberOfSteps=0
        
        
        
        config.simulation.constrainActorLength = True        
        
        my_dir = os.path.abspath(os.path.join(os.getcwd(), 'functions'))
        
        TCPID = 0
        
        createArm(1, 4, os.path.join(script_dir, "data", "experiments", "gripper", "initial9_gripper"), context)

        adeNodes = context['adeNodes']
        nodeList = context['nodeList']
        TCPOffsets = context['TCPOffsets']
        
        
        oGround = mbs.AddObject(ObjectGround(referencePosition = [0,0,0]))
        
        useInverseKinematic = True

        config.simulation.useMoveList = False
        if config.simulation.setMeshWithCoordinateConst:
            config.simulation.numberOfBoundaryPoints = 0 #change here number of boundary 
        
        
        #Options for Visualization
        config.simulation.frameList = True
        config.simulation.showJointAxes=False
        config.simulation.animation=True
        config.simulation.showLabeling = False #shows nodeNumbers, ID and sideNumbers, simulationTime
        
        
        context.update({
            'visualizeTrajectory': True,
            'visualizeSurfaces': False,   
            'viusalizeGrid': True,
            'CoordinateSystemTCP': True,
            'state': state,
        })
        
        
        
        usePTP = False
        useSyncPTP = True
        useCPspline = False
        
        
        # +++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++
        ## start program ++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++
        timeTotal=0
        numberOfSteps=0
        
        
        try:
            mbs, ADE, adeIDs = createSurrogateModel(mbs, adeNodes)
            
            context.update({
                'SC': SC,
                'mbs': mbs,
            })
            initializeSimulation(context)
            simulationSettings = context['simulationSettings']
            SC = context['SC']
    
            
            
            # create sensor for "MCP"   
            context['ADE'] = ADE
            context['adeIDs'] = adeIDs
            sensorTCPList=sensorList(nodeList, adeNodes, mbs, context)
            sensorTCP=sensorTCPList
    
            mbsV = SC.AddSystem()
            mbsV.variables['mbs']=mbs
            mbs.variables['mbsV']=mbsV
            
            if context['visualizeSurfaces']:
                def UFgraphics2(mbs, objectNum):
                    mbsV = mbs.variables['mbsV']
                    mbsV.SendRedrawSignal()
                    return []
                
                if context['visualizeTrajectory']:
                    ground2 = mbs.AddObject(ObjectGround(visualization=VObjectGround(graphicsDataUserFunction=UFgraphics2)))        
            
            
            if context['viusalizeGrid']:
                nTiles = 14
                edgeLength = 0.229
                checkerboard = GraphicsDataCheckerBoard(point=[0,edgeLength*(nTiles/2-2),-0.02],size=edgeLength*nTiles, nTiles=nTiles)
                
                mbs.AddObject(ObjectGround(
                    visualization=VObjectGround(graphicsData=[checkerboard])
                ))
            
            ##+++++++++++++++++++++++++++++++++++++++
            ## Assemble Mesh 
            mbs.Assemble()
            sysStateList = mbs.systemData.GetSystemState()
    
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
                SensorValues[0][ID] = np.asmatrix([
                    mbs.GetSensorValues(ADE[ID]['sensors'][0])[0:2],
                    mbs.GetSensorValues(ADE[ID]['sensors'][1])[0:2],
                    mbs.GetSensorValues(ADE[ID]['sensors'][2])[0:2]
                ])
            # client.Put('SensorValues',pickle.dumps(SensorValues))
            SensorValuesTime[0]=SensorValues.copy()
            ##+++++++++++++++++++++++++++++++++++++++
            for ID in adeIDs:
                SensorValues2[0][ID] = np.asmatrix([
                    mbs.GetSensorValues(ADE[ID]['sensors'][0])[0:2],
                    mbs.GetSensorValues(ADE[ID]['sensors'][1])[0:2],
                    mbs.GetSensorValues(ADE[ID]['sensors'][2])[0:2]
                ])
            # client.Put('SensorValues',pickle.dumps(SensorValues))
            SensorValuesTime2[0]=SensorValues2.copy()
            
            if context['visualizeTrajectory']:
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
            
            context['mbsV'] = mbsV
    
    
            
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
    
            exu.SolveStatic(mbs, simulationSettings=simulationSettings, showHints=True)
            # exu.SolveDynamic(mbs, simulationSettings=simulationSettings, showHints=True)
            
            
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
                cnt=-1
    
                
                
                config.simulation.useInverseKinematic=True
                ##+++++++++++++++++++++++++++++++++++++++
                ## Starting Simulation
                if config.simulation.useInverseKinematic == True:
        
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
                    
                    context.update({
                        'adeNodes': adeNodes,
                        'nodeList': nodeList,
                        'sensorTCPList': sensorTCPList,
                    })
        
                    ikObj, mbs2 = createKinematicModel(mbs, mbs2, context)
                    # ikObj2, mbs4 = createKinematicModel(mbs, mbs2, context)
                    
                    ########################################################
                    
                    cnt=0
                    actorList = []
                    sortedActorStrokesList=[] 
                     
                    setConfigFileKinModel()  # get ActuatorLength from KinModel 
                    # l1 = ikObj.GetActuatorLength()
                    
        
                    if config.simulation.massPoints2DIdealRevoluteJoint2D:
                        refLength = (config.lengths.L0) 
                    else:
                        refLength = (config.lengths.L0-2*(config.lengths.L1+config.lengths.L2))
   
                    
                    # set max velocity and acceleration
                    context['vMax'] = 20
                    context['aMax'] = 10
                  
                    
                    # # calculate trajectory for every actuator
                    # if usePTP:
                    #     traj = tf.PTPTrajectory(startLengths, targetLengths, vmax, amax)
                    # elif useSyncPTP:
                    #     traj = tf.syncPTPTrajectory(startLengths, targetLengths, vmax, amax)
                    
                    # object coords
                    # objPos=np.array([0.229, 0.8])
                    # call object spawning function here
                    
                    # vector for node positition for ikine calculations
                    # trajPositions = []
                    
                    # vector with positions as tuple ([x, y], angle)
                    # poseVec = [
                    #     ([objPos[0], objPos[1]-0.7], 0),
                    #     ([0, 0.4], 0)
                    # ]
                    # test for CP
                    # poseVec = [
                    #     ([0.229, 0], 0),
                    #     ([0, 0.4], 0),    
                    #     ([-0.229*2, 0], 0),    
                    #     ([0, 0.4], 0)    
                    # ]
                    
                    # poseVec = [
                    #     ([0, 0], 0),
                    #     ([1, 1.5], 0),
                    #     ([3, 0.5], 0),
                    #     ([2, 2.5], 0)
                    # ]
                    
                    # maximum nach rechts mit winkel=0: 1
                    # maximum nach links mit winkel=0: 0.8
                    # maximum nach oben mit winkel=0: 0.4                  
                    # poseVec = [
                    #      ([0, 0.3], 0),    
                    #      ([0, 0.1], 0),    
                    #      ([0, 0.1], 0),    
                    #      ([0, 0.1], 0),    
                    #      ([0, 0.1], 0),    
                    #      ([0, 0.1], 0),    
                    #      ([0, 0.1], 0),    
                    #      ([0, 0.1], 0),    
                    #      ([0, 0.1], 0),    
                    # ]
                    # poseVec = [
                    #     ([0.229, 0], np.pi/8)    
                    # ]
                
                    # store node out of nodeList to use for ikine calculation                    
                    ikineNodeList = [i for i in range(len(nodeList))]
                    
                    # get actor lengths for start position
                    # start = np.array([0] * len(ikineNodeList) * 2)
                    # ikObj.InverseKinematicsRelative(None, start, ikineNodeList)
                    prevActorLengths = ((np.array(ikObj.GetActuatorLength()) - refLength) * 10000)
                    context['currentLen'] = prevActorLengths
                    
                    
                    # pose = targetToNodeListCoords([0.229, 0.229], -np.pi/16, context['TCPOffsets'])
                    # ikObj2.InverseKinematicsRelative(None, pose, ikineNodeList)
                    
                    # vector for all trajectories
                    trajVec = []
                    
                    # if useSyncPTP:
                    #     # calculate all trajectories
                    #     for pos, angle in poseVec:
                    #         posVec = targetToNodeListCoords(pos, angle, context['TCPOffsets'])
                    #         ikObj.InverseKinematicsRelative(None, np.array(posVec), ikineNodeList)
                    #         actorLengths = ((np.array(ikObj.GetActuatorLength()) - refLength) * 10000)#
                    #         realLength = refLength + actorLengths / 10000
                    #         if min(realLength) < 0.2285 or max(realLength) > 0.3295:
                    #             print(pos, ': liegt außerhalb des Konfigurationsraumes mit max:', max(realLength), 'und min: ', min(realLength))
                    #             break
                    #         trajVec.append(tf.PTPTrajectory(prevActorLengths, actorLengths, vmax, amax, sync=True))
                    #         prevActorLengths = actorLengths
                    # elif useCPspline:
                    #     trajVec.append(tf.CPTrajectory(poseVec, 20, ikObj, ikineNodeList, context, refLength, method='linear'))
                    # else:
                    #     print('no traj type selected')
                        
                    
                    # time all trajactories need in total
                    # tTotal = sum(traj.t_total for traj in trajVec)
                    
                    # plot_trajectory(trajVec[0], refLength)
                    
                    
                    # dt = 1
                    # numberOfLoadSteps = int(np.ceil(tTotal) / dt)
                    numberOfLoadSteps = 25
                    simulationSettings.staticSolver.numberOfLoadSteps = numberOfLoadSteps
                    simulationSettings.staticSolver.loadStepGeometric = False
                    print('numberOfLoadSteps:', numberOfLoadSteps)
                    
                    context.update({
                        'refLength': refLength,
                        'sortedActorStrokesList': sortedActorStrokesList,
                        'simulationSettings': simulationSettings,
                        'mbs': mbs,
                        'timeTotal': timeTotal,
                        'numberOfSteps': numberOfSteps,
                        'SensorValuesTime': SensorValuesTime,
                        'SensorValues': SensorValues,
                        'SensorValues2': SensorValues2,
                        'SensorValuesTime2': SensorValuesTime2,
                        'cnt': cnt,
                        'counterSensorPlot': counterSensorPlot,
                        'trajectoryVector': trajVec,
                        'numberOfLoadSteps': numberOfLoadSteps,
                        # 'tTotal': tTotal,
                        'ikObj': ikObj,
                        'ikineNodeList': ikineNodeList,
                        'tprev': 0,
                    })
                    
                    mbs.variables.update({'context': context})
                

                    mbs.variables['line']=cnt+1
                    SensorValuesTime[cnt+1]={} 

                    # setConfigFileKinModel()    
                    setConfigFileSurrogateModel()

                    simulationSettings.displayComputationTime = False  
                    simulationSettings.timeIntegration.verboseMode = 0
                    simulationSettings.displayStatistics = False       
                    
                    
                    mbs.SetPreStepUserFunction(PreStepUserFunction)
                    

                    simulationSettings.staticSolver.verboseMode=0
                    
                    context.update({
                        'SensorValuesTime': SensorValuesTime,
                        # 'sortedActorStrokesList': sortedActorStrokesList,
                        'cnt': cnt,
                    })
                    mbs.variables.update({'context': context})

                    exu.SolveStatic(mbs, simulationSettings,showHints=True) 
                    # exu.SolveDynamic(mbs, simulationSettings,showHints=True) 
                    
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
                            
                    
                    mbs.WaitForUserToContinue() 
                    
            
            if not mbs.GetRenderEngineStopFlag():  
                mbs.WaitForUserToContinue()      
                exu.StopRenderer() #safely close rendering window!
                
            try:
                if config.simulation.solutionViewer:
                      sol = LoadSolutionFile(config.simulation.solutionViewerFile, safeMode=True)#, maxRows=100) 
                      SolutionViewer(mbs,sol) #can also be entered in 
            except:
                logging.warning('Couldnt load Solution File')
    
    
            ##+++++++++++++++++++++++++++++++++++++++
            # sensorValuesPath = os.path.join(output_dir, os.path.basename(config.simulation.sensorValuesFile))
            # sensorValuesTimePath = os.path.join(output_dir, os.path.basename(config.simulation.sensorValuesTimeFile))
            # sensorValuesPath2 = os.path.join(output_dir, os.path.basename(config.simulation.sensorValuesFile2))
            # sensorValuesTimePath2 = os.path.join(output_dir, os.path.basename(config.simulation.sensorValuesTimeFile2))
            
            ## Saving Sensor Data to File
            if config.simulation.saveSensorValues:
                  # pickle.dump(  SensorValues,  open(config.simulation.sensorValuesFile,  'wb')  )
                  # pickle.dump(  SensorValuesTime,  open(config.simulation.sensorValuesTimeFile,  'wb')  )
                  
                  # pickle.dump(  SensorValues2,  open(config.simulation.sensorValuesFile2,  'wb')  )
                  # pickle.dump(  SensorValuesTime2,  open(config.simulation.sensorValuesTimeFile2,  'wb')  )
                  
                  pickle.dump(SensorValues,  open(sensorValuesPath,  'wb')  )
                  pickle.dump(SensorValuesTime, open(sensorValuesTimePath, 'wb'))
                  pickle.dump(SensorValues2, open(sensorValuesPath2, 'wb'))
                  pickle.dump(SensorValuesTime2, open(sensorValuesTimePath2, 'wb'))
                 
            ##+++++++++++++++++++++++++++++++++++++++
            ## Resetting Simulation
            mbs.Reset()
        
        except Exception as ExeptionMessage: 
            logging.warning('Error ocured during Simulation')
            logging.warning(ExeptionMessage)
            # mbs.Reset()
        # time.sleep(config.simulation.looptime-time.time()%config.simulation.looptime)
        
        exu.StopRenderer()
        print('time: ', timeTotal, 'numberOfSteps: ',numberOfSteps)
        print('userFunction: ',mbs.variables['counterForTesting'])


    def nextState(self, context):
        print(f'\n >> State: {self.state.name}')
        
        if self.state == gripperStates.IDLE:
            print("System ist bereit. Warte auf Startsignal.")
            self.state = gripperStates.APPROACH_OBJECT

        elif self.state == gripperStates.APPROACH_OBJECT:
            print("Annäherung an das Objekt...")
            
            
            
            
            
            
            
            
            
            self.state = gripperStates.POSITION_GRIPPER

        elif self.state == gripperStates.POSITION_GRIPPER:
            print("Greifer wird in Position gebracht...")
            self.state = gripperStates.CLOSE_GRIPPER
        
        elif self.state == gripperStates.CLOSE_GRIPPER:
            print("Greifer schließt...")
            self.state = gripperStates.MOVE_OBJECT
        
        elif self.state == gripperStates.MOVE_OBJECT:
            print("Objekt wird bewegt...")
            self.state = gripperStates.OPEN_GRIPPER
        
        elif self.state == gripperStates.OPEN_GRIPPER:
            print("Greifer öffnet...")
            self.state = gripperStates.RETRACT_GRIPPER
        
        elif self.state == gripperStates.RETRACT_GRIPPER:
            print("Greifer fährt zurück...")
            self.state = gripperStates.IDLE

if __name__ == "__main__":
    fsm = gripperFSM()



