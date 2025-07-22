#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Data and information that support the findings of the article:
# M. Pieber and Z. Zhang and P. Manzl and J. Gerstmayr,  
# Surrogate models for compliant joints in programmable structures
# Proceedings of the ASME 2024
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
    if os.getcwd().split('/')[-1].__contains__('functions'):
        os.chdir('..')
    if sys.path[0].split('/')[-1].__contains__('functions'):
        sys.path.append(os.getcwd())


solutionFilePath="/Users/fynnheidenreich/Desktop/Uni/6/bachelor/HeidenreichFynn/programPARTS24/solution/coordinatesSolution.txt"
if os.path.isfile(solutionFilePath):
    os.remove(solutionFilePath)
    

import config
config.init()

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

from numba import jit

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
import numpy as np



    
def map_to_range(value, min_value, max_value, new_min, new_max):
    normalized_value = (value - min_value) / (max_value - min_value)
    mapped_value = (normalized_value * (new_max - new_min)) + new_min
    return mapped_value



### values for springs for connectors between two elements
kk = config.lengths.stiffnessConnectors #stiffness
dd = config.lengths.dampingConnectors #damping

## Visualization
if config.simulation.frameList:
    fL = 0.01 #frame length
    graphicsX = {'type':'Line', 'color':[0.8,0.1,0.1,1], 'data':[0,0,0, fL,0,0]}
    graphicsY = {'type':'Line', 'color':[0.1,0.8,0.1,1], 'data':[0,0,0, 0,fL,0]}
    graphicsZ = {'type':'Line', 'color':[0.1,0.1,0.8,1], 'data':[0,0,0, 0,0,fL]}
    frameList = [graphicsX,graphicsY,graphicsZ]
else:
    frameList = [] #no frames


def PreStepUserFunction2(mbs, t):
    """
    Prestep User Function 
    """
    # print('preNewton')
    #print("called")
    #print("mbs=", mbs2)
    LrefExt = mbs.variables['LrefExt']
    Actors = mbs.variables['springDamperDriveObjects']
    Connectors = mbs.variables['Connectors']
    Vertices = mbs.variables['VertexLoads']
    TimeStamp = mbs.variables['TimeStamp']
    line = mbs.variables['line']
    
    sensorError=mbs.variables['sensorTCP']
    
    
    # sensorError=sensorError[0]
    # positionInteractivNode = mbs.GetSensorValues(sensorError)


 
    # config.simulation.massPoints2DIdealRevoluteJoint2D = True
    if config.simulation.massPoints2DIdealRevoluteJoint2D:
        refLength = (config.lengths.L0) 
    else:
        refLength = (config.lengths.L0-2*(config.lengths.L1+config.lengths.L2))  
    
 
    # refLength = (config.lengths.L0-2*(config.lengths.L1+config.lengths.L2))
    
        
    for i in sensorError:
        positionInteractivNode = mbs.GetSensorValues(i)
        if config.simulation.constrainActorLength:
            for oSpringDamper in mbs.variables['springDamperDriveObjects']: 
                ChangeActorStiffnessStartingNode(mbs ,t, oSpringDamper,positionInteractivNode)
                
        
    # if config.simulation.massPoints2DIdealRevoluteJoint2D:
    #     mapIdealToReal= 2*(config.lengths.L1+config.lengths.L2)
    # else:
    #     mapIdealToReal=0
        
        
        
        
    for i, oSpringDamper in enumerate(mbs.variables['springDamperDriveObjects']): 
        actorLengthNew=mbs.GetObjectOutput(oSpringDamper,  exu.OutputVariableType.Distance, ) - refLength
        
        if config.simulation.constrainActorLength:
            if actorLengthNew<config.lengths.minStroke:
                actorLengthNew=config.lengths.minStroke
            if actorLengthNew>config.lengths.maxStroke:
                actorLengthNew=config.lengths.maxStroke
        
        mbs.SetObjectParameter(oSpringDamper,'referenceLength',actorLengthNew+refLength)   
        
        
                    
    return True #succeeded


# def PostNewtonUserFunction(mbs, t):
#     # print('STARTpostNewton')
#     refLength = (config.lengths.L0-2*(config.lengths.L1+config.lengths.L2))
    
#     for i, oSpringDamper in enumerate(mbs.variables['springDamperDriveObjects']): 
#         actorLengthNew=mbs.GetObjectOutput(oSpringDamper,  exu.OutputVariableType.Distance, ) - refLength
        
#         if actorLengthNew<0:
#             actorLengthNew=0
#         if actorLengthNew>0.1:
#             actorLengthNew=0.1
        
#         # print(actorLengthNew)
        
#         mbs.SetObjectParameter(oSpringDamper,'referenceLength',actorLengthNew+refLength)  
        
#     retry = 2
#     # if(t>1):
#     #     return [0, 1e-6]
#     return [retry,0.2] # [discontinousError, recommendedStepSize]        


class InverseKinematicsATE(): 
    def __init__(self, myMbs, sensorsError, mbs, oGround,ADE): 
        self.mbsIK = myMbs
        # self.nodesError = nodesError
        self.sensorsError = sensorsError 
        # self.sensorsForSpringsToGround = sensorsForSpringsToGround 
        self.markerError = []
        self.markerForAllToGround=[]
        self.mGroundList = []
        self.nodeList = []
           
        self.mbsIK.SetPreStepUserFunction(PreStepUserFunction2)   
        # self.mbsIK.SetPostNewtonUserFunction(PostNewtonUserFunction)
        
        for sensor in sensorsError: 
            mySensor = self.mbsIK.GetSensor(sensor)
            if mySensor['outputVariableType'] != exu.OutputVariableType.Position or mySensor['sensorType'] != 'Marker': 
                raise ValueError("Only InverseKinematicsATE only allows SensorMarker with Position output.")
            self.markerError += [mySensor['markerNumber']]
        
        
        self.position0 = []
        self.constraints = []
        self.l0 = mbs.variables['LrefExt']
        
        # 
        for i in range(len(self.markerError)):  
            p0 = self.mbsIK.GetSensorValues(self.sensorsError[i])
            self.position0 += p0[:2].tolist()

        

           
        # ###########    
        # ###########
        self.allSensors=[]
        for index in ADE:
            if index >= 1:
                for i in range(3):
                    self.allSensors+= [ADE[index]['sensors'][i]]
                    mySensor= self.mbsIK.GetSensor(ADE[index]['sensors'][i])
                    self.markerForAllToGround += [mySensor['markerNumber']]
                
        self.position02 = []
        self.constraints2 = []
        self.mGroundList2=[]
        # 
        for i in range(len(self.markerForAllToGround)):  
            p0 = self.mbsIK.GetSensorValues(self.allSensors[i])
            self.position02 += p0[:2].tolist()


        # create constraints to ground
        for i in range(len(self.markerForAllToGround)): 
            self.mGroundList2 += [self.mbsIK.AddMarker(MarkerBodyRigid(bodyNumber=oGround, localPosition = self.position02[i*2:i*2+2] + [0]))]
            # self.constraints2 += [self.mbsIK.AddObject(RevoluteJoint2D(markerNumbers=[self.mGroundList2[-1], self.markerForAllToGround[i]]))]
            k=[1e1,1e1,0]
            d=[k[0]*1e-2,k[0]*1e-2,0]
            self.constraints2 += [self.mbsIK.AddObject(CartesianSpringDamper(markerNumbers=[self.mGroundList2[-1], self.markerForAllToGround[i]], stiffness=k, damping=d, visualization=VNodePointGround(show=False)))]
        ###########    
        # ###########



        # create constraints on the nodes to compensate the error
        for i in range(len(self.markerError)): 
            self.mGroundList += [self.mbsIK.AddMarker(MarkerBodyRigid(bodyNumber=oGround, localPosition = self.position0[i*2:i*2+2] + [0]))]
            self.constraints += [self.mbsIK.AddObject(RevoluteJoint2D(markerNumbers=[self.mGroundList[-1], self.markerError[i]]))]
            
            # k=[1e15,1e15,0]
            # d=[1e2,1e2,0]
            # self.constraints += [self.mbsIK.AddObject(CartesianSpringDamper(markerNumbers=[self.mGroundList[-1], self.markerError[i]], stiffness=k, damping=d))]        
        
        
                
        # for oSpringDamper in myMbs.variables['springDamperDriveObjects']: 
        #     self.mbsIK.SetObjectParameter(oSpringDamper, 'stiffness', 1)
        
        # for oSpringDamper in myMbs.variables['springDamperDriveObjects']: 
        #     ChangeActorStiffnessStartingNode(self.mbsIK ,0, oSpringDamper,ADE)
            
        # for itemNumber in Actors:
        #     ChangeActorStiffness(mbs,t,itemNumber)
        
        self.nConstraints = len(self.constraints)
            

        # set simulation settings for static solver 
        self.simulationSettings = exudyn.SimulationSettings()
        self.simulationSettings.solutionSettings.writeSolutionToFile = False
        self.simulationSettings.solutionSettings.binarySolutionFile = False
        # self.simulationSettings.linearSolverSettings.ignoreSingularJacobian = True
        self.simulationSettings.displayComputationTime = False
        self.simulationSettings.displayStatistics = False
        
        self.simulationSettings.staticSolver.newton.maxIterations = 20 # ursprünglihc 50
        self.simulationSettings.staticSolver.adaptiveStep = True
        self.simulationSettings.staticSolver.verboseMode = 0
        self.simulationSettings.displayGlobalTimers = 0
            
        self.simulationSettings.linearSolverType = exu.LinearSolverType.EigenSparse
        self.simulationSettings.staticSolver.newton.weightTolerancePerCoordinate = True 
       
        self.simulationSettings.staticSolver.discontinuous.iterationTolerance = 1 # 1 initial value
        self.simulationSettings.staticSolver.discontinuous.maxIterations = 2 # 5 initial value
        
        self.simulationSettings.displayStatistics = False     
        self.simulationSettings.displayComputationTime = False  
            
        # self.simulationSettings.staticSolver.newton.relativeTolerance = 1e-4
        # self.simulationSettings.staticSolver.newton.absoluteTolerance = 1e-4
        
        # self.simulationSettings.staticSolver.numberOfLoadSteps = 50
        self.simulationSettings.staticSolver.numberOfLoadSteps = 10 # ursprünglihc 25
        # self.simulationSettings.staticSolver.newton.maxIterations = 50 
        # self.simulationSettings.staticSolver.adaptiveStep = True
        # self.simulationSettings.staticSolver.verboseMode = 0
        # self.simulationSettings.displayGlobalTimers = 0
        self.staticSolver = exudyn.MainSolverStatic()
        
        
        # assemble with new constraints
        self.mbsIK.Assemble()
        
    def InverseKinematicsRelative(self, mbs1, offsetList, TCPlist): 
        # actorLength_current = []
        # if len(offsetList) != self.nConstraints*2: 
        #     raise ValueError("InverseKinematicsATE: function InverseKinematicsRelative expected offsetList of length {} but got {}".format(self.nConstraints*2, len(offsetList)))
            
        if not(mbs1 is None): 
            for i, oSpringDamper in enumerate(mbs1.variables['springDamperDriveObjects']): 
                actorLength_current =  mbs1.GetObjectParameter(oSpringDamper,  'referenceLength')
                self.mbsIK.SetObjectParameter(self.mbsIK.variables['springDamperDriveObjects'][i], 'referenceLength', actorLength_current)
            sysStateList = mbs1.systemData.GetSystemState()
            # self.mbsIK.systemData.SetODE2Coordinates(sysStateList[0], exu.ConfigurationType.Initial)
            # self.mbsIK.systemData.SetODE2Coordinates_t(sysStateList[1], exu.ConfigurationType.Initial)
            # self.mbsIK.systemData.SetAECoordinates(np.concatenate([sysStateList[3], [0,0]*self.nConstraints]), exu.ConfigurationType.Initial)
        else: 
            sysStateList = self.mbsIK.systemData.GetSystemState()
            self.mbsIK.systemData.SetSystemState(sysStateList, exu.ConfigurationType.Initial)
            mbs1 = self.mbsIK
        
        currentPos = []
        # for i in range(len(self.sensorsError)): 
        for i in TCPlist:
            p0 = self.mbsIK.GetSensorValues(self.sensorsError[i])
            currentPos += p0[:2].tolist()
            # currentPos  += mbs1.GetNodeOutput(self.nodesError[i], exu.OutputVariableType.Displacement, )[0:2].tolist()
        
        # for i in range(self.nConstraints): 
        #     newPos = currentPos[i*2:i*2+2] + offsetList[i*2:i*2+2]
        #     self.mbsIK.SetMarkerParameter(self.mGroundList[i], 'localPosition', newPos.tolist() + [0])
        for i in range(len(TCPlist)):
            newPos = currentPos[i*2:i*2+2] + offsetList[i*2:i*2+2]
            self.mbsIK.SetMarkerParameter(self.mGroundList[TCPlist[i]], 'localPosition', newPos.tolist() + [0])
            
            # self.mbsIK.SetObjectParameter(self.constraints[i], 'offset', currentPos[i] + offsetList[i])
            # print('constraint {}, val {}'.format(i, currentPos[i] + offsetList[i] ))
            
        self.SolveSystem()
        
        return
        
    def InverseKinematics(self, mbs1, errorList): 
        
        pass
    
    def GetActuatorLength(self): 
        actorLength = []
        
            
        for i, oSpringDamper in enumerate(self.mbsIK.variables['springDamperDriveObjects']): 
            actorLengthNew=self.mbsIK.GetObjectOutput(oSpringDamper,  exu.OutputVariableType.Distance, )  #- self.l0[i]
            
            # actorLengthNew=(np.round(actorLengthNew, 5)*10000).tolist()
            # print(actorLengthNew)
            
            # if actorLengthNew <= 0:
            #     # actorLengthNew=0
            #     print('stroke too low')
            # if actorLengthNew >= 0.1:
            #     # actorLengthNew = 0.1
            #     print('stroke too high')


            actorLength += [ actorLengthNew ]
        return actorLength
        
    def SolveSystem(self, verbose=False):
        success = self.staticSolver.SolveSystem(self.mbsIK, self.simulationSettings)
        if verbose: 
            if success: 
                print('*************************\nIK was successful\n*************************')
            else: 
                print('*************************\nIK NOT successful\n*************************')
        
        return success
    
    
# def GetActuatorLength(mbs): 
#     actorLength = []
#     l0 = mbs.variables['LrefExt']
    
#     for i, oSpringDamper in enumerate(mbs.variables['springDamperDriveObjects']): 
#         actorLengthNew=mbs.GetObjectOutput(oSpringDamper,  exu.OutputVariableType.Distance, ) - l0[i]       
        
#         # if actorLengthNew <= 0:
#         #     actorLengthNew=0
#         #     print('stroke too low')
#         # if actorLengthNew >= 0.1:
#         #     actorLengthNew = 0.1
#         #     print('stroke too high')

#         actorLength += [ actorLengthNew ]
#     return actorLength   
    
class CSixBarLinkageParameters:
    def __init__(self, L1 = config.lengths.L1, 
                 L2 = config.lengths.L2, 
                 L3 = config.lengths.L3, 
                 L4 = config.lengths.L4, 
                 massRigid = config.lengths.weightBeams, 
                 inertiaRigid = config.lengths.inertiaBeams,
                 fFriction = config.simulation.fFriction, #0.274146552249392
                 zeroZoneFriction = config.simulation.zeroZoneFriction): #initial length, mass and inertia
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.L4 = L4
        self.massRigid = massRigid
        self.inertiaRigid =inertiaRigid
        self.fFriction =fFriction
        self.zeroZoneFriction=zeroZoneFriction
        self.psi = np.arctan(self.L3/self.L1)
        self.CE = self.L3/np.sin(np.arctan(self.L3/self.L1))    

    def getGeometricParameters(self):
        return [self.L1,self.L2,self.L3,self.L4,self.psi,self.CE]

    def getMaterialParameters(self):
        return [self.massRigid,self.inertiaRigid]

class CSixBarLinkageParametersConnectors:
    def __init__(self, k1 = [40996.40177929759,5899.28918022937,0.006295515837712327,-9188.909873844023, 5.028748265681435,-0.15573908870458356,0,0,4.183050877058288], d1 = [1e4,1e4,1e2*0]): #initial stiffness and damping
        self.k1 = k1
        self.d1 = d1

class CSixBarLinkageVisualization:
    def __init__(self):
        self.bLength = 0.5e-3 #initial length
        self.graphicsCE = []
        self.graphicsL2 = []
        self.graphicsCE1 = []
        self.graphicsL21 = []
        self.graphicsCE2 = []
        self.graphicsL22 = []
        
class CSixBarLinkage:
    def __init__(self, centerOfRotation = [0,0], 
                 orientation = 0., 
                 startAngle = np.pi/4, 
                 SixBarLinkageParameters = CSixBarLinkageParameters, 
                 SixBarLinkageParametersConnectors = CSixBarLinkageParametersConnectors, 
                 SixBarLinkageVisualization = CSixBarLinkageVisualization):
        self.centerOfRotation = centerOfRotation
        self.orientation = orientation
        self.startAngle = startAngle
        self.SixBarLinkageParameters = SixBarLinkageParameters()
        self.SixBarLinkageParametersConnectors = SixBarLinkageParametersConnectors()
        self.SixBarLinkageVisualization = SixBarLinkageVisualization()
    coordinatePoints=[]
    markersForPrismaticJoints=[]
    markersForConnectors=[]
    
    #ToDo
    nodes = []

class CATCParameters: #spring for prismaticJoint (Actuator) 
    def __init__(self, stiffness = config.lengths.stiffnessPrismaticJoints, damping = config.lengths.dampingPrismaticJoints): #initial stiffness and damping
        self.stiffness = stiffness
        self.damping = damping

class CATC:
    SixBarLinkage = {}
    def __init__(self, ParametersATC = CATCParameters):
        self.SixBarLinkage[1] = CSixBarLinkage()
        self.SixBarLinkage[2] = CSixBarLinkage()
        self.SixBarLinkage[3] = CSixBarLinkage()
        self.ParametersATC = ParametersATC()

    markersForConnectorsATC = []
    markersForPrismaticJointsATC = []

    #ToDo
    coordinatePointsATC = []
    nodesATC = []
    
#user function for friction against velocity vector, including zeroZone
#CartesianSpringDamper user function for friction
def UserFunctionSpringDamperFriction(mbs, t, itemNumber, u, v, k, d, offset):
    # print(f"[DEBUG] Friction function called for item {itemNumber} at t={t}, v={v}, offset={offset}")
    vNorm = NormL2(v)
    f=[v[0],v[1],v[2]]
    if abs(vNorm) < offset[0]:
        f = ScalarMult(offset[1]/offset[0], f)
    else:
        f = ScalarMult(offset[1]/vNorm, f)
    return f

## USER defined functions
def To3D(p): #convert numpy 2D vector to 3D list
    return [p[0],p[1],0]   

def RotZVec(angleRad,vec):
    Rz=np.array([ [np.cos(angleRad),-np.sin(angleRad)],
                      [np.sin(angleRad), np.cos(angleRad)] ])
    return np.dot(Rz,np.array(vec)); 

def centreM(vec1, vec2): #Centre M of a line AB
    return (vec1+vec2)/2

def CompliantJointsGeometricallyExactBeam2D(mbs,markerLeftBodyM, markerRightBodyM, alpha, alphaMean, phi3, phi4, phi5, P1,P2, L,width):
#%%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   
    #shape parameters
    W=6.5e-3 #width #6.5e-3*2 #14e-3            # width of rectangular ANCF element in m
    # L=7.5e-3                 # length of ANCF element in m

    exactBeam2D=True
    circularHinge=True
    cornerFilleted=False
    ellipticalHinge=False
    powerFunction=False
    h=1e-3 #1.15e-3 #*1e-1 #*1e-2
    #Circular Hinge
    if circularHinge:
        R=3.75e-3
    
    #Corner-filleted
    if cornerFilleted:
        r=3.4e-3
        
    #Elliptical Hinge
    if ellipticalHinge:
        rx=3.75e-3
        ry=3e-3
    
    #Elliptical Hinge
    if powerFunction:
        n=3

    
    
    T=8e-3                 # height of rectangular ANCF element in m   
    l=2*np.sqrt(R**2-((h/2+R)-T/2)**2)  # length of ANCF element in m

    
    # material parameters
    rho=1220                # density of ANCF element in kg/m^3   
    E=26e6 #*1.3               # Young's modulus of ANCF element in N/m^2; TPU: 26 MPa
    
    nu=0.45

    # ks = 10*(1+nu)/(12+11*nu)
    ks=1
    G = E/(2*(1+nu)) #7.9615e10           #E/(2*(1+nu))
    
    #process parameters
    f=1.2*0
    p=1.2*0
    m=-0.05*0
    
    
    numberOfElementsBeam = config.simulation.numberOfElementsBeam
    numberOfElements = config.simulation.numberOfElements

    
    M=m
    F=f
    P=p
    
    
    #background
    # rect = [0,-0.0005,L/2,0] #xmin,ymin,xmax,ymax
    # background = {'type':'Line', 'color':[0.1,0.1,0.8,1], 'data':[rect[0],rect[1],0, rect[2],rect[1],0, rect[2],rect[3],0, rect[0],rect[3],0, rect[0],rect[1],0]} #background
    # oGround=mbs.AddObject(ObjectGround(referencePosition= [0,0,0], visualization=VObjectGround(graphicsData= [background])))
    # slope of elements in reference position, calculated from first and last node position:
    cableSlopeVec = Normalize(VSub(P2, P1))
    
    
    if VSub(P2, P1)[1]<0:
        sign=-1
    else:
        sign=1    
    if exactBeam2D:
        initAngle= np.arccos(  np.dot(VSub(P2, P1),[1,0]) / np.linalg.norm(VSub(P2, P1))*np.linalg.norm([1,0]) )*sign
        nc0 = mbs.AddNode(Rigid2D(referenceCoordinates=[P1[0],P1[1],initAngle]))
    else:
        initAngle= np.arccos(  np.dot(VSub(P2, P1),[1,0]) / np.linalg.norm(VSub(P2, P1))*np.linalg.norm([1,0]) )
        nc0 = mbs.AddNode(Point2DS1(referenceCoordinates=[P1[0],P1[1],cableSlopeVec[0],cableSlopeVec[1]]))        
    
    ancfNodes1=[nc0]          
        
    ancfObjects1=[]
    lElemBeam = ((L-l)/2) / (numberOfElementsBeam)
    lElemHinge = l / numberOfElements
    
    
    lElemOld=0
    lengthDiscreteList=[]
    WbList=[]
    listT=[]

    for i in range(numberOfElements + 2*numberOfElementsBeam):
        
        if i < numberOfElementsBeam:
            lElem = lElemBeam*(i+1)
            Tb=T

        elif i >= numberOfElementsBeam and i < numberOfElementsBeam+numberOfElements:
            lElem = lElemBeam*numberOfElementsBeam+lElemHinge*(i+1-numberOfElementsBeam)
            x=np.linspace(-l/2, l/2, numberOfElements+1) 
            
            if circularHinge:
                Tvar =  (h + 2*R - 2*np.sqrt(R**2 - x**2)) #circular
            if ellipticalHinge:
                Tvar =  (h + 2*ry * (1 - np.sqrt(1 - x**2/rx**2)) ) #elliptical
            if powerFunction:
                Tvar =  (h + (T-h) / (l/2)**n * abs(x)**n )
                

            if cornerFilleted:
                Tvar = []
                for j in x:
                    if j < -l/2+r:
                        Tvar += [h + 2*r - 2* np.sqrt(r**2- (j+l/2-r)**2 )]
                    elif j <= l/2-r:
                        Tvar += [h]
                    else:
                        Tvar += [h + 2*r - 2* np.sqrt(r**2- (j-l/2+r)**2 )]
                
                Tvar=np.array(Tvar)

                
            Tb = ( Tvar[i-numberOfElementsBeam] + Tvar[i-numberOfElementsBeam+1] ) / 2
            
        else:
            lElem = lElemBeam*numberOfElementsBeam + lElemHinge*numberOfElements + lElemBeam*(i+1-numberOfElementsBeam-numberOfElements)
            Tb=T
        
        if Tb > T:
            Tb = T        
        
        A=W*Tb                  # cross sectional area of ANCF element in m^2
        I=W*Tb**3/12               
        GA=ks*G*A
        # length of one element, calculated from first and last node position:
        # cableLength = NormL2(VSub(positionOfNode1, positionOfNode0))/numberOfElements

        if exactBeam2D:
            nLast = mbs.AddNode(Rigid2D(referenceCoordinates=[P1[0]+lElem*cableSlopeVec[0],P1[1]+lElem*cableSlopeVec[1],initAngle], 
                                        initialCoordinates = [0,0,0], 
                                        initialVelocities= [0,0,0]))
            
            nObject = mbs.AddObject(ObjectBeamGeometricallyExact2D(nodeNumbers=[int(nc0)+i,int(nc0)+i+1],
                                                                   physicsLength=lElem-lElemOld, 
                                                                   physicsMassPerLength=rho*A, 
                                                                   physicsBendingStiffness=E*I, 
                                                                   physicsAxialStiffness=E*A, 
                                                                   physicsShearStiffness=GA,
                                                                   physicsBendingDamping=E*I*0.05,
                                                                   physicsAxialDamping=E*A*0.1,
                                                                   physicsShearDamping=GA*0.1,                                                                     
                                                                   visualization=VCable2D(drawHeight=Tb)))            
        else:  
            nLast = mbs.AddNode(Point2DS1(referenceCoordinates=[P1[0]+lElem*cableSlopeVec[0],P1[1]+lElem*cableSlopeVec[1],cableSlopeVec[0],cableSlopeVec[1]]))
            nObject = mbs.AddObject(Cable2D(physicsLength=lElem-lElemOld, physicsMassPerLength=rho*A, 
                                  physicsBendingStiffness=E*I, physicsAxialStiffness=E*A, nodeNumbers=[int(nc0)+i,int(nc0)+i+1],
                                  visualization=VCable2D(drawHeight=Tb)))
        
        
        ancfNodes1+=[nLast]
        ancfObjects1+=[nObject]
        
        lElemOld=lElem
        lengthDiscreteList+=[lElem]
        WbList+=[W*Tb**2/6]  
        listT+=[Tb]
         
    
    # markerBodyCableL = mbs.AddMarker(MarkerBodyRigid(bodyNumber=ancfObjects1[0], localPosition=[0,0,0]))
    # markerBodyCableR = mbs.AddMarker(MarkerBodyRigid(bodyNumber=ancfObjects1[-1], localPosition=[lElemBeam,0,0]))   

    if exactBeam2D:
        markerBodyCableL = mbs.AddMarker(MarkerBodyRigid(bodyNumber=ancfObjects1[0], localPosition=[-lElemBeam/2,0,0]))
        markerBodyCableR = mbs.AddMarker(MarkerBodyRigid(bodyNumber=ancfObjects1[-1], localPosition=[lElemBeam/2,0,0]))   
    else:
        markerBodyCableL = mbs.AddMarker(MarkerBodyRigid(bodyNumber=ancfObjects1[0], localPosition=[0,0,0]))
        markerBodyCableR = mbs.AddMarker(MarkerBodyRigid(bodyNumber=ancfObjects1[-1], localPosition=[lElemBeam,0,0]))      




    #add mass left side
    a = -0.001     #x-dim of pendulum
    b = T/2 #L/2+0.005    #y-dim of pendulum
    graphics2 = {'type':'Line', 'color':[0.1,0.1,0.8,1], 'data':[-a*0,-b,0, a,-b,0, a,b,0, -a*0,b,0, -a*0,-b,0]} #background

    massRigid = config.lengths.weightBeams
    inertiaRigid = config.lengths.inertiaBeams  
    # massRigid = 1e-3
    # inertiaRigid = massRigid/12*(2*a)**2
    nRigidL = mbs.AddNode(Rigid2D(referenceCoordinates=[P1[0],P1[1],alpha], initialVelocities=[0,0,0]));
    oRigid = mbs.AddObject(RigidBody2D(physicsMass=massRigid, physicsInertia=inertiaRigid,nodeNumber=nRigidL,visualization=VObjectRigidBody2D(graphicsData= [graphics2])))
    
    markerLeftBodyMassLeft = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid, localPosition=[0,0,0]))

    
    #add mass right side
    a = 0.001     #x-dim of pendulum
    b =T/2 #L/2+0.001    #y-dim of pendulum
    graphics2 = {'type':'Line', 'color':[0.1,0.1,0.8,1], 'data':[-a*0,-b,0, a,-b,0, a,b,0, -a*0,b,0, -a*0,-b,0]} #background
    massRigid = config.lengths.weightBeams
    inertiaRigid = config.lengths.inertiaBeams    
    # massRigid = 1e-3
    # inertiaRigid = massRigid/12*(2*a)**2

    
    nRigidR = mbs.AddNode(Rigid2D(referenceCoordinates=[P2[0],P2[1],alpha], initialVelocities=[0,0,0]));
    oRigid = mbs.AddObject(RigidBody2D(physicsMass=massRigid, physicsInertia=inertiaRigid,nodeNumber=nRigidR,visualization=VObjectRigidBody2D(graphicsData= [graphics2])))
    
    markerRightBodyT = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid, localPosition=[0,b,0]))
    markerRightBodyB = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid, localPosition=[0,-b,0]))
    markerRightBodyMassRight = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid, localPosition=[0,0,0]))
    
    nForceF=mbs.AddLoad(Force(markerNumber = markerRightBodyMassRight, loadVector=[P*np.cos(alpha),P*np.sin(alpha),0]))
    nForceP=mbs.AddLoad(Force(markerNumber = markerRightBodyMassRight, loadVector=[-F*np.sin(alpha),F*np.cos(alpha),0]))
    nTorque=mbs.AddLoad(Torque(markerNumber = markerRightBodyMassRight, loadVector=[0,0,-M]))
    


    
    #connect cableLeft with given Marker Left
    initialRotation1 = RotationMatrixZ(phi4)
    initialRotation2 = RotationMatrixZ(alpha+alphaMean)
    
    mbs.AddObject(GenericJoint(markerNumbers=[markerLeftBodyMassLeft, markerLeftBodyM], 
                                rotationMarker0=initialRotation1,
                                rotationMarker1=initialRotation2,
                                constrainedAxes=[1,1,0,0,0,1],
                                visualization=VObjectJointGeneric(axesRadius=0.5e-4, axesLength=0.5e-3)))
    
    
    #connect cableRight with given Marker Right
    initialRotation1 = RotationMatrixZ(phi5)
    initialRotation2 = RotationMatrixZ(phi3)    
    mbs.AddObject(GenericJoint(markerNumbers=[markerRightBodyMassRight, markerRightBodyM], 
                                rotationMarker0=initialRotation1,
                                rotationMarker1=initialRotation2,
                                constrainedAxes=[1,1,0,0,0,1],
                                visualization=VObjectJointGeneric(axesRadius=0.5e-4, axesLength=0.5e-3)))
  


    if exactBeam2D:
        mC00 = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nRigidL, coordinate=0))
        mC11 = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nRigidL, coordinate=1))
        mC22 = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nRigidL, coordinate=2))   
    
        n0 = ancfNodes1[0]
        mC0 = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=n0, coordinate=0))
        mC1 = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=n0, coordinate=1))
        mC2 = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=n0, coordinate=2))
        mbs.AddObject(CoordinateConstraint(markerNumbers=[mC00, mC0]))
        mbs.AddObject(CoordinateConstraint(markerNumbers=[mC11, mC1]))
        mbs.AddObject(CoordinateConstraint(markerNumbers=[mC22, mC2]))         
        

        mC00 = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nRigidR, coordinate=0))
        mC11 = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nRigidR, coordinate=1))
        mC22 = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nRigidR, coordinate=2))     
        
        n0 = ancfNodes1[-1]
        mC0 = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=n0, coordinate=0))
        mC1 = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=n0, coordinate=1))
        mC2 = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=n0, coordinate=2))
        mbs.AddObject(CoordinateConstraint(markerNumbers=[mC00, mC0]))
        mbs.AddObject(CoordinateConstraint(markerNumbers=[mC11, mC1]))
        mbs.AddObject(CoordinateConstraint(markerNumbers=[mC22, mC2]))          
        
        
        
    else:
        #connect mass right side with cableRight
        initialRotation1 = RotationMatrixZ(0)
        initialRotation2 = RotationMatrixZ(0)    
        mbs.AddObject(GenericJoint(markerNumbers=[markerBodyCableR, markerRightBodyMassRight], 
                                    rotationMarker0=initialRotation1,
                                    rotationMarker1=initialRotation2,
                                    constrainedAxes=[1,1,0,0,0,1],
                                    visualization=VObjectJointGeneric(axesRadius=0.5e-4, axesLength=0.5e-3)))        
        
        #connect cableLeft with given Marker Left
        initialRotation1 = RotationMatrixZ(phi4)
        initialRotation2 = RotationMatrixZ(alpha+alphaMean)
        
        mbs.AddObject(GenericJoint(markerNumbers=[markerBodyCableL, markerLeftBodyM], 
                                    rotationMarker0=initialRotation1,
                                    rotationMarker1=initialRotation2,
                                    constrainedAxes=[1,1,0,0,0,1],
                                    visualization=VObjectJointGeneric(axesRadius=0.5e-3, axesLength=0.5e-2)))
    
    
    # #add sensores
    # s1 = mbs.AddSensor(SensorMarker(markerNumber=markerRightBodyT, #fileName=sensorFileNameS1, 
    #                             outputVariableType=exu.OutputVariableType.Position))    
    # s2 = mbs.AddSensor(SensorMarker(markerNumber=markerRightBodyB, #fileName=sensorFileNameS1, 
    #                             outputVariableType=exu.OutputVariableType.Position)) 
        
    
    # sensorLowerBeam=[]
    # for i in range(int(len(ancfObjects1))):
    #     # midNode = ancfObjects1[int(len(ancfObjects1))-1]
    #     midNode = ancfObjects1[i]
    #     sensorLowerBeam += [mbs.AddSensor(SensorBody(bodyNumber=midNode, #fileName=sensorFileNameS1, 
    #                             outputVariableType=exu.OutputVariableType.TorqueLocal))]
        
    
    
    # mF2 = mbs.AddMarker(MarkerBodyPosition(bodyNumber=oRigid, localPosition=[-L/2,0,0])) #end point
    # sM = mbs.AddSensor(SensorMarker(markerNumber=mF2, #fileName=sensorFileNameMarker, 
    #                             outputVariableType=exu.OutputVariableType.Position))

#build parameterized model for parameter variation
def CrossSpringPivot(mbs, markerLeftBodyM, markerRightBodyM, alpha, alphaMean, phi3, phi4, phi5, P1,P2, L, width):
    alpha2=alpha
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Fürhapter, BA 2021
    #shape parameters
    # L=9.45e-3                # length of ANCF element in m
    
    L0=NormL2(VSub(P2, P1))
    
    T=0.4049e-3 #0.68e-3      # height of rectangular ANCF element in m
    W=3e-3*2 #width/2 #3e-3*2                # width of rectangular ANCF element in m (*2 symmetric)
    
    # material parameters
    rho=1240               # density of ANCF element in kg/m^3   
    E=2.3465e9             # Young's modulus of ANCF element in N/m^2
    
    f=0
    p=6.5821*0
    m=0.05*0
    

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    A=W*T                  # cross sectional area of ANCF element in m^2
    I=W*T**3/12            # second moment of area of ANCF element in m^4   
    
           
    numberOfElements = config.simulation.numberOfElements     
    Lambda=0.5     
    
    
    alpha=39.8955*np.pi/180  #42*np.pi/180   
    # alpha=60*np.pi/180 
    
    L=L0/np.cos(alpha)
    # print(L)
    M=m*(E*I)/L
    F=f*(E*I)/(L**2)
    P=p*(E*I)/(L**2)
 
    # #background
    # rect = [0,-0.0005,Lambda*L*np.cos(alpha),0] #xmin,ymin,xmax,ymax
    # background = {'type':'Line', 'color':[0.1,0.1,0.8,1], 'data':[rect[0],rect[1],0, rect[2],rect[1],0, rect[2],rect[3],0, rect[0],rect[3],0, rect[0],rect[1],0]} #background
    # oGround=mbs.AddObject(ObjectGround(referencePosition= [P1[0],P1[1],0], visualization=VObjectGround(graphicsData= [background])))
    
    # nGround = mbs.AddNode(NodePointGround(referenceCoordinates=[0,0,0])) #ground node for coordinate constraint
    # mGround = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber = nGround, coordinate=0)) #Ground node ==> no action
    
    # mGround0 = mbs.AddMarker(MarkerBodyRigid(bodyNumber = oGround, localPosition = [0,0,0]))
    
    a = -0.0005     #x-dim of pendulum
    b =(1-Lambda)*L*np.sin(alpha)+0.001    #y-dim of pendulum
    graphics2 = {'type':'Line', 'color':[0.1,0.1,0.8,1], 'data':[-a*0,-b,0, a,-b,0, a,b,0, -a*0,b,0, -a*0,-b,0]} #background
    # massRigid = 1e-3
    # inertiaRigid = massRigid/12*(2*a)**2
    
    massRigid = config.lengths.weightBeams
    inertiaRigid = config.lengths.inertiaBeams  
    
    nRigid = mbs.AddNode(Rigid2D(referenceCoordinates=[P1[0],P1[1],alpha2], initialVelocities=[0,0,0]));
    oRigid = mbs.AddObject(RigidBody2D(physicsMass=massRigid, physicsInertia=inertiaRigid,nodeNumber=nRigid,visualization=VObjectRigidBody2D(graphicsData= [graphics2])))
    
    
    markerBodyLL = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid, localPosition=[0,-(Lambda)*L*np.sin(alpha),0]))
    markerBodyLU = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid, localPosition=[0,(Lambda)*L*np.sin(alpha),0]))
    markerBodyM = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid, localPosition=[0,0,0]))
    

    # initialRotation1 = RotationMatrixZ(alpha2+alphaMean)
    # mbs.AddObject(GenericJoint(markerNumbers=[markerLeftBodyM, markerBodyM], 
    #                             rotationMarker0=initialRotation1,
    #                             # rotationMarker1=initialRotation1,
    #                             constrainedAxes=[1,1,0,0,0,1],
    #                             visualization=VObjectJointGeneric(axesRadius=0.5e-4, axesLength=0.5e-3)))
    
    
    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #generate ANCF beams with utilities function    
    cableTemplate = Cable2D(#physicsLength = L / numberOfElements, #set in GenerateStraightLineANCFCable2D(...)
                            physicsMassPerLength = rho*A,
                            physicsBendingStiffness = E*I,
                            physicsAxialStiffness = E*A,
                            visualization=VCable2D(drawHeight=T)
                            #nodeNumbers = [0, 0], #will be filled in GenerateStraightLineANCFCable2D(...)
                            )
    
    R1 = RotationMatrixZ(alpha2)
    pLowerL=[0, -Lambda*L*np.sin(alpha), 0]
    # pUpperL=[L*np.cos(alpha), (1-Lambda)*L*np.sin(alpha+theta), 0]
    pUpperL=[0, +Lambda*L*np.sin(alpha), 0]
    pLeftG=[P1[0],P1[1],0]
    pRightG=[P2[0],P2[1],0]
    
    positionOfNode0 = R1@pLowerL+pLeftG # starting point of line
    positionOfNode1 = R1@pUpperL+pRightG # end point of line
    #alternative to mbs.AddObject(Cable2D(...)) with nodes:
    ancf=GenerateStraightLineANCFCable2D(mbs,
                    positionOfNode0, positionOfNode1,
                    numberOfElements,
                    cableTemplate, #this defines the beam element properties
                    massProportionalLoad = [0,0,0], #optionally add gravity
                    fixedConstraintsNode0 = [0,0,0,0], #add constraints for pos and rot (r'_y)
                    fixedConstraintsNode1 = [0,0,0,0])
    ancfNodes1 = ancf[0]
    ancfObjects1 = ancf[1]
    
    #generate ANCF beams with utilities function
    cableTemplate = Cable2D(#physicsLength = L / nElements, #set in GenerateStraightLineANCFCable2D(...)
                            physicsMassPerLength = rho*A,
                            physicsBendingStiffness = E*I,
                            physicsAxialStiffness = E*A,
                            visualization=VCable2D(drawHeight=T)
                            #nodeNumbers = [0, 0], #will be filled in GenerateStraightLineANCFCable2D(...)
                            )
    
    positionOfNode0 = R1@pUpperL+pLeftG # starting point of line
    positionOfNode1 = R1@pLowerL+pRightG # end point of line
    #alternative to mbs.AddObject(Cable2D(...)) with nodes:
    ancf=GenerateStraightLineANCFCable2D(mbs,
                    positionOfNode0, positionOfNode1,
                    numberOfElements,
                    cableTemplate, #this defines the beam element properties
                    massProportionalLoad = [0,0,0], #optionally add gravity
                    fixedConstraintsNode0 = [0,0,0,0], #add constraints for pos and rot (r'_y)
                    fixedConstraintsNode1 = [0,0,0,0])
    ancfNodes2 = ancf[0]
    ancfObjects2 = ancf[1]

    markerBodycLL = mbs.AddMarker(MarkerBodyRigid(bodyNumber=ancfObjects1[0], localPosition=[0,0,0]))
    markerBodycLU = mbs.AddMarker(MarkerBodyRigid(bodyNumber=ancfObjects2[0], localPosition=[0,0,0]))    
    
    markerBodycU = mbs.AddMarker(MarkerBodyRigid(bodyNumber=ancfObjects1[-1], localPosition=[L/numberOfElements,0,0]))
    markerBodycL = mbs.AddMarker(MarkerBodyRigid(bodyNumber=ancfObjects2[-1], localPosition=[L/numberOfElements,0,0]))
    

    a = 0.0005     #x-dim of pendulum
    b =(1-Lambda)*L*np.sin(alpha)+0.001    #y-dim of pendulum
    graphics2 = {'type':'Line', 'color':[0.1,0.1,0.8,1], 'data':[-a*0,-b,0, a,-b,0, a,b,0, -a*0,b,0, -a*0,-b,0]} #background
    # massRigid = 1e-3
    # inertiaRigid = massRigid/12*(2*a)**2

    massRigid = config.lengths.weightBeams
    inertiaRigid = config.lengths.inertiaBeams  
    
    nRigid = mbs.AddNode(Rigid2D(referenceCoordinates=[P2[0],P2[1],alpha2], initialVelocities=[0,0,0]));
    oRigid = mbs.AddObject(RigidBody2D(physicsMass=massRigid, physicsInertia=inertiaRigid,nodeNumber=nRigid,visualization=VObjectRigidBody2D(graphicsData= [graphics2])))
    
    
    markerBodyL = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid, localPosition=[0,-(1-Lambda)*L*np.sin(alpha),0]))
    markerBodyU = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid, localPosition=[0,(1-Lambda)*L*np.sin(alpha),0]))
    
    
    markerBodyF = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid, localPosition=[0,0,0]))

    #connect CSP to markerRightBodyM
    # initialRotation1 = RotationMatrixZ(0)
    # mbs.AddObject(GenericJoint(markerNumbers=[markerRightBodyM, markerBodyF], 
    #                             # rotationMarker0=initialRotation1,
    #                             rotationMarker1=initialRotation1,
    #                             constrainedAxes=[1,1,0,0,0,1],
    #                             visualization=VObjectJointGeneric(axesRadius=0.5e-4, axesLength=0.5e-3)))


    nForceF=mbs.AddLoad(Force(markerNumber = markerBodyF, loadVector=[P,0,0]))
    nForceP=mbs.AddLoad(Force(markerNumber = markerBodyF, loadVector=[0,F,0]))
    nTorque=mbs.AddLoad(Torque(markerNumber = markerBodyF, loadVector=[0,0,-M]))
    

    initialRotation1 = RotationMatrixZ(-alpha )
    initialRotation2 = RotationMatrixZ( alpha )
    

    mbs.AddObject(GenericJoint(markerNumbers=[markerBodycL, markerBodyL], 
                                # rotationMarker0=initialRotation2
                                rotationMarker1=initialRotation1,
                                constrainedAxes=[1,1,0,0,0,1],
                                visualization=VObjectJointGeneric(axesRadius=0.5e-4, axesLength=0.5e-3)))
    mbs.AddObject(GenericJoint(markerNumbers=[markerBodycU, markerBodyU], 
                                # rotationMarker0=initialRotation1,
                                rotationMarker1=initialRotation2,
                                constrainedAxes=[1,1,0,0,0,1],
                                visualization=VObjectJointGeneric(axesRadius=0.5e-4, axesLength=0.5e-3)))
    
    
    mbs.AddObject(GenericJoint(markerNumbers=[markerBodycLL, markerBodyLL], 
                                # rotationMarker0=initialRotation2,
                                rotationMarker1=initialRotation2,
                                constrainedAxes=[1,1,0,0,0,1],
                                visualization=VObjectJointGeneric(axesRadius=0.5e-4, axesLength=0.5e-3)))
    mbs.AddObject(GenericJoint(markerNumbers=[markerBodycLU, markerBodyLU], 
                                # rotationMarker0=initialRotation1,
                                rotationMarker1=initialRotation1,
                                constrainedAxes=[1,1,0,0,0,1],
                                visualization=VObjectJointGeneric(axesRadius=0.5e-4, axesLength=0.5e-3)))


    # # initialRotation2 = RotationMatrixZ(phi3)   
    # initialRotation1 = RotationMatrixZ(phi3)
    # mbs.AddObject(GenericJoint(markerNumbers=[markerRightBodyM, markerBodyF], 
    #                             rotationMarker0=initialRotation1,
    #                             # rotationMarker1=initialRotation1,
    #                             constrainedAxes=[1,1,0,0,0,1],
    #                             visualization=VObjectJointGeneric(axesRadius=0.5e-4, axesLength=0.5e-3)))
    #connect cableLeft with given Marker Left
    initialRotation1 = RotationMatrixZ(phi4)
    initialRotation2 = RotationMatrixZ(alpha2+alphaMean)
    
    mbs.AddObject(GenericJoint(markerNumbers=[markerBodyM, markerLeftBodyM], 
                                rotationMarker0=initialRotation1,
                                rotationMarker1=initialRotation2,
                                constrainedAxes=[1,1,0,0,0,1],
                                visualization=VObjectJointGeneric(axesRadius=0.5e-4, axesLength=0.5e-3)))
    
    
    #connect cableRight with given Marker Right
    initialRotation1 = RotationMatrixZ(phi5)
    initialRotation2 = RotationMatrixZ(phi3)    
    mbs.AddObject(GenericJoint(markerNumbers=[markerBodyF, markerRightBodyM], 
                                rotationMarker0=initialRotation1,
                                rotationMarker1=initialRotation2,
                                constrainedAxes=[1,1,0,0,0,1],
                                visualization=VObjectJointGeneric(axesRadius=0.5e-4, axesLength=0.5e-3)))

def calculateVisualization(cSixBarLinkage,localP,mode,sideNumber,ID,nodeNumber):
    CE=cSixBarLinkage.SixBarLinkageParameters.CE
    L2=cSixBarLinkage.SixBarLinkageParameters.L2
    psi=cSixBarLinkage.SixBarLinkageParameters.psi
    alphaMean=cSixBarLinkage.startAngle
    theta=cSixBarLinkage.orientation
    ATCID=[]
    actuatorSideNumber=[]
    nodeNumberGraphics=[]
    
    if config.simulation.showLabeling:
        actuatorSideNumber = GraphicsDataText (point= [0.05,0,0], text= str(sideNumber), color= [0.,0.,0.,1.])
        nodeNumberGraphics = GraphicsDataText (point= [-0.06,0.008,0], text= str(nodeNumber), color= [0.,0.,0.,1.])
    
        if ID>0:
            ATCID = GraphicsDataText (point= [0.05,-0.05,0], text= str(ID), color= [0.,0.,0.,1.])

    #Draw++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if mode=='graphicsSimplifed':
        bLength = cSixBarLinkage.SixBarLinkageVisualization.bLength   #y-dim of pendulum    
        
        graphicsCE = {'type':'Line', 'color':[0.1,0.1,0.8,1], 'data':[-CE/2,-bLength,0, CE/2,-bLength,0, CE/2,bLength,0, -CE/2,bLength,0, -CE/2,-bLength,0]}
        graphicsL2 = {'type':'Line', 'color':[0.1,0.1,0.8,1], 'data':[-L2/2,-bLength,0, L2/2,-bLength,0, L2/2,bLength,0, -L2/2,bLength,0, -L2/2,-bLength,0]}
    
        ad=list(RotZVec(0,[-CE,-bLength]))
        bd=list(RotZVec(0,[0,-bLength]))
        cd=list(RotZVec(0,[0,bLength]))
        dd=list(RotZVec(0,[-CE,bLength]))
        ed=list(RotZVec(0,[-CE,-bLength]))    
        graphicsCE1 = {'type':'Line', 'color':[0.1,0.1,0.8,1], 'data':ad+[0]+ bd+[0]+ cd+[0]+ dd+[0]+ ed+[0]} #background
        
        ad=list(RotZVec(0,[0,-bLength]))
        bd=list(RotZVec(0,[L2*np.cos(psi),L2*np.sin(psi)-bLength]))
        cd=list(RotZVec(0,[L2*np.cos(psi),L2*np.sin(psi)+bLength]))
        dd=list(RotZVec(0,[0,bLength]))
        ed=list(RotZVec(0,[0,-bLength]))  
        graphicsL21 = {'type':'Line', 'color':[0.1,0.1,0.8,1], 'data':ad+[0]+ bd+[0]+ cd+[0]+ dd+[0]+ ed+[0]} #background
        
        ad=list(RotZVec(0,[-CE*np.cos(psi),-CE*np.sin(psi)-bLength]))
        bd=list(RotZVec(0,[0,-bLength]))
        cd=list(RotZVec(0,[0,bLength]))
        dd=list(RotZVec(0,[-CE*np.cos(psi),-CE*np.sin(psi)+bLength]))
        ed=list(RotZVec(0,[-CE*np.cos(psi),-CE*np.sin(psi)-bLength]))    
        graphicsCE2 = {'type':'Line', 'color':[0.1,0.1,0.8,1],'data':ad+[0]+ bd+[0]+ cd+[0]+ dd+[0]+ ed+[0]}
    
        ad=list(RotZVec(0,[0,-bLength]))
        bd=list(RotZVec(0,[L2,-bLength]))
        cd=list(RotZVec(0,[L2,bLength]))
        dd=list(RotZVec(0,[0,bLength]))
        ed=list(RotZVec(0,[0,-bLength]))  
        graphicsL22 = {'type':'Line', 'color':[0.1,0.1,0.8,1], 'data':ad+[0]+ bd+[0]+ cd+[0]+ dd+[0]+ ed+[0]} #background 
    
        cSixBarLinkage.SixBarLinkageVisualization.graphicsL2ground = [graphicsL2]
        cSixBarLinkage.SixBarLinkageVisualization.graphicsCE1R=[graphicsCE1,graphicsL21]
        cSixBarLinkage.SixBarLinkageVisualization.graphicsCER=[graphicsCE]
        cSixBarLinkage.SixBarLinkageVisualization.graphicsCEL=[graphicsCE]
        cSixBarLinkage.SixBarLinkageVisualization.graphicsCE2L=[graphicsCE2,graphicsL22]
        cSixBarLinkage.SixBarLinkageVisualization.graphicsL2 = [nodeNumberGraphics,actuatorSideNumber,ATCID,graphicsL2]


    if mode=='graphicsSTL':
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #graphics Transformations: stl-files with 66.25° starting position
        #graphicsCER = GraphicsDataFromSTLfileTxt(fileName='./data/stlFiles/graphicsCER.stl',color=[0.1,0.1,0.8,1],verbose=False) 
        #graphicsCEL = GraphicsDataFromSTLfileTxt(fileName='./data/stlFiles/graphicsCEL.stl',color=[0.1,0.1,0.8,1],verbose=False) 
        #graphicsL2 = GraphicsDataFromSTLfileTxt(fileName='./data/stlFiles/graphicsL2.stl',color=[0.1,0.1,0.8,1],verbose=False) 
        #graphicsL2ground = GraphicsDataFromSTLfileTxt(fileName='./data/stlFiles/graphicsL2ground.stl',color=[0.1,0.1,0.8,1],verbose=False) 
        #graphicsCE1R = GraphicsDataFromSTLfileTxt(fileName='./data/stlFiles/graphicsCE1R.stl',color=[0.1,0.1,0.8,1],verbose=False) 
        #graphicsCE2L = GraphicsDataFromSTLfileTxt(fileName='./data/stlFiles/graphicsCE2L.stl',color=[0.1,0.1,0.8,1],verbose=False) 
        graphicsCER = GraphicsDataFromSTLfileTxt(fileName='/Users/fynnheidenreich/Desktop/Uni/6/bachelor/HeidenreichFynn/programPARTS24/data/stlFiles/graphicsCER.stl',color=[0.1,0.1,0.8,1],verbose=False) 
        graphicsCEL = GraphicsDataFromSTLfileTxt(fileName='/Users/fynnheidenreich/Desktop/Uni/6/bachelor/HeidenreichFynn/programPARTS24/data/stlFiles/graphicsCEL.stl',color=[0.1,0.1,0.8,1],verbose=False) 
        graphicsL2 = GraphicsDataFromSTLfileTxt(fileName='/Users/fynnheidenreich/Desktop/Uni/6/bachelor/HeidenreichFynn/programPARTS24/data/stlFiles/graphicsL2.stl',color=[0.1,0.1,0.8,1],verbose=False) 
        graphicsL2ground = GraphicsDataFromSTLfileTxt(fileName='/Users/fynnheidenreich/Desktop/Uni/6/bachelor/HeidenreichFynn/programPARTS24/data/stlFiles/graphicsL2ground.stl',color=[0.1,0.1,0.8,1],verbose=False) 
        graphicsCE1R = GraphicsDataFromSTLfileTxt(fileName='/Users/fynnheidenreich/Desktop/Uni/6/bachelor/HeidenreichFynn/programPARTS24/data/stlFiles/graphicsCE1R.stl',color=[0.1,0.1,0.8,1],verbose=False) 
        graphicsCE2L = GraphicsDataFromSTLfileTxt(fileName='/Users/fynnheidenreich/Desktop/Uni/6/bachelor/HeidenreichFynn/programPARTS24/data/stlFiles/graphicsCE2L.stl',color=[0.1,0.1,0.8,1],verbose=False) 

        M0Local=list((np.array(localP[4])-np.array(localP[3]))/2+np.array(localP[3]))
        M2Local=list((np.array(localP[6])-np.array(localP[4]))/2+np.array(localP[4]))
        M5Local=list((np.array(localP[7])-np.array(localP[9]))/2+np.array(localP[9]))
        M7Local=list((np.array(localP[9])-np.array(localP[8]))/2+np.array(localP[8]))

        
        alphaMeanSTL = 66.25*np.pi/180
        P9LocalSTL=[CE*np.cos(alphaMeanSTL-psi)+L2*np.cos(alphaMeanSTL),CE*np.sin(alphaMeanSTL-psi)+L2*np.sin(alphaMeanSTL)]
        P8LocalSTL=[CE*np.cos(alphaMeanSTL-psi),CE*np.sin(alphaMeanSTL-psi)]
        M7LocalSTL=list((np.array(P9LocalSTL)-np.array(P8LocalSTL))/2+np.array(P8LocalSTL))
    
    
        A=RotationMatrixZ(theta)
        p=np.array(list((np.array(M0Local)-np.array(localP[1]) ))+[0])
        graphicsL2ground = MoveGraphicsData(graphicsL2ground,-A.T@p,A.T)
    
        A=RotationMatrixZ(-theta)
        p=np.array(list((np.array(localP[1])-np.array(localP[1]) ))+[0.00325])
        graphicsL2ground = [MoveGraphicsData(graphicsL2ground,-A.T@p,A.T)]
    
    
        
        ## graphicsCE1R
        A=RotationMatrixZ(alphaMeanSTL-alphaMean)
        p1=np.array(list(   (np.array(localP[3])-np.array(localP[1]))  )  +[0.0])
        p2=np.array(list(   (np.array(localP[3])-np.array(localP[1]))  )  +[0.0])
        graphicsCE1R = MoveGraphicsData(graphicsCE1R, p2-A.T@p1,A.T)
        
        A=RotationMatrixZ((alphaMean-psi))
        p=np.array(list(np.array(localP[5])-np.array(localP[1]))+[0.00325])
        graphicsCE1R = [MoveGraphicsData(graphicsCE1R, -A.T@p,A.T)]
           
        
        ## graphicsCER
        A=RotationMatrixZ((alphaMeanSTL-alphaMean))
        p1=np.array(list(   (np.array(localP[4])-np.array(localP[1]))  )  +[0.0])
        p2=np.array(list(   (np.array(localP[4])-np.array(localP[1]))  )  +[0.0])
        graphicsCER = MoveGraphicsData(graphicsCER, p2-A.T@p1,A.T)
        
         
        A=RotationMatrixZ((alphaMean-psi))
        p=np.array(list( (np.array(M2Local)- np.array(localP[1])) )+[0.00325])
        graphicsCER = [MoveGraphicsData(graphicsCER, -A.T@p,A.T)]
    
    
        ## graphicsCEL    
        A=RotationMatrixZ( 0 )
        p2=np.array(list(   (np.array(P9LocalSTL)-np.array(localP[9]))  )  +[0.0])
        graphicsCEL = MoveGraphicsData(graphicsCEL, -p2,A.T)
        
        A=RotationMatrixZ(psi)
        p=np.array(list(np.array(M5Local)-np.array(localP[1]))+[0.00325])
        graphicsCEL = [MoveGraphicsData(graphicsCEL, -A.T@p,A.T)]
    
        ## graphicsCE2L     
        A=RotationMatrixZ( 0 )
        p2=np.array(list(   (np.array(P8LocalSTL)-np.array(localP[8]))  )  +[0.0])
        graphicsCE2L = MoveGraphicsData(graphicsCE2L, -p2,A.T)
        
        A=RotationMatrixZ(0)
        p=np.array(list(np.array(localP[5])-np.array(localP[1]))+[0.00325])
        graphicsCE2L = [MoveGraphicsData(graphicsCE2L, -A.T@p,A.T)]
    
        ## graphicsL2    
        A=RotationMatrixZ( (alphaMeanSTL-(alphaMean)) )
        p1=np.array(list(   (np.array(P8LocalSTL)-np.array(localP[1]))  )  +[0.0])
        p2=np.array(list(   (np.array(P8LocalSTL)-np.array(localP[1]))  )  +[0.0])
        graphicsL2 = MoveGraphicsData(graphicsL2, p2-A.T@p1,A.T)
        
        A=RotationMatrixZ( (alphaMeanSTL-(alphaMean))*0 )
        p1=np.array(list(   (np.array(P8LocalSTL)-np.array(localP[8]))  )  +[0.0])
        graphicsL2 = MoveGraphicsData(graphicsL2, -p1,A.T)
        
        A=RotationMatrixZ(alphaMean)
        p=np.array(list(np.array(M7Local)-np.array(localP[1]))+[0.00325])
        graphicsL2 = [MoveGraphicsData(graphicsL2, -A.T@p,A.T)]   
    
        #cSixBarLinkage.SixBarLinkageVisualization.graphicsL2ground = graphicsL2ground
        #cSixBarLinkage.SixBarLinkageVisualization.graphicsCE1R=graphicsCE1R
        #cSixBarLinkage.SixBarLinkageVisualization.graphicsCER=graphicsCER
        #cSixBarLinkage.SixBarLinkageVisualization.graphicsCEL=graphicsCEL
        cSixBarLinkage.SixBarLinkageVisualization.graphicsCE2L=graphicsCE2L
        #cSixBarLinkage.SixBarLinkageVisualization.graphicsL2 = graphicsL2
        
        P58Local=list(np.array(localP[8])-np.array(localP[5]))
        P56Local=list(np.array(localP[6])-np.array(localP[5]))
        #diemensions for links
        w=0.005
        zAxis=1.2*w/2
        
        #dimensions for axis
        rAxis=0.0005
        dZAxis=0.015
        dZAxisSmall=0.003
        #dimensions for actuatorHousing
        actorL=L2+0.005
        actorB=0.015
        actorH=0.01
        actorR=0.004
    
        #dimensions for actuatorHousing
        actorHousingL=0.105
        actorHousingB=0.01
        actorHousingH=0.01
        actorStroke=0.12
    
        #dimensions for connector
        actorConnectorL=0.01
        actorConnectorB=0.005
        actorConnectorH=0.005
    
        #graphicsActorConnectorFemale
        actorConnectorFemaleL=0.005
        actorConnectorFemaleB=0.01
        actorConnectorFemaleH=0.005
    
        #graphicsActorConnectorFemale
        actorConnectorMaleL=0.005
        actorConnectorMaleB=0.01
        actorConnectorMaleH=0.005        
        actorConnectorMaleStroke=0.006  
        actorConnectorMaleRadius=0.002 

        # actor
        graphicsActor = GraphicsDataOrthoCubePoint([0,0]+[-actorH/2-2*zAxis], [actorL,actorB,actorH], color4blue,)    
        graphicsActorHousing = GraphicsDataOrthoCubePoint([actorHousingL/2,0]+[-actorH/2-2*zAxis], [actorHousingL,actorHousingB,actorHousingH], color4blue,)
        graphicsActorStroke = GraphicsDataCylinder(pAxis=[L2/2,0]+[-actorH/2-2*zAxis], vAxis=[actorStroke,0]+[0], radius=actorR, color=color4grey, nTiles=16)   
    
        graphicsActorConnector = GraphicsDataOrthoCubePoint([-L2/2-actorConnectorL/2,0]+[-actorH/2-2*zAxis], [actorConnectorL,actorConnectorB,actorConnectorH], color4violet)
        graphicsActorConnectorFemale = GraphicsDataOrthoCubePoint([-L2/2-actorConnectorL,-actorConnectorFemaleB/4]+[-actorH/2-2*zAxis], [actorConnectorFemaleL,actorConnectorFemaleB,actorConnectorFemaleH], color4violet)
        graphicsActorConnectorMale = GraphicsDataOrthoCubePoint([-L2/2-actorConnectorL,actorConnectorMaleB/4]+[-actorH/2-2*zAxis], [actorConnectorMaleL,actorConnectorMaleB,actorConnectorMaleH], color4violet)
        
        graphicsActorConnectorMale2 = GraphicsDataCylinder(pAxis=[-L2/2-actorConnectorL,actorConnectorMaleB/4]+[-actorH/2-2*zAxis], vAxis=[0,actorConnectorMaleStroke]+[0], radius=actorConnectorMaleRadius, color=color4grey, nTiles=16)
    

        # axis
        graphicsAxesP5 = GraphicsDataCylinder(pAxis=[0,0]+[-0.5*dZAxis], vAxis=[0,0]+[dZAxis], radius=rAxis, color=color4red, nTiles=12)
        graphicsAxesP49 = GraphicsDataCylinder(pAxis=[-CE/2,0.]+[-0.5*dZAxisSmall], vAxis=[0,0]+[dZAxisSmall], radius=rAxis, color=color4red, nTiles=12)
        graphicsAxesP67 = GraphicsDataCylinder(pAxis=[CE/2,0.]+[-0.5*dZAxis], vAxis=[0,0]+[dZAxis], radius=rAxis, color=color4red, nTiles=12)
        graphicsAxesP38 = GraphicsDataCylinder(pAxis=[-L2/2,0.]+[-0.5*dZAxisSmall], vAxis=[0,0]+[dZAxisSmall], radius=rAxis, color=color4red, nTiles=12)

    
        cSixBarLinkage.SixBarLinkageVisualization.graphicsL2ground = [graphicsAxesP38,graphicsActor,graphicsActorHousing,graphicsActorConnector,graphicsActorConnectorFemale]+graphicsL2ground
        cSixBarLinkage.SixBarLinkageVisualization.graphicsCE1R = [graphicsAxesP5]+graphicsCE1R
        cSixBarLinkage.SixBarLinkageVisualization.graphicsCER = [graphicsAxesP67,graphicsAxesP49]+graphicsCER
        cSixBarLinkage.SixBarLinkageVisualization.graphicsCEL = [graphicsAxesP67,graphicsAxesP49]+graphicsCEL
        cSixBarLinkage.SixBarLinkageVisualization.graphicsL2 = [nodeNumberGraphics,actuatorSideNumber,ATCID,graphicsAxesP38,graphicsActor,graphicsActorStroke,graphicsActorConnector,graphicsActorConnectorMale,graphicsActorConnectorMale2]+graphicsL2

    if mode=='graphicsStandard':
        graphicsL2=[]
        
        P58Local=list(np.array(localP[8])-np.array(localP[5]))
        P56Local=list(np.array(localP[6])-np.array(localP[5]))
        #diemensions for links
        w=0.005
        zAxis=1.2*w/2
        
        #dimensions for axis
        rAxis=0.0005
        dZAxis=0.015
        dZAxisSmall=0.003
        #dimensions for actuatorHousing
        actorL=L2+0.005
        actorB=0.015
        actorH=0.01
        actorR=0.004
    
        #dimensions for actuatorHousing
        actorHousingL=0.105
        actorHousingB=0.01
        actorHousingH=0.01
        actorStroke=0.11
    
        #dimensions for connector
        actorConnectorL=0.01
        actorConnectorB=0.005
        actorConnectorH=0.005
    
        #graphicsActorConnectorFemale
        actorConnectorFemaleL=0.005
        actorConnectorFemaleB=0.01
        actorConnectorFemaleH=0.005
    
        #graphicsActorConnectorFemale
        actorConnectorMaleL=0.005
        actorConnectorMaleB=0.01
        actorConnectorMaleH=0.005        
        actorConnectorMaleStroke=0.006  
        actorConnectorMaleRadius=0.002 
        
        
        #graphics for links
        graphicsCE = GraphicsDataRigidLink(p0=[-CE/2,0.]+[-zAxis],p1=[CE/2,0.]+[-zAxis], 
                                             axis0=[0,0,1], axis1=[0,0,1], radius=[0.5*w,0.5*w], 
                                             thickness = w, width = [1.2*w,1.2*w], color=color4orange,)

        graphicsCEOtherColor = GraphicsDataRigidLink(p0=[-CE/2,0.]+[-zAxis],p1=[CE/2,0.]+[-zAxis], 
                                             axis0=[0,0,1], axis1=[0,0,1], radius=[0.5*w,0.5*w], 
                                             thickness = w, width = [1.2*w,1.2*w], color=color4orange,)
        
        graphicsCE1 = GraphicsDataRigidLink(p0=[-CE,0.]+[-zAxis],p1=[0,0]+[-zAxis], 
                                              axis0=[0,0,1], axis1=[0,0,1], radius=[0.5*w,0.5*w], 
                                              thickness = w, width = [1.2*w,1.2*w], color=color4orange,)    
        
        graphicsL21 = GraphicsDataRigidLink(p0=[0,0]+[-zAxis],p1=[L2*np.cos(psi),L2*np.sin(psi)]+[zAxis], 
                                              axis0=[0,0,1], axis1=[0,0,1], radius=[0.5*w,0.5*w], 
                                              thickness = w, width = [1.2*w,1.2*w], color=color4orange,)   
        
        # graphicsL21=[]
        
        graphicsCE2 = GraphicsDataRigidLink(p0=P58Local+[-zAxis],p1=[0,0,zAxis], 
                                              axis0=[0,0,1], axis1=[0,0,1], radius=[0.5*w,0.5*w], 
                                              thickness = w, width = [1.2*w,1.2*w], color=color4orange,)
        
        graphicsL22 = GraphicsDataRigidLink(p0=[0,0]+[zAxis],p1=P56Local+[zAxis], 
                                              axis0=[0,0,1], axis1=[0,0,1], radius=[0.5*w,0.5*w], 
                                              thickness = w, width = [1.2*w,1.2*w], color=color4orange,)        
    

        # actor
        graphicsActor = GraphicsDataOrthoCubePoint([0,0]+[-actorH/2-2*zAxis], [actorL,actorB,actorH], color4blue,)    
        graphicsActorHousing = GraphicsDataOrthoCubePoint([actorHousingL/2,0]+[-actorH/2-2*zAxis], [actorHousingL,actorHousingB,actorHousingH], color4blue,)
        graphicsActorStroke = GraphicsDataCylinder(pAxis=[L2/2,0]+[-actorH/2-2*zAxis], vAxis=[actorStroke,0]+[0], radius=actorR, color=color4grey, nTiles=16)   
    
        graphicsActorConnector = GraphicsDataOrthoCubePoint([-L2/2-actorConnectorL/2,0]+[-actorH/2-2*zAxis], [actorConnectorL,actorConnectorB,actorConnectorH], color4violet)
        graphicsActorConnectorFemale = GraphicsDataOrthoCubePoint([-L2/2-actorConnectorL,-actorConnectorFemaleB/4]+[-actorH/2-2*zAxis], [actorConnectorFemaleL,actorConnectorFemaleB,actorConnectorFemaleH], color4violet)
        graphicsActorConnectorMale = GraphicsDataOrthoCubePoint([-L2/2-actorConnectorL,actorConnectorMaleB/4]+[-actorH/2-2*zAxis], [actorConnectorMaleL,actorConnectorMaleB,actorConnectorMaleH], color4violet)
        
        graphicsActorConnectorMale2 = GraphicsDataCylinder(pAxis=[-L2/2-actorConnectorL,actorConnectorMaleB/4]+[-actorH/2-2*zAxis], vAxis=[0,actorConnectorMaleStroke]+[0], radius=actorConnectorMaleRadius, color=color4grey, nTiles=16)
    

        # axis
        graphicsAxesP5 = GraphicsDataCylinder(pAxis=[0,0]+[-0.5*dZAxis], vAxis=[0,0]+[dZAxis], radius=rAxis, color=color4red, nTiles=12)
        graphicsAxesP49 = GraphicsDataCylinder(pAxis=[-CE/2,0.]+[-0.5*dZAxisSmall], vAxis=[0,0]+[dZAxisSmall], radius=rAxis, color=color4red, nTiles=12)
        graphicsAxesP67 = GraphicsDataCylinder(pAxis=[CE/2,0.]+[-0.5*dZAxis], vAxis=[0,0]+[dZAxis], radius=rAxis, color=color4red, nTiles=12)
        graphicsAxesP38 = GraphicsDataCylinder(pAxis=[-L2/2,0.]+[-0.5*dZAxisSmall], vAxis=[0,0]+[dZAxisSmall], radius=rAxis, color=color4red, nTiles=12)
     
    
        cSixBarLinkage.SixBarLinkageVisualization.graphicsL2ground = [graphicsL2,graphicsAxesP38,graphicsActor,graphicsActorHousing,graphicsActorConnector,graphicsActorConnectorFemale]
        cSixBarLinkage.SixBarLinkageVisualization.graphicsCE1R = [graphicsCE1,graphicsL21,graphicsAxesP5]
        cSixBarLinkage.SixBarLinkageVisualization.graphicsCER = [graphicsCE,graphicsAxesP67,graphicsAxesP49]
        cSixBarLinkage.SixBarLinkageVisualization.graphicsCEL = [graphicsCEOtherColor,graphicsAxesP67,graphicsAxesP49]
        cSixBarLinkage.SixBarLinkageVisualization.graphicsCE2L = [graphicsCE2,graphicsL22]
        cSixBarLinkage.SixBarLinkageVisualization.graphicsL2 = [nodeNumberGraphics,actuatorSideNumber,ATCID,graphicsL2,graphicsAxesP38,graphicsActor,graphicsActorStroke,graphicsActorConnector,graphicsActorConnectorMale,graphicsActorConnectorMale2]

    return cSixBarLinkage

def SixBarLinkage(cSixBarLinkage, mbs, sideNumber, ID, nodeNumber):
    #initialPosition
    P00=cSixBarLinkage.centerOfRotation
    theta=cSixBarLinkage.orientation
    alphaMean=cSixBarLinkage.startAngle
    L1=cSixBarLinkage.SixBarLinkageParameters.L1
    L2=cSixBarLinkage.SixBarLinkageParameters.L2
    L3=cSixBarLinkage.SixBarLinkageParameters.L3
    L4=cSixBarLinkage.SixBarLinkageParameters.L4
    psi=cSixBarLinkage.SixBarLinkageParameters.psi
    CE=cSixBarLinkage.SixBarLinkageParameters.CE
    zeroZoneFriction=cSixBarLinkage.SixBarLinkageParameters.zeroZoneFriction
    fFriction=cSixBarLinkage.SixBarLinkageParameters.fFriction
    
    L=8e-3 #mm length of sixBarLinkage
    
    #parameters for RigidBody2D
    massRigid = cSixBarLinkage.SixBarLinkageParameters.massRigid
    inertiaRigid = cSixBarLinkage.SixBarLinkageParameters.inertiaRigid
    
    #parameters for CartesianSpringDamper for SixBarLinkages
    if config.simulation.cartesianSpringDamperActive or config.simulation.simplifiedLinksIdealCartesianSpringDamper:
        k=cSixBarLinkage.SixBarLinkageParametersConnectors.k1
        d=cSixBarLinkage.SixBarLinkageParametersConnectors.d1
        
        k=[k[0],k[1],0]
        d=[d[0],d[1],0]
        
        k=[3e5,3e5,0]
        d=[1e2,1e2,0]
        
    #parameters for RigidBodySpringDamper for SixBarLinkages
    if config.simulation.connectorRigidBodySpringDamper or config.simulation.rigidBodySpringDamperNonlinearStiffness:
        k=np.zeros((6, 6))
        k[0,0]=cSixBarLinkage.SixBarLinkageParametersConnectors.k1[0]
        k[1,1]=cSixBarLinkage.SixBarLinkageParametersConnectors.k1[1]
        k[5,5]=cSixBarLinkage.SixBarLinkageParametersConnectors.k1[2]

        k[0,1]=cSixBarLinkage.SixBarLinkageParametersConnectors.k1[3]
        k[0,5]=cSixBarLinkage.SixBarLinkageParametersConnectors.k1[4]     
        k[1,5]=cSixBarLinkage.SixBarLinkageParametersConnectors.k1[5]
        
        k[1,0]=cSixBarLinkage.SixBarLinkageParametersConnectors.k1[6]

        k[5,0]=cSixBarLinkage.SixBarLinkageParametersConnectors.k1[7]
        k[5,1]=cSixBarLinkage.SixBarLinkageParametersConnectors.k1[8]

        d=np.zeros((6, 6))
        d[0,0]=cSixBarLinkage.SixBarLinkageParametersConnectors.d1[0]
        d[1,1]=cSixBarLinkage.SixBarLinkageParametersConnectors.d1[1]
        d[5,5]=cSixBarLinkage.SixBarLinkageParametersConnectors.d1[2]
        
    #define the points/coordinates one six-bar linkage
    P01Local=[0,0]
    P1Local=[L1,0]
    P2Local=[L1+L2,0]
    P3Local=[L1,L3]
    P4Local=[L1+L2,L3]
    P5Local=[L1+CE*np.cos(alphaMean-psi),L3+CE*np.sin(alphaMean-psi)]
    P6Local=[L1+L2+CE*np.cos(alphaMean-psi),L3+CE*np.sin(alphaMean-psi)]
    P7Local=[L1+CE*np.cos(alphaMean-psi)+L2*np.cos(alphaMean),L3+CE*np.sin(alphaMean-psi)+L2*np.sin(alphaMean)]
    P8Local=[CE*np.cos(alphaMean-psi),CE*np.sin(alphaMean-psi)]
    P9Local=[CE*np.cos(alphaMean-psi)+L2*np.cos(alphaMean),CE*np.sin(alphaMean-psi)+L2*np.sin(alphaMean)]
    P10Local=[L1*np.cos(alphaMean),L1*np.sin(alphaMean)]
    P11Local=[(L1+L2)*np.cos(alphaMean),(L1+L2)*np.sin(alphaMean)]     
    
    M0Local=list((np.array(P4Local)-np.array(P3Local))/2+np.array(P3Local))
    M2Local=list((np.array(P6Local)-np.array(P4Local))/2+np.array(P4Local))
    M5Local=list((np.array(P7Local)-np.array(P9Local))/2+np.array(P9Local))
    M7Local=list((np.array(P9Local)-np.array(P8Local))/2+np.array(P8Local))
    
    
    PList=[P01Local,P1Local,P2Local,P3Local,P4Local,P5Local,P6Local,P7Local,P8Local,P9Local,P10Local,P11Local]
    
    #calculate vectors of the two long arms
    P53Local=list(np.array(PList[5])-np.array(PList[3]))
    P56Local=list(np.array(PList[6])-np.array(PList[5]))
    P57Local=list(np.array(PList[7])-np.array(PList[5]))
    P58Local=list(np.array(PList[8])-np.array(PList[5]))
     
    
    #displacement and rotation of all points P01-P11
    pT=[]
    for x in range(len(PList)):
        pT+=[list(RotZVec(theta,np.array(PList[x]))+P00)]
    
        P01=pT[0]
    P1=pT[1]
    P2=pT[2]
    P3=pT[3]
    P4=pT[4]
    P5=pT[5]
    P6=pT[6]
    P7=pT[7]
    P8=pT[8]
    P9=pT[9]
    P10=pT[10]
    P11=pT[11]
    
    M1=P5
    M6=P5
    
    P57=list(np.array(P7)-np.array(P5))
    P58=list(np.array(P8Local)-np.array(P5Local))
    
    P3L= P3 - L/2* (np.array(P5)-np.array(P3)) / np.linalg.norm(np.array(P5)-np.array(P3))
    P3R= P3 + L/2* (np.array(P5)-np.array(P3)) / np.linalg.norm(np.array(P5)-np.array(P3))
    
    P4L= P4 - L/2* (np.array(P6)-np.array(P4)) / np.linalg.norm(np.array(P6)-np.array(P4))
    P4R= P4 + L/2* (np.array(P6)-np.array(P4)) / np.linalg.norm(np.array(P6)-np.array(P4))
    
    P9L= P9 - L/2* (np.array(P7)-np.array(P9)) / np.linalg.norm(np.array(P7)-np.array(P9))
    P9R= P9 + L/2* (np.array(P7)-np.array(P9)) / np.linalg.norm(np.array(P7)-np.array(P9))
    
    P8L= P8 - L/2* (np.array(P5)-np.array(P8)) / np.linalg.norm(np.array(P5)-np.array(P8))
    P8R= P8 + L/2* (np.array(P5)-np.array(P8)) / np.linalg.norm(np.array(P5)-np.array(P8))
    
    P6L= list(np.array(P6) + np.array([-L/2*np.cos(np.pi/2-(alphaMean-psi)/2-theta) , L/2*np.sin(np.pi/2-(alphaMean-psi)/2-theta)]))
    P6R= list(np.array(P6) + np.array([L/2*np.cos(np.pi/2-(alphaMean-psi)/2-theta) , -L/2*np.sin(np.pi/2-(alphaMean-psi)/2-theta)]))
    
    P7L= list(np.array(P7) + np.array([-L/2*np.cos(np.pi/2-(alphaMean-psi)/2-psi-theta) , L/2*np.sin(np.pi/2-(alphaMean-psi)/2-psi-theta)]))
    P7R= list(np.array(P7) + np.array([L/2*np.cos(np.pi/2-(alphaMean-psi)/2-psi-theta) , -L/2*np.sin(np.pi/2-(alphaMean-psi)/2-psi-theta)]))
    
    
    P5L= list(np.array(P5) + np.array([-L/2*np.cos(np.pi/2-(alphaMean)/2-theta) , L/2*np.sin(np.pi/2-(alphaMean)/2-theta)]))
    P5R= list(np.array(P5) + np.array([L/2*np.cos(np.pi/2-(alphaMean)/2-theta) , -L/2*np.sin(np.pi/2-(alphaMean)/2-theta)]))
    
    P582Local=list(np.array(P8Local)-np.array(P5Local)) - (np.array(P8Local)-np.array(P5Local)) / np.linalg.norm(np.array(P8Local)-np.array(P5Local)) * L/2
    
    M0=list((np.array(P4)-np.array(P3))/2+np.array(P3))
    M2=list((np.array(P6)-np.array(P4))/2+np.array(P4))
    M5=list((np.array(P7)-np.array(P9))/2+np.array(P9))
    M7=list((np.array(P9)-np.array(P8))/2+np.array(P8))

    #visualization+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    cSixBarLinkage=calculateVisualization(cSixBarLinkage,PList,config.simulation.graphics,sideNumber,ID,nodeNumber)


    graphicsL2ground = cSixBarLinkage.SixBarLinkageVisualization.graphicsL2ground
    graphicsCER = cSixBarLinkage.SixBarLinkageVisualization.graphicsCER
    graphicsCEL = cSixBarLinkage.SixBarLinkageVisualization.graphicsCEL
    graphicsCE2L = cSixBarLinkage.SixBarLinkageVisualization.graphicsCE2L
    graphicsL2 = cSixBarLinkage.SixBarLinkageVisualization.graphicsL2
    graphicsCE1R = cSixBarLinkage.SixBarLinkageVisualization.graphicsCE1R
     

    mB0B5=config.lengths.weightActuator # divided by 3 for wight on each node, divided by 2 for each side
    IB0B5=config.lengths.inertiaActuator
    
    #Define rigid bodies and markers+++++++++++++++++++++++++++++++++++++++++++
    #rigid body B0 with markers mB0L,mB0R, mG1
    nRigid0 = mbs.AddNode(Rigid2D(referenceCoordinates=M0+[theta], initialVelocities=[0.,0.,0.]));
    oRigid0 = mbs.AddObject(RigidBody2D(physicsMass=mB0B5, physicsInertia=IB0B5,nodeNumber=nRigid0,visualization=VObjectRigidBody2D(graphicsData= graphicsL2ground)))
    mB0L = mbs.AddMarker(MarkerBodyPosition(bodyNumber=oRigid0, localPosition=[-L2/2,0.,0.]))
    mB0R = mbs.AddMarker(MarkerBodyPosition(bodyNumber=oRigid0, localPosition=[L2/2,0.,0.]))
    mG1 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid0, localPosition=list(-np.array(M0Local))+[0.]))   

    if not config.simulation.simplifiedLinksIdealCartesianSpringDamper and not config.simulation.simplifiedLinksIdealRevoluteJoint2D:    
        #rigid body B1 with markers mB1L,mB1R, mB1M
        nRigid1 = mbs.AddNode(Rigid2D(referenceCoordinates=P5+[alphaMean-psi+theta], initialVelocities=[0,0,0]));
        oRigid1 = mbs.AddObject(RigidBody2D(physicsMass=massRigid, physicsInertia=inertiaRigid, nodeNumber=nRigid1,visualization=VObjectRigidBody2D(graphicsData= graphicsCE1R)))
        mB1L = mbs.AddMarker(MarkerBodyPosition(bodyNumber=oRigid1, localPosition=[-CE,0.,0.])) #support point
        mB1R = mbs.AddMarker(MarkerBodyPosition(bodyNumber=oRigid1, localPosition=[L2*np.cos(psi),L2*np.sin(psi),0.])) #end point
        mB1M = mbs.AddMarker(MarkerBodyPosition(bodyNumber=oRigid1, localPosition=[0,0,0.])) #end point   
        #rigid body B2 with markers mB2L,mB2R
        nRigid2 = mbs.AddNode(Rigid2D(referenceCoordinates=M2+[alphaMean-psi+theta], initialVelocities=[0,0,0]));
        oRigid2 = mbs.AddObject(RigidBody2D(physicsMass=massRigid, physicsInertia=inertiaRigid,nodeNumber=nRigid2,visualization=VObjectRigidBody2D(graphicsData= graphicsCER)))
        mB2L = mbs.AddMarker(MarkerBodyPosition(bodyNumber=oRigid2, localPosition=[-CE/2,0.,0.])) #support point
        mB2R = mbs.AddMarker(MarkerBodyPosition(bodyNumber=oRigid2, localPosition=[CE/2,0,0.])) #end point
        #rigid body B3 with markers mB3L,mB3R, mB3M
        nRigid3 = mbs.AddNode(Rigid2D(referenceCoordinates=P5+[theta], initialVelocities=[0,0,0]));
        oRigid3 = mbs.AddObject(RigidBody2D(physicsMass=massRigid, physicsInertia=inertiaRigid,nodeNumber=nRigid3,visualization=VObjectRigidBody2D(graphicsData= graphicsCE2L)))
        mB3L = mbs.AddMarker(MarkerBodyPosition(bodyNumber=oRigid3, localPosition=P58+[0.])) #support point
        mB3R = mbs.AddMarker(MarkerBodyPosition(bodyNumber=oRigid3, localPosition=[L2,0.,0.])) #end point
        mB3M = mbs.AddMarker(MarkerBodyPosition(bodyNumber=oRigid3, localPosition=[0,0,0])) #end point
        #rigid body B4 with markers mB4L,mB4R
        nRigid4 = mbs.AddNode(Rigid2D(referenceCoordinates=M5+[psi+theta], initialVelocities=[0,0,0]));
        oRigid4 = mbs.AddObject(RigidBody2D(physicsMass=massRigid, physicsInertia=inertiaRigid,nodeNumber=nRigid4,visualization=VObjectRigidBody2D(graphicsData= graphicsCEL)))
        mB4L = mbs.AddMarker(MarkerBodyPosition(bodyNumber=oRigid4, localPosition=[-CE/2,0.,0.])) #support point
        mB4R = mbs.AddMarker(MarkerBodyPosition(bodyNumber=oRigid4, localPosition=[CE/2,0,0.])) #end point
           
    #rigid body B5 with markers mB5L,mB5R, mG2, mForce
    nRigid5 = mbs.AddNode(Rigid2D(referenceCoordinates=M7+[alphaMean+theta], initialVelocities=[0,0,0]));
    oRigid5 = mbs.AddObject(RigidBody2D(physicsMass=mB0B5, physicsInertia=IB0B5,nodeNumber=nRigid5,visualization=VObjectRigidBody2D(graphicsData= graphicsL2)))
    mB5L = mbs.AddMarker(MarkerBodyPosition(bodyNumber=oRigid5, localPosition=[-L2/2,0,0.])) #support point
    mB5R = mbs.AddMarker(MarkerBodyPosition(bodyNumber=oRigid5, localPosition=[L2/2,0,0.])) #end point
    mG2 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid5, localPosition=[-L1-L2/2,L3,0.]))
    
    
    
    ####################################
    # mbs.AddLoad(Force(markerNumber = mG1, loadVector=[-0.983*np.cos(15*np.pi/180)*0.2,0,0])) 
    ####################################
    
    
    
    #markers for prismatic joints
    mP1 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid0, localPosition=[L2/2,0.,0.]))
    mP2 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid5, localPosition=[L2/2,0.,0.]))
    
    # #plug and socket for connection
    mC1 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid0, localPosition=[-L2/2-L4,-L3,0.]))
    mC2 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid5, localPosition=[-L2/2-L4,L3,0.]))    

    
    mR13 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid5, localPosition=[-L2/2.,0.,0.]))
    mR14 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid5, localPosition=[L2/2.,0.,0.]))

    mBRR13 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid5, localPosition=[-L2/2.,0.,0.]))
    mBRR14 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid5, localPosition=[L2/2.,0.,0.]))


    #markers for actuator (measurement)
    mA1 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid0, localPosition=[-L2/2.-L4,0.,0.]))
    mA2 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid5, localPosition=[-L2/2.-L4,0.,0.]))

    #markers for prismatic joints
    mP02 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid0, localPosition=[L2/2.,0.,0.]))
    mP14 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid5, localPosition=[L2/2.,0.,0.]))


    #markers for sensors (viconMarkers)
    mPVicon1 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid0, localPosition=[-L2/2+0.01494255,-0.0052677,0.]))
    mPVicon2 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid5, localPosition=[-L2/2+0.00494255,0.0052677,0.]))
    
    mPVicon3 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid5, localPosition=[-L2/2+0.0049,0.0053,0.]))
    
    #markers for friction at the pins
    mPFriction1 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid0, localPosition=[-L2/2+0.00494255,-0.0052677,0.]))
    mPFriction2 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid5, localPosition=[-L2/2+0.00494255,0.0052677,0.]))
    
    #friction
    oGround = mbs.AddObject(ObjectGround(referencePosition = [0,0,0]))
    mGround0 = mbs.AddMarker(MarkerBodyPosition(bodyNumber=oGround, localPosition = [0,0,0]))
    
    visFriction = VObjectConnectorCartesianSpringDamper(show=False) #do not show friction elements
    mbs.AddObject(CartesianSpringDamper(markerNumbers=[mGround0, mPFriction1], 
                                    offset=[config.simulation.zeroZoneFriction, config.simulation.fFriction, 0], 
                                    springForceUserFunction=UserFunctionSpringDamperFriction, 
                                    visualization=visFriction))
    mbs.AddObject(CartesianSpringDamper(markerNumbers=[mGround0, mPFriction2], 
                                    offset=[config.simulation.zeroZoneFriction, config.simulation.fFriction, 0], 
                                    springForceUserFunction=UserFunctionSpringDamperFriction,
                                    visualization=visFriction))

    if not config.simulation.simplifiedLinksIdealCartesianSpringDamper and not config.simulation.simplifiedLinksIdealRevoluteJoint2D:    
        alphaMeanInitial=66.25*np.pi/180
        
        deltaAlpha=(alphaMeanInitial-alphaMean)
        #add marker for flexure hinges with lumped compliance and circular hinge contour
        alphaMeanDummy=alphaMean
        
        alphaMean=alphaMeanInitial
        mB0LGEB2D = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid0, localPosition=[-L2/2-L/2*np.cos(alphaMean-psi),-L/2*np.sin(alphaMean-psi),0.]))
        mB1LGEB2D = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid1, localPosition=[-CE+L/2,0,0]))
        
        mB0RGEB2D = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid0, localPosition=[L2/2-L/2*np.cos(alphaMean-psi),-L/2*np.sin(alphaMean-psi),0.]))
        mB2LGEB2D = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid2, localPosition=[-CE/2+L/2,0.,0.]))
        
        mB5RGEB2D = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid5, localPosition=[L2/2-L/2*np.cos(alphaMean-psi),L/2*np.sin(alphaMean-psi),0]))
        mB4LGEB2D = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid4, localPosition=[-CE/2+L/2,0.,0.]))
        
        mB5LGEB2D = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid5, localPosition=[-L2/2-L/2*np.cos(alphaMean-psi),L/2*np.sin(alphaMean-psi),0]))
        mB3LGEB2D = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid3, localPosition=[P582Local[0],P582Local[1],0.]))
        
        mB3RGEB2D = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid3, localPosition=[L2-L/2*np.cos(np.pi/2-(alphaMean-psi)/2),L/2*np.sin(np.pi/2-(alphaMean-psi)/2),0.]))
        mB2RGEB2D = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid2, localPosition=[CE/2-L/2*np.cos(np.pi/2-(alphaMean-psi)/2),-L/2*np.sin(np.pi/2-(alphaMean-psi)/2),0.]))
        
        mB4RGEB2D = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid4, localPosition=[CE/2-L/2*np.cos(np.pi/2-(alphaMean-psi)/2),L/2*np.sin(np.pi/2-(alphaMean-psi)/2),0.]))
        mB1RGEB2D = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid1, localPosition=[L2*np.cos(psi)-L/2*np.cos(np.pi/2-(alphaMean-psi)/2+psi),L2*np.sin(psi)-L/2*np.sin(np.pi/2-(alphaMean-psi)/2+psi),0.]))
        
        mB3MGEB2D = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid3, localPosition=[L/2*np.cos(np.pi/2-alphaMean/2),-L/2*np.sin(np.pi/2-alphaMean/2),0.]))
        mB1MGEB2D = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid1, localPosition=[L/2*np.cos(-alphaMean/2+psi+np.pi/2),L/2*np.sin(-alphaMean/2+psi+np.pi/2),0.]))


    #   Compliant joints
    if config.simulation.useCompliantJoints:  
        circularHingeHight=8e-3*1e8    #6.5e-3 *2 (atm not implemented)
        circularHingeHight2=8e-3*1e8   #40e-3 (atm not implemented)
        
        CompliantJointsGeometricallyExactBeam2D(mbs,mB0LGEB2D,mB1LGEB2D,(alphaMean-psi+theta ) -deltaAlpha ,-theta +deltaAlpha,0,0,0, P3L,P3R, L,circularHingeHight)
        CompliantJointsGeometricallyExactBeam2D(mbs,mB0RGEB2D,mB2LGEB2D,(alphaMean-psi+theta) -deltaAlpha,-theta +deltaAlpha,0,0,0, P4L,P4R, L,circularHingeHight)
        
        CompliantJointsGeometricallyExactBeam2D(mbs,mB5RGEB2D,mB4LGEB2D,psi+theta,-alphaMean-theta ,0,0,0, P9L,P9R, L,circularHingeHight)
        CompliantJointsGeometricallyExactBeam2D(mbs,mB5LGEB2D,mB3LGEB2D,psi+theta,-alphaMean-theta,psi,0,0, P8L,P8R, L,circularHingeHight)
            
        
        CompliantJointsGeometricallyExactBeam2D(mbs,mB3RGEB2D,mB2RGEB2D,-(np.pi/2-(alphaMean-psi)/2)+theta -deltaAlpha/2 , -theta +deltaAlpha/2, np.pi/2-(alphaMean-psi)/2 -np.pi ,0, 0, P6L,P6R, L,circularHingeHight2)
        CompliantJointsGeometricallyExactBeam2D(mbs,mB4RGEB2D,mB1RGEB2D,-(np.pi/2-(alphaMean-psi)/2)+psi+theta -deltaAlpha/2 , -psi-theta +deltaAlpha/2  ,np.pi/2-(alphaMean-psi)/2+psi -np.pi ,0,0, P7L,P7R, L,circularHingeHight2) 
        
        CompliantJointsGeometricallyExactBeam2D(mbs,mB1MGEB2D,mB3MGEB2D,-((np.pi/2-alphaMean/2))+theta -deltaAlpha/2 , -alphaMean+psi-theta - np.pi/2*0 +deltaAlpha/2  ,(-np.pi/2+(alphaMean)/2) , 0, 0, P5L,P5R, L,circularHingeHight2)
    
    # #define joints+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   
    elif config.simulation.useCrossSpringPivot:    
        crossSpringPivotHight=7.5e-3 *2 #(atm not implemented)
        crossSpringPivotHight2=40e-3 #(atm not implemented)
        
        CrossSpringPivot(mbs,mB0LGEB2D,mB1LGEB2D,(alphaMean-psi+theta ) -deltaAlpha,-theta +deltaAlpha,0,0,0, P3L,P3R, L,crossSpringPivotHight)
        CrossSpringPivot(mbs,mB0RGEB2D,mB2LGEB2D,(alphaMean-psi+theta) -deltaAlpha,-theta +deltaAlpha,0,0,0, P4L,P4R, L,crossSpringPivotHight)
        
        CrossSpringPivot(mbs,mB5RGEB2D,mB4LGEB2D,psi+theta,-alphaMean-theta,0,0,0, P9L,P9R, L,crossSpringPivotHight)
        CrossSpringPivot(mbs,mB5LGEB2D,mB3LGEB2D,psi+theta,-alphaMean-theta,psi,0,0, P8L,P8R, L,crossSpringPivotHight)
        
        CrossSpringPivot(mbs,mB3RGEB2D,mB2RGEB2D,-(np.pi/2-(alphaMean-psi)/2)+theta -deltaAlpha/2 , -theta +deltaAlpha/2, np.pi/2-(alphaMean-psi)/2 -np.pi ,0, 0, P6L,P6R, L, crossSpringPivotHight2)
        CrossSpringPivot(mbs,mB4RGEB2D,mB1RGEB2D,-(np.pi/2-(alphaMean-psi)/2)+psi+theta -deltaAlpha/2, -psi-theta+deltaAlpha/2  ,np.pi/2-(alphaMean-psi)/2+psi -np.pi,0,0, P7L,P7R, L, crossSpringPivotHight2) 
        
        CrossSpringPivot(mbs,mB1MGEB2D,mB3MGEB2D,-((np.pi/2-alphaMean/2))+theta -deltaAlpha/2 , -alphaMean+psi-theta - np.pi/2*0 +deltaAlpha/2 ,(-np.pi/2+(alphaMean)/2), 0, 0, P5L,P5R, L, crossSpringPivotHight2)      
     
    elif config.simulation.cartesianSpringDamperActive:
        mbs.AddObject(CartesianSpringDamper(markerNumbers=[mB0L,mB1L],stiffness=k,damping=d))
        mbs.AddObject(CartesianSpringDamper(markerNumbers=[mB0R,mB2L],stiffness=k,damping=d))
              
        mbs.AddObject(CartesianSpringDamper(markerNumbers=[mB2R,mB3R],stiffness=k,damping=d))
        mbs.AddObject(CartesianSpringDamper(markerNumbers=[mB3M,mB1M],stiffness=k,damping=d))
        mbs.AddObject(CartesianSpringDamper(markerNumbers=[mB4R,mB1R],stiffness=k,damping=d))
        
        mbs.AddObject(CartesianSpringDamper(markerNumbers=[mB5L,mB3L],stiffness=k,damping=d))
        mbs.AddObject(CartesianSpringDamper(markerNumbers=[mB5R,mB4L],stiffness=k,damping=d))
        
    elif config.simulation.connectorRigidBodySpringDamper:
        initialRotation=66.25*np.pi/180-alphaMean
        
        mBRR01 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid0, localPosition=[-L2/2,0,0.]))
        mBRR02 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid0, localPosition=[L2/2,0,0.]))
    
        mBRR1 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid1, localPosition=[-CE,0.,0.]))
        mBRR2 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid1, localPosition=[L2*np.cos(psi),L2*np.sin(psi),0.]))    
        mBRR15 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid1, localPosition=[0.,0.,0.]))
    
        mBRR3 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid2, localPosition=[-CE/2.,0.,0]))
        mBRR4 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid2, localPosition=[CE/2.,0.,0]))
    
        mBRR9 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid4, localPosition=[-CE/2.,0.,0]))
        mBRR10 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid4, localPosition=[CE/2.,0.,0]))
    
        mBRR13 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid5, localPosition=[-L2/2.,0.,0.]))
        mBRR14 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid5, localPosition=[L2/2.,0.,0.]))
    
        mBRR11 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid3, localPosition=P58+[0.]))
        mBRR12 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid3, localPosition=[L2,0.,0.]))
        mBRR65 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid3, localPosition=[0.,0.,0.]))


        k2=k
        k=k2*2  
        #connect mass left side with cableLeft
        # initialRotation=0
        #P3    
        initialRotation1 = RotationMatrixZ(alphaMean-psi+initialRotation)
        initialRotation2 = RotationMatrixZ(0+initialRotation*0)
        mbs.AddObject(RigidBodySpringDamper(markerNumbers=[mBRR01,mBRR1],
                                            stiffness=k,
                                            damping=d,
                                            rotationMarker0=initialRotation1,
                                            rotationMarker1=initialRotation2,                                        
                                            ))
        
        #P4    
        mbs.AddObject(RigidBodySpringDamper(markerNumbers=[mBRR02,mBRR3],
                                            stiffness=k,
                                            damping=d,
                                            rotationMarker0=initialRotation1,
                                            rotationMarker1=initialRotation2,                                        
                                            ))
     
    
        #P9
        initialRotation1 = RotationMatrixZ(-alphaMean+psi -initialRotation )
        initialRotation2 = RotationMatrixZ(0+initialRotation-initialRotation)  
        mbs.AddObject(RigidBodySpringDamper(markerNumbers=[mBRR14,mBRR9],
                                            stiffness=k,
                                            damping=d,
                                            rotationMarker0=initialRotation1,
                                            rotationMarker1=initialRotation2,                                        
                                            ))    
        
        
        #P8
        initialRotation1 = RotationMatrixZ(-alphaMean+psi -initialRotation)
        initialRotation2 = RotationMatrixZ(psi+initialRotation-initialRotation)        
        mbs.AddObject(RigidBodySpringDamper(markerNumbers=[mBRR13,mBRR11],
                                            stiffness=k,
                                            damping=d,
                                            rotationMarker0=initialRotation1,
                                            rotationMarker1=initialRotation2,                                        
                                            ))   
                  
        k=k2*6 
        
        #P5                         
        initialRotation1 = RotationMatrixZ( (-alphaMean/2+psi)-np.pi/2 -initialRotation/2)
        initialRotation2 = RotationMatrixZ( (alphaMean/2-np.pi/2) +initialRotation/2)         
        mbs.AddObject(RigidBodySpringDamper(markerNumbers=[mBRR15,mBRR65],
                                            stiffness=k,
                                            damping=d,
                                            rotationMarker0=initialRotation1,
                                            rotationMarker1=initialRotation2,                                          
                                            ))
        
        #P6 markerNumbers and rotationMarker0 and rotationMarker1 swapped
        initialRotation1 = RotationMatrixZ( -(alphaMean-psi)/2 - np.pi/2 -initialRotation/2)
        initialRotation2 = RotationMatrixZ( (alphaMean-psi)/2 - np.pi/2  +initialRotation/2)    
        mbs.AddObject(RigidBodySpringDamper(markerNumbers=[mBRR12,mBRR4],
                                            stiffness=k,
                                            damping=d,
                                            rotationMarker0=initialRotation2,
                                            rotationMarker1=initialRotation1,                                          
                                            ))                                           
    
        #P7 markerNumbers and rotationMarker0 and rotationMarker1 swapped
        initialRotation1 = RotationMatrixZ(psi- (alphaMean-psi)/2 - np.pi/2 -initialRotation/2)
        initialRotation2 = RotationMatrixZ( (alphaMean-psi)/2 - np.pi/2 +initialRotation/2)
        mbs.AddObject(RigidBodySpringDamper(markerNumbers=[mBRR10,mBRR2],
                                            stiffness=k,
                                            damping=d,
                                            rotationMarker0=initialRotation2,
                                            rotationMarker1=initialRotation1,                                        
                                            ))            
        
        # mbs.AddObject(RigidBodySpringDamper(markerNumbers=[mBRR01,mBRR1],stiffness=k,damping=d))
        # mbs.AddObject(RigidBodySpringDamper(markerNumbers=[mBRR02,mBRR3],stiffness=k,damping=d))
              
        # mbs.AddObject(RigidBodySpringDamper(markerNumbers=[mBRR10,mBRR2],stiffness=k,damping=d))
        # mbs.AddObject(RigidBodySpringDamper(markerNumbers=[mBRR14,mBRR9],stiffness=k,damping=d))
        # mbs.AddObject(RigidBodySpringDamper(markerNumbers=[mBRR13,mBRR11],stiffness=k,damping=d))
        
        # mbs.AddObject(RigidBodySpringDamper(markerNumbers=[mBRR15,mBRR65],stiffness=k,damping=d))
        # mbs.AddObject(RigidBodySpringDamper(markerNumbers=[mBRR4,mBRR12],stiffness=k,damping=d))        


    elif config.simulation.rigidBodySpringDamperNonlinearStiffness:
        initialRotation=alphaMeanInitial-alphaMean
        
        mBRR01 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid0, localPosition=[-L2/2,0,0.]))
        mBRR02 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid0, localPosition=[L2/2,0,0.]))
    
        mBRR1 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid1, localPosition=[-CE,0.,0.]))
        mBRR2 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid1, localPosition=[L2*np.cos(psi),L2*np.sin(psi),0.]))    
        mBRR15 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid1, localPosition=[0.,0.,0.]))
    
        mBRR3 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid2, localPosition=[-CE/2.,0.,0]))
        mBRR4 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid2, localPosition=[CE/2.,0.,0]))
    
        mBRR9 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid4, localPosition=[-CE/2.,0.,0]))
        mBRR10 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid4, localPosition=[CE/2.,0.,0]))
    
        mBRR13 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid5, localPosition=[-L2/2.,0.,0.]))
        mBRR14 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid5, localPosition=[L2/2.,0.,0.]))
    
        mBRR11 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid3, localPosition=P58Local+[0.]))
        mBRR12 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid3, localPosition=P56Local+[0.]))
        mBRR65 = mbs.AddMarker(MarkerBodyRigid(bodyNumber=oRigid3, localPosition=[0.,0.,0.]))
    
        # k=np.zeros((6, 6))
        # d=np.zeros((6, 6))
    
        #connect mass left side with cableLeft
        
        if config.simulation.rigidBodySpringDamperNeuralNetworkCircularHinge:
                
            input_size = 3
            hidden_size = 16
            output_size = 3  
            # Define the neural network model
            class NeuralNetwork(nn.Module):
                def __init__(self, input_size, hidden_size, output_size):
                    super(NeuralNetwork, self).__init__()
                    self.layer1 = nn.Linear(input_size, hidden_size)
                    self.layer2 = nn.Linear(hidden_size, hidden_size)
                    self.layer3 = nn.Linear(hidden_size, hidden_size)
                    self.elu = nn.ELU()
                    self.layer4 = nn.Linear(hidden_size, output_size)
                def forward(self, x):
                    x = self.layer1(x)
                    x = self.elu(x)
                    x = self.layer2(x)
                    x = self.elu(x)
                    x = self.layer3(x)
                    x = self.elu(x)
                    x = self.layer4(x)
                    return x
                
            
                
            # circularHinge
            maxValueFx=0.776301559864187 #in N
            maxValueFy=0.776301559864187 #in N
            maxValueM=0.00284923912846732 #in Nm
            maxValueVx=100 #in um
            maxValueVy=150 #in um
            maxValuePhi=25*np.pi/180 #in rad
        
    
        
        if config.simulation.rigidBodySpringDamperNeuralNetworkCrossSprings:      
            # Define the neural network model
            input_size = 3
            hidden_size = 16
            output_size = 3  
            # Define the neural network model
            class NeuralNetwork(nn.Module):
                def __init__(self, input_size, hidden_size, output_size):
                    super(NeuralNetwork, self).__init__()
                    self.layer1 = nn.Linear(input_size, hidden_size)
                    self.layer2 = nn.Linear(hidden_size, hidden_size)
                    self.layer3 = nn.Linear(hidden_size, hidden_size)
                    self.elu = nn.ELU()
                    self.layer4 = nn.Linear(hidden_size, output_size)
                def forward(self, x):
                    x = self.layer1(x)
                    x = self.elu(x)
                    x = self.layer2(x)
                    x = self.elu(x)
                    x = self.layer3(x)
                    x = self.elu(x)
                    x = self.layer4(x)
                    return x
 
            # crossSpring
            maxValueFx=2 #in N
            maxValueFy=2 #in N
            maxValueM=0.004 #in Nm
            
            maxValueVx=100 #in um
            maxValueVy=150 #in um
            maxValuePhi=25*np.pi/180 #in rad   
        
        if config.simulation.rigidBodySpringDamperNeuralNetworkCircularHinge or config.simulation.rigidBodySpringDamperNeuralNetworkCrossSprings:
            def map_OutputVectorBack(value):
                Fx=value[0]
                Fy=value[1]
                M=value[2]
                
                min_value = -1.0
                max_value = 1.0
                new_min = -maxValueFx
                new_max = maxValueFx      
                FxMapped=map_to_range(Fx, min_value, max_value, new_min, new_max)
            
                min_value = -1.0
                max_value = 1.0
                new_min = -maxValueFy
                new_max = maxValueFy       
                FyMapped=map_to_range(Fy, min_value, max_value, new_min, new_max)    
                
                min_value = -1.0
                max_value = 1.0
                new_min = -maxValueM
                new_max = maxValueM      
                mMapped=map_to_range(M, min_value, max_value, new_min, new_max)        
                
                return [FxMapped, FyMapped, mMapped]
            
            def map_InputVector(value):
                Vx=value[0]
                Vy=value[1]
                Phi=value[2]
                
                min_value = -maxValueVx
                max_value = maxValueVx
                new_min = -1.0
                new_max = 1.0        
                VxMapped=map_to_range(Vx, min_value, max_value, new_min, new_max)
            
                min_value = -maxValueVy
                max_value = maxValueVy
                new_min = -1.0
                new_max = 1.0        
                VyMapped=map_to_range(Vy, min_value, max_value, new_min, new_max)    
                
                min_value = -maxValuePhi
                max_value = maxValuePhi
                new_min = -1.0
                new_max = 1.0        
                PhiMapped=map_to_range(Phi, min_value, max_value, new_min, new_max)        
                
                return [VxMapped, VyMapped, PhiMapped]        
    



        if config.simulation.rigidBodySpringDamperNeuralNetworkCircularHinge or config.simulation.rigidBodySpringDamperNeuralNetworkCrossSprings:
            
            # Create an instance of the model (with the same architecture)
            model = NeuralNetwork(input_size, hidden_size, output_size).double()
            # Load the model's state dictionary from the file
            
            if config.simulation.rigidBodySpringDamperNeuralNetworkCircularHinge:
                model.load_state_dict(torch.load('pyTorchCircularHinge.pth'))
                model.eval()
            if config.simulation.rigidBodySpringDamperNeuralNetworkCrossSprings:
                model.load_state_dict(torch.load('pyTorchCrossSpring.pth'))  
                model.eval()

            if config.simulation.neuralnetworkJIT:
                # Extract weights and biases into numpy arrays
                weights = {}
                for name, param in model.named_parameters():
                    weights[name] = param.detach().numpy()
                
                # Example of accessing the weights and biases
                W1 = weights['layer1.weight']
                b1 = weights['layer1.bias']
                W2 = weights['layer2.weight']
                b2 = weights['layer2.bias']
                W3 = weights['layer3.weight']
                b3 = weights['layer3.bias']
                W4 = weights['layer4.weight']
                b4 = weights['layer4.bias']
                
                # Define activation functions
                # JIT compile the ELU function
                @jit(nopython=True)
                def elu(x, alpha=1.0):
                    return np.where(x > 0, x, alpha * (np.exp(x) - 1))
                
                
                # JIT compile the predict function
                @jit(nopython=True)
                def predict(x):
                    # Layer 1: Linear + ELU
                    x = np.dot(x, W1.T) + b1
                    x = elu(x)
                    
                    # Layer 2: Linear + ReLU
                    x = np.dot(x, W2.T) + b2
                    x = elu(x)
                    
                    # Layer 3: Linear + ELU
                    x = np.dot(x, W3.T) + b3
                    x = elu(x)
                    
                    # Layer 4: Linear (Output layer, no activation)
                    x = np.dot(x, W4.T) + b4
                    
                    return x          
        
        def UFforce(mbs, t, itemNumber, displacement, rotation, velocity, angularVelocity,
                        stiffness, damping, rotJ0, rotJ1, offset):
  
                # mbs.variables['counterForTesting']=mbs.variables['counterForTesting']+1
                
                if config.simulation.rigidBodySpringDamperNonlinearStiffnessCrossSpringPivot: 
                    x0=displacement[0]
                    x1=displacement[1]
                    x2=rotation[2]                                                     
                    ############++++++++++++++++++++++++++++End Version of Cross Spring
                    para_fx_CS=[ 5.25715101e+05 ,-1.91270421e+01 , 1.07390097e-03 ,-1.57676916e+09,#CrossSpring_output9 nonlinear3
                      2.36104272e+05, -4.13776939e+08, -3.35276273e+01,  1.17969942e+05,
                      -5.91303162e+02, -7.74020374e+04,  6.99234373e+04, -3.18149758e+08,
                      -8.34484757e+03, -3.41417346e+07,  4.61470356e+08, -3.81932478e+03,
                      1.50717694e+04, -3.71020344e+07, -3.00846994e+03,  4.67301155e+06,
                      1.35256450e+08,  3.18143220e+08, -3.16436584e+07, -4.59678571e+08,
                      -5.44210730e+06, -1.35256402e+08,  2.00886011e-02, -1.95372745e-02]
                    
                    para_fy_CS=[-1.89304965e+01 , 3.58680763e+05 ,-1.29252988e+00 , 2.42061057e+05,
                      -7.92473650e+08, -4.27994996e+05,  2.47646481e+04, -5.65711037e+00,
                      2.26000111e-02, -1.27429454e+08, -3.43275022e+04,  2.03400874e+08,
                      2.48193932e+04,  5.41622542e+08, -1.25655594e+08,  3.39662953e+07,
                      -1.04675622e+05, -7.14901254e+08,  4.73144272e+04,  5.97583420e+08,
                      1.09444302e+08,  2.01424687e+08,  1.73824373e+08,  1.25655315e+08,
                      5.97261412e+08, -1.09833563e+08, -1.95419449e+02, -2.07265965e-05]
                    
                    para_m_CS=[-1.37935468e-03, -6.18070331e+00,  1.49509033e-02,  1.62580717e+00,
                      2.62928692e+06, -2.16229794e+03, -8.90247952e+02, -1.56385357e-01,
                      2.26715321e-06, -8.13080583e+05, -6.10553065e+06, -1.00139563e+05,
                      1.57906389e+06, -1.46587717e+06, -2.28662162e+05, -6.06670125e+06,
                      -3.03750544e+06,  1.43126314e+06,  1.03364911e+01, -5.66586124e+05,
                      1.37288552e+06,  1.93200544e+05,  3.78808522e+04,  2.28662221e+05,
                      1.20118773e+06, -1.37627242e+06,  1.00442186e+00,  4.01209368e-08]
                    
                    
                    var=[x0,x1,x2,x0*x0,x1*x0,x1*x1,x2*x0,x2*x1,x2*x2,
                          x0*x0*x0,x0*x0*x1,x0*x0*x2,x0*x1*x1,x0*x1*x2,x0*x2*x2,
                          x1*x0*x0,x1*x0*x1,x1*x0*x2,x1*x1*x1,x1*x1*x2,x1*x2*x2,
                          x2*x0*x0,x2*x0*x1,x2*x0*x2,x2*x1*x1,x2*x1*x2,x2*x2*x2,1]
                    
                    Fx=VMult(var, para_fx_CS)
                    Fy=VMult(var, para_fy_CS)
                    M=VMult(var, para_m_CS)
                    
                    # Fx, Fy, M = calculate_forces(displacement, rotation, para_fx_CS, para_fy_CS, para_m_CS)
                
                if config.simulation.rigidBodySpringDamperNonlinearStiffnessCircularHinge:
                    x0 = displacement[0]
                    x1 = displacement[1]
                    x2 = rotation[2]                    
                    
                    
                    para_fx_NH=[ 4.00493796e+04 ,-1.77632813e-01,  3.26495550e-06, -3.96579050e+07,
                      1.64359899e+03,  3.10235955e+06,  8.64913296e-02,  1.77662811e+04,
                     -1.06427744e+01, -6.42016208e+02,  1.49291473e+04,  1.37602813e+06,
                      4.53687492e+04, -1.12961208e+07,  1.23519573e+07, -3.80804774e+03,
                     -4.97186517e+04, -1.12876883e+07,  9.02663681e+06,  1.58388129e+07,
                     -8.69967843e+05, -1.37587586e+06, -1.13030810e+07, -1.23488345e+07,
                     -1.58383768e+07,  8.69967460e+05, -3.32561388e-05,  5.33176091e-03]
                    
                    para_fy_NH=[ 1.75237630e+00,  4.57755006e+03,  2.03092861e-03, -4.22954872e+04,
                      6.66588330e+06,  1.41477354e+03,  1.68403231e+04, -3.38465467e-01,
                      6.12011929e-05,  4.07024542e+08, -1.24469945e+10,  1.43951494e+09,
                     -1.27926165e+08,  1.65004438e+08,  1.47270799e+09, -8.65042375e+09,
                      3.41958330e+07, -5.95260621e+08,  1.40516801e+09, -3.32647555e+09,
                     -2.42679601e+08, -1.45070808e+09,  4.30253329e+08, -1.47270799e+09,
                      3.33123251e+09,  2.42686484e+08, -5.12644619e+00, -1.24880290e-05]
                    
                    para_m_NH=[ 4.18091162e-04, -3.55106206e-03,  6.40152669e-03, -1.78905377e+01,
                      1.79202781e+04,  3.42203746e-01, -2.11664168e+01, -2.07993008e-04,
                      7.67023058e-08,  2.12018896e+05,  5.83950940e+06,  2.12329513e+05,
                      8.36222989e+04, -3.07569916e+05, -4.75100028e+05,  4.53444883e+05,
                     -1.01540480e+05,  1.97750745e+05,  1.60641963e+06,  3.08406735e+03,
                     -4.31091511e+04, -2.01254755e+05,  1.09822600e+05,  4.75100027e+05,
                      4.29522609e+03,  4.30933453e+04,  5.68438644e-03, -1.93944401e-09]   
                    
                    var=[x0,x1,x2,x0*x0,x1*x0,x1*x1,x2*x0,x2*x1,x2*x2,
                         x0*x0*x0,x0*x0*x1,x0*x0*x2,x0*x1*x1,x0*x1*x2,x0*x2*x2,
                         x1*x0*x0,x1*x0*x1,x1*x0*x2,x1*x1*x1,x1*x1*x2,x1*x2*x2,
                         x2*x0*x0,x2*x0*x1,x2*x0*x2,x2*x1*x1,x2*x1*x2,x2*x2*x2,1]
                    
                    Fx=VMult(var, para_fx_NH)
                    Fy=VMult(var, para_fy_NH)
                    M=VMult(var, para_m_NH)                   
                    
                    
                if config.simulation.rigidBodySpringDamperNeuralNetworkCircularHinge or config.simulation.rigidBodySpringDamperNeuralNetworkCrossSprings:
                    # # print(t)
                    # # Example usage:
                    # # t = 12
                    # # usebatch=(t>0.23 and t<0.24)
                    # # usebatch=False
                    
                    # if mbs.variables['flag']:
                    #     # usebatch= t>0.002
                    #     usebatch= t>0.2
                        
                    # else:
                    #     usebatch=False
                    # usebatch=False
                    # print(usebatch)
                    # # usebatch
                    # usebatch=False
                    
                    usebatch=False #set usebatch=False for noBatch
                    
                    
                    if not usebatch:
                        # uu = list(np.array(displacement[0:2])*1e6)+[rotation[2]]
                        uu = [displacement[0]*1e6,displacement[1]*1e6,rotation[2]]
                        X_test = map_InputVector(uu)
                        
                        if config.simulation.neuralnetworkJIT:
                            predictions=predict(np.array(X_test))
                        else:
                            predictions=model(torch.tensor(X_test, dtype=torch.float64)).detach().numpy()
                        
                        output=map_OutputVectorBack(predictions)
                    # output=[output[0],output[1],output[2]]
                    # print(itemNumber,'single:',output)

                    # else:
                        # print('batch!')
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   
                    
                        # # Convert items in the list to integers (if they are not already)
                        # rigid_body_spring_dampers = mbs.variables['rigidBodySpringDampers']
                        # # Find the index of 'target' in 'rigidBodySpringDampers'
                        # index = rigid_body_spring_dampers.index(itemNumber)
                        # # Retrieve the corresponding predicted forces using the index
                        # predicted_force = mbs.variables['predictedForces'][index]
                        
                        # predicted_force=map_OutputVectorBack(predicted_force)
                        # output2=[predicted_force[0],predicted_force[1],-predicted_force[2]]

                        # output=output2                    

                    
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++                      
                    
                    # uu = list(np.array(displacement[0:2])*1e6)+[rotation[2]]
                    # X_test = map_InputVector(uu)       
                    # predictions=model(torch.tensor(X_test, dtype=torch.float64)) #.detach().numpy()
                    # output=map_OutputVectorBack(predictions)
                    
                    
                    Fx=output[0]
                    Fy=output[1]
                    M=-output[2]
                    
                    
                return [Fx,Fy,0,0,0,M]  
        
        
        
        
        
        


        # initialRotation=0
        #P3    
        initialRotation1 = RotationMatrixZ(alphaMean-psi +initialRotation)
        initialRotation2 = RotationMatrixZ(0+initialRotation*0)

        mbs.AddObject(RigidBodySpringDamper(markerNumbers=[mBRR01,mBRR1],
                                            stiffness=k,
                                            damping=d,
                                            rotationMarker0=initialRotation1,
                                            rotationMarker1=initialRotation2,                                        
                                            springForceTorqueUserFunction = UFforce
                                            ))
        
        #P4    
        mbs.AddObject(RigidBodySpringDamper(markerNumbers=[mBRR02,mBRR3],
                                            stiffness=k,
                                            damping=d,
                                            rotationMarker0=initialRotation1,
                                            rotationMarker1=initialRotation2,                                        
                                            springForceTorqueUserFunction = UFforce
                                            ))
     
    
        #P9
        initialRotation1 = RotationMatrixZ(-alphaMean+psi -initialRotation )
        initialRotation2 = RotationMatrixZ(0+initialRotation-initialRotation)    
        mbs.AddObject(RigidBodySpringDamper(markerNumbers=[mBRR14,mBRR9],
                                            stiffness=k,
                                            damping=d,
                                            rotationMarker0=initialRotation1,
                                            rotationMarker1=initialRotation2,                                        
                                            springForceTorqueUserFunction = UFforce
                                            ))
        
        
        #P8
        initialRotation1 = RotationMatrixZ(-alphaMean+psi -initialRotation)
        initialRotation2 = RotationMatrixZ(psi+initialRotation-initialRotation)      
        mbs.AddObject(RigidBodySpringDamper(markerNumbers=[mBRR13,mBRR11],
                                            stiffness=k,
                                            damping=d,
                                            rotationMarker0=initialRotation1,
                                            rotationMarker1=initialRotation2 ,                                       
                                            springForceTorqueUserFunction = UFforce
                                            ))
                  
        
        #P5                         
        initialRotation1 = RotationMatrixZ( (-alphaMean/2+psi)-np.pi/2 -initialRotation/2)
        initialRotation2 = RotationMatrixZ( (alphaMean/2-np.pi/2) +initialRotation/2)         
        mbs.AddObject(RigidBodySpringDamper(markerNumbers=[mBRR15,mBRR65],
                                            stiffness=k,
                                            damping=d,
                                            rotationMarker0=initialRotation1,
                                            rotationMarker1=initialRotation2,                                 
                                            springForceTorqueUserFunction = UFforce
                                            ))
        
        #P6
        initialRotation1 = RotationMatrixZ( -(alphaMean-psi)/2 - np.pi/2 -initialRotation/2)
        initialRotation2 = RotationMatrixZ( (alphaMean-psi)/2 - np.pi/2  +initialRotation/2)    
        mbs.AddObject(RigidBodySpringDamper(markerNumbers=[mBRR12,mBRR4],
                                            stiffness=k,
                                            damping=d,
                                            rotationMarker0=initialRotation2,
                                            rotationMarker1=initialRotation1,                                          
                                            springForceTorqueUserFunction = UFforce
                                            ))                                          
    
        #P7
        initialRotation1 = RotationMatrixZ(psi- (alphaMean-psi)/2 - np.pi/2 -initialRotation/2)
        initialRotation2 = RotationMatrixZ( (alphaMean-psi)/2 - np.pi/2 +initialRotation/2)
        mbs.AddObject(RigidBodySpringDamper(markerNumbers=[mBRR10,mBRR2],
                                            stiffness=k,
                                            damping=d,
                                            rotationMarker0=initialRotation2,
                                            rotationMarker1=initialRotation1,                                        
                                            springForceTorqueUserFunction = UFforce
                                            ))  







        

    elif config.simulation.simplifiedLinksIdealRevoluteJoint2D:
        mbs.AddObject(RevoluteJoint2D(markerNumbers=[mG1,mG2]))
 

    elif config.simulation.simplifiedLinksIdealCartesianSpringDamper:
        mbs.AddObject(CartesianSpringDamper(markerNumbers=[mG1,mG2],stiffness=k,damping=d))
        
    else:
        mbs.AddObject(RevoluteJoint2D(markerNumbers=[mB0L,mB1L]))
        mbs.AddObject(RevoluteJoint2D(markerNumbers=[mB0R,mB2L]))
               
        mbs.AddObject(RevoluteJoint2D(markerNumbers=[mB2R,mB3R]))
        mbs.AddObject(RevoluteJoint2D(markerNumbers=[mB3M,mB1M]))
        mbs.AddObject(RevoluteJoint2D(markerNumbers=[mB4R,mB1R]))
    
        mbs.AddObject(RevoluteJoint2D(markerNumbers=[mB5L,mB3L]))
        mbs.AddObject(RevoluteJoint2D(markerNumbers=[mB5R,mB4L]))
    



    
    cSixBarLinkage.coordinatePoints = pT
    cSixBarLinkage.markersForPrismaticJoints = [mP02,mP14]
    
    if config.simulation.simplifiedLinksIdealCartesianSpringDamper or config.simulation.simplifiedLinksIdealRevoluteJoint2D:
        cSixBarLinkage.markersForConnectors = [mG1,mG2]
    else:
        cSixBarLinkage.markersForConnectors = [mC1,mC2]
        
    cSixBarLinkage.markersForActuatorMeasurement = [mA1,mA2]
    
    #toDo
    cSixBarLinkage.nodes = [nRigid0,nRigid5]
    cSixBarLinkage.objects = [oRigid0,oRigid5]
    # cSixBarLinkage.markers = [mB1L,mB1R,mB1L,mB1R,mB1L,mB1L,mB1L,mG1]  
    cSixBarLinkage.markers = [mG1]  
    # cSixBarLinkage.markersBodyRigid = [mBRR1,mBRR3,mBRR4,mBRR15,mBRR2,mBRR13,mBRR9]      
    cSixBarLinkage.markersViconMeasuringPoints = [mPVicon1,mPVicon2]
    # cSixBarLinkage.markersViconMeasuringPoints = [oRigid7]
    # cSixBarLinkage.markersViconMeasuringPoints = [mPVicon3]
    cSixBarLinkage.markersForFriction = [mPFriction1,mPFriction2]
    return cSixBarLinkage, mbs

def ATC(P1,cATC,mbs,P2=None,P3=None,sideLengths=None,P2Direction = None, connectedSide=None, ID=0, nodeNumber=None):
    """
    Adding an ATC to the mbs 

    Sized either defined through Points P1,P2,P3 or P1,P2Direction and sideLengths

    Orientation defined through connected Side

    """
    L1=cATC.SixBarLinkage[1].SixBarLinkageParameters.L1
    L2=cATC.SixBarLinkage[1].SixBarLinkageParameters.L2
    
    if type(P2) == np.ndarray:


        a=np.linalg.norm(np.array(P3)-np.array(P2))
        b=np.linalg.norm(np.array(P3)-np.array(P1))
        c=np.linalg.norm(np.array(P2)-np.array(P1))
    else:
        a = sideLengths[0,2]
        b = sideLengths[0,1]
        c = sideLengths[0,0]

    

    
    alphaAux=np.arccos((np.square(b)+np.square(c)-np.square(a))/(2*b*c))
    betaAux=np.arccos((np.square(a)+np.square(c)-np.square(b))/(2*a*c))
    gammaAux=np.arccos((np.square(a)+np.square(b)-np.square(c))/(2*a*b))

    if type(P2) != np.ndarray:
        
        n=(P2Direction-P1)/LA.norm(P2Direction-P1)
        P2 = P1+n*c
        P3= P1+b*np.matmul(n,RotAz(-alphaAux))

    if connectedSide == None or connectedSide == 1:
        A = P1
        B = P2
        C = P3

        alpha = alphaAux
        beta = betaAux
        gamma = gammaAux
    elif connectedSide == 2:
        A = P2
        B = P3
        C = P1

        alpha = betaAux
        beta = gammaAux
        gamma = alphaAux
    elif connectedSide == 3:
        A = P3
        B = P1
        C = P2

        alpha = gammaAux
        beta = alphaAux
        gamma = betaAux



    a=np.linalg.norm(np.array(C)-np.array(B))
    b=np.linalg.norm(np.array(C)-np.array(A))
    c=np.linalg.norm(np.array(B)-np.array(A))

   
    AB=B-A

    offsetAngle=np.arctan2(AB[1],AB[0])

    
    if config.simulation.massPoints2DIdealRevoluteJoint2D:
    
        node1 = mbs.AddNode(NodePoint2D(referenceCoordinates=[0,0], initialCoordinates=A))
        body1 = mbs.AddObject(MassPoint2D(physicsMass=10, nodeNumber=node1))
        mBody1 = mbs.AddMarker(MarkerBodyPosition(bodyNumber=body1, localPosition=[0,0,0]))
    
        node2 = mbs.AddNode(NodePoint2D(referenceCoordinates=[0,0], initialCoordinates=B))
        body2 = mbs.AddObject(MassPoint2D(physicsMass=10, nodeNumber=node2))
        mBody2 = mbs.AddMarker(MarkerBodyPosition(bodyNumber=body2, localPosition=[0,0,0]))
    
        node3 = mbs.AddNode(NodePoint2D(referenceCoordinates=[0,0], initialCoordinates=C))
        body3 = mbs.AddObject(MassPoint2D(physicsMass=10, nodeNumber=node3))
        mBody3 = mbs.AddMarker(MarkerBodyPosition(bodyNumber=body3, localPosition=[0,0,0]))
    
    
        La=a#-2*L1-2*L2
        Lb=b#-2*L1-2*L2
        Lc=c#-2*L1-2*L2 
        
        
        #Lref=[Lc,La,Lb]
        #Lref = [La,Lb,Lc]
        #Lref = [La,Lc,Lb]
        Lref = [Lc,Lb,La] 
        kPJ=cATC.ParametersATC.stiffness
        dPJ=cATC.ParametersATC.damping
        
        mFC=[mBody2,mBody1,mBody1,mBody3,mBody3,mBody2]   
        # mFC=[mBody1,mBody2,mBody2,mBody3,mBody3,mBody1]   
        
        # mFPJ=[mBody1,mBody2,mBody2,mBody3,mBody3,mBody1]
        mFPJ=[mBody2,mBody1,mBody1,mBody3,mBody3,mBody2]
    
        springDamperDriveObjects=[]
        for i in range(round(len(mFPJ)/3)+1):  
            #aktorLabel = {'type':'text','text' : 'Side '+str(i),'position':[0.5*Lref[i],-20,0]}
            springDamperDriveObjects+=[mbs.AddObject(SpringDamper(markerNumbers = [mFPJ[i*2],mFPJ[i*2+1]], 
                                            stiffness = kPJ, damping=dPJ, 
                                            referenceLength=abs(Lref[i]),
                                            activeConnector=True,
                                            # visualization=VNodePointGround(show=False),
                                            # springForceUserFunction = UFDrive
                                            ))]    

        jointsObjects = {}
    
        if config.simulation.useRevoluteJoint2DTower:
            jointsObjects[1]=mbs.AddObject(RevoluteJoint2D(markerNumbers=[mFC[0],mFC[1]],activeConnector=False))
            jointsObjects[2]=mbs.AddObject(RevoluteJoint2D(markerNumbers=[mFC[2],mFC[3]],activeConnector=False))
            jointsObjects[3]=mbs.AddObject(RevoluteJoint2D(markerNumbers=[mFC[4],mFC[5]],activeConnector=False))
        else:
            jointsObjects[1]=mbs.AddObject(CartesianSpringDamper(markerNumbers=[mFC[0],mFC[1]],activeConnector=False,stiffness=[kk,kk,0],damping=[dd,dd,0]))
            jointsObjects[2]=mbs.AddObject(CartesianSpringDamper(markerNumbers=[mFC[2],mFC[3]],activeConnector=False,stiffness=[kk,kk,0],damping=[dd,dd,0]))
            jointsObjects[3]=mbs.AddObject(CartesianSpringDamper(markerNumbers=[mFC[4],mFC[5]],activeConnector=False,stiffness=[kk,kk,0],damping=[dd,dd,0]))


        SensorA = mbs.AddSensor(SensorMarker(markerNumber=mBody1, outputVariableType=exu.OutputVariableType.Position))
        SensorB = mbs.AddSensor(SensorMarker(markerNumber=mBody2, outputVariableType=exu.OutputVariableType.Position))
        SensorC = mbs.AddSensor(SensorMarker(markerNumber=mBody3, outputVariableType=exu.OutputVariableType.Position))        
    
        cATC.nodesATC=[node2,node1,node1,node3,node3,node2]
        #add the markes for connectors and prismatic joints (counter clockwise)
        cATC.markersForConnectorsATC = mFC
        cATC.markersForPrismaticJointsATC = mFPJ
        
        #toDo
        pT=list(np.zeros(36))
        pT[0]=list(A)
        pT[12]=list(B)
        pT[24]=list(C)
        cATC.coordinatePointsATC=pT   
        
        
        
        vertices=[mBody1,mBody2,mBody3]
        viconMeasuringPoints=[mBody1,mBody2,mBody3]
        
        cATC.SixBarLinkage[1].centerOfRotation = A
        cATC.SixBarLinkage[2].centerOfRotation = B
        cATC.SixBarLinkage[3].centerOfRotation = C
        
    else:
        cATC.SixBarLinkage[1].centerOfRotation=A
        cATC.SixBarLinkage[1].orientation=offsetAngle
        cATC.SixBarLinkage[1].startAngle=alpha
        
        cATC.SixBarLinkage[2].centerOfRotation=B
        cATC.SixBarLinkage[2].orientation=np.pi-beta+offsetAngle
        cATC.SixBarLinkage[2].startAngle=beta
    
        cATC.SixBarLinkage[3].centerOfRotation=C
        cATC.SixBarLinkage[3].orientation=-(beta+gamma)+offsetAngle
        cATC.SixBarLinkage[3].startAngle=gamma
        
        s1=SixBarLinkage(cATC.SixBarLinkage[1], mbs,2, ID,nodeNumber[ID][0])
        mbs=s1[1]
        s1=s1[0]
        s2=SixBarLinkage(cATC.SixBarLinkage[2], mbs,1, 0,nodeNumber[ID][1])
        mbs=s2[1]
        s2=s2[0]    
        s3=SixBarLinkage(cATC.SixBarLinkage[3], mbs,3, 0,nodeNumber[ID][2])   
        mbs=s3[1]
        s3=s3[0]
        
    
        #markersForConnectorsATC sorted for one ATC (clockwise, start with bottom,right)
        mFC=[s2.markersForConnectors[1],s1.markersForConnectors[0],s1.markersForConnectors[1],s3.markersForConnectors[0],s3.markersForConnectors[1],s2.markersForConnectors[0]]
        #markersForPismaticJointsATC sorted for one ATC (counter clockwise, start with bottom,left)
        mFPJ=[s2.markersForPrismaticJoints[1],s1.markersForPrismaticJoints[0],s1.markersForPrismaticJoints[1],s3.markersForPrismaticJoints[0],s3.markersForPrismaticJoints[1],s2.markersForPrismaticJoints[0]]
    
    
        jointsObjects = {}
    
        if config.simulation.useRevoluteJoint2DTower:
            jointsObjects[1]=mbs.AddObject(RevoluteJoint2D(markerNumbers=[mFC[0],mFC[1]],activeConnector=False))
            jointsObjects[2]=mbs.AddObject(RevoluteJoint2D(markerNumbers=[mFC[2],mFC[3]],activeConnector=False))
            jointsObjects[3]=mbs.AddObject(RevoluteJoint2D(markerNumbers=[mFC[4],mFC[5]],activeConnector=False))
        else:
            jointsObjects[1]=mbs.AddObject(CartesianSpringDamper(markerNumbers=[mFC[0],mFC[1]],activeConnector=False,stiffness=[kk,kk,0],damping=[dd,dd,0],visualization=VNodePointGround(show=False)))
            jointsObjects[2]=mbs.AddObject(CartesianSpringDamper(markerNumbers=[mFC[2],mFC[3]],activeConnector=False,stiffness=[kk,kk,0],damping=[dd,dd,0],visualization=VNodePointGround(show=False)))
            jointsObjects[3]=mbs.AddObject(CartesianSpringDamper(markerNumbers=[mFC[4],mFC[5]],activeConnector=False,stiffness=[kk,kk,0],damping=[dd,dd,0],visualization=VNodePointGround(show=False)))
    
        #define joints+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for i in range(round(len(mFPJ)/3)+1):
            mbs.AddObject(PrismaticJoint2D(markerNumbers=[mFPJ[i*2],mFPJ[i*2+1]],axisMarker0=[1.,0.,0.],normalMarker1=[0.,-1.,0.], constrainRotation=True))
    
    
        La=a-2*L1-2*L2
        Lb=b-2*L1-2*L2
        Lc=c-2*L1-2*L2   
        #Lref=[Lc,La,Lb]
        #Lref = [La,Lb,Lc]
        #Lref = [La,Lc,Lb]
        Lref = [Lc,Lb,La]
        # mbs.variables['LrefExt'] += Lref
        kPJ=cATC.ParametersATC.stiffness
        dPJ=cATC.ParametersATC.damping
        
        springDamperDriveObjects=[]
        for i in range(round(len(mFPJ)/3)+1):  
            #aktorLabel = {'type':'text','text' : 'Side '+str(i),'position':[0.5*Lref[i],-20,0]}
            springDamperDriveObjects+=[mbs.AddObject(SpringDamper(markerNumbers = [mFPJ[i*2],mFPJ[i*2+1]], 
                                            stiffness = kPJ, damping=dPJ, 
                                            referenceLength=abs(Lref[i]),
                                            activeConnector=True,
                                            # springForceUserFunction = UFDrive
                                            ))]
        
        #add the three six-bar linkages
        cATC.SixBarLinkage[1]=s1
        cATC.SixBarLinkage[2]=s2
        cATC.SixBarLinkage[3]=s3
        
        #add the markes for connectors and prismatic joints (counter clockwise)
        cATC.markersForConnectorsATC = mFC
        cATC.markersForPrismaticJointsATC = mFPJ
        
        #toDo
        cATC.coordinatePointsATC=s1.coordinatePoints+s2.coordinatePoints+s3.coordinatePoints
    
        cATC.nodesATC=[s2.nodes[1],s1.nodes[0],s1.nodes[1],s3.nodes[0],s3.nodes[1],s2.nodes[0]]
        # cATC.markersViconMeasuringPoints = [s1.markersViconMeasuringPoints[0],s2.markersViconMeasuringPoints[1],s2.markersViconMeasuringPoints[0],s3.markersViconMeasuringPoints[1],s3.markersViconMeasuringPoints[0],s1.markersViconMeasuringPoints[1]]
        cATC.markersViconMeasuringPoints = [s1.markersViconMeasuringPoints[0],s2.markersViconMeasuringPoints[0],s3.markersViconMeasuringPoints[0]]
    
        SensorA = mbs.AddSensor(SensorMarker(markerNumber=s1.markers[-1], outputVariableType=exu.OutputVariableType.Position))
        SensorB = mbs.AddSensor(SensorMarker(markerNumber=s2.markers[-1], outputVariableType=exu.OutputVariableType.Position))
        SensorC = mbs.AddSensor(SensorMarker(markerNumber=s3.markers[-1], outputVariableType=exu.OutputVariableType.Position))
    
        vertices=[s1.markers[-1],s2.markers[-1],s3.markers[-1]]
        viconMeasuringPoints=[s1.markersViconMeasuringPoints[0],s2.markersViconMeasuringPoints[0],s3.markersViconMeasuringPoints[0]]


        # La=-2*L1-2*L2
        # Lb=-2*L1-2*L2
        # Lc=-2*L1-2*L2   
        # #Lref=[Lc,La,Lb]
        # #Lref = [La,Lb,Lc]
        # #Lref = [La,Lc,Lb]
        # Lref = [Lc,Lb,La] 

    thisdict = {
        "cATC": cATC,
        "Lref": Lref,
        "springDamperDriveObjects": springDamperDriveObjects,
        # "viconMeasuringPoints": [s1.markersViconMeasuringPoints[0],s2.markersViconMeasuringPoints[1],s2.markersViconMeasuringPoints[0],s3.markersViconMeasuringPoints[1],s3.markersViconMeasuringPoints[0],s1.markersViconMeasuringPoints[1]],
        "viconMeasuringPoints": viconMeasuringPoints,
        "connectors":jointsObjects,
        "sensors": [SensorA,SensorB,SensorC]
 
       }
    mbs.variables['Vertices'] += vertices
    mbs.variables['MarkersConnectors'] += [mFC[0],mFC[1],mFC[2],mFC[3],mFC[4],mFC[5]]
    mbs.variables['Sensors'] += [SensorA,SensorB,SensorC]
    mbs.variables['Connectors'] += [jointsObjects[1],jointsObjects[2],jointsObjects[3]]
    return thisdict, mbs

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

def getActuators(xpos, ypos, nrAct, nx,ny, thisdict): # xpos=[1, nx],ypos=[1, ny], nrAct [0-5], nx,ny, thisdict
    """
    Function to get Actuators / old - not in use 
    """
    springDamperDriveObjects = thisdict.get('springDamperDriveObjects')    
    Lref = thisdict.get('Lref')    

    actuatorsList=[]
    LrefList=[]

                            
    for j in range(ypos[1]-ypos[0]+1):
        for i in range(xpos[1]-xpos[0]+1):
            actuatorsList+=[springDamperDriveObjects[nrAct::6][(ypos[0]+j-1)*nx + xpos[0]-1+i   ]]           
            LrefList+=[Lref[nrAct::6][(ypos[0]+j-1)*nx + xpos[0]-1+i   ]]
    
    return actuatorsList, LrefList

def setActuatorLength(actuators, actuatorsDrivesLref,factor,mbs): # actuators to set, leftDrivesLref, factorMultipliedToLength Chi
    """
    Function to set the Actuator Lengths
    """
    mbs.variables['springDamperDriveObjects'] += actuators
    mbs.variables['LrefExt'] += actuatorsDrivesLref
    
    for i in range(int(len(actuators))):
        mbs.variables['springDamperDriveObjectsFactor'] += [factor*1]
    

    return mbs

def getConnectionsOfCells(xpos, ypos, strCon , nx,ny, thisdict): # xpos=nx,ypos=ny, strCon='topCon', nx,ny, thisdict
    """
    Function to get Connection of Cells / old - not in use 
    """
    connectionsOfCells=[-1]
    if strCon == 'topCon' or strCon == 'bottomCon':
        connectionsOfCells=thisdict.get(strCon)[xpos*2+(ypos-1)*(nx*2)-2:xpos*2+(ypos-1)*(nx*2)]  
        
    if strCon == 'sRightCon' or strCon == 'sLeftCon':
        connectionsOfCells=thisdict.get(strCon)[ny*(xpos-1)*2+ypos*2-2:ny*(xpos-1)*2+ypos*2]      
        
    return connectionsOfCells

def GenerateMesh(adeNodes,mbs,P1 =np.array([0,0]),P2 = np.array([1,0]),adeMoveList = None):
    """
    Generate a MBS Mesh of ADEs defined in adeNodes(class, see simulation example Scripts), 
    adeIDList defines the order in which the Mesh gets Build
    Optional:
    P1,P2 Start Point and Orientation of Mesh
    adeMoveList defines Start Lengths

    """
    

    
    
    
    
    ADE = {}
    strokeOffset = config.lengths.L0 
    strokeADE = {}
    connectors = {}


    if type(adeNodes) == dict:
        adeNodesDict = deepcopy(adeNodes)
        adeNodes = deepcopy(adeNodesDict[0])

    adeIDList = []
    for ID in adeNodes.elements:
        adeIDList.append(ID)

    for i in range(1,config.simulation.numberOfBoundaryPoints):
        ADE[-i],mbs = AddBoundary(mbs)


    for ID in adeNodes.elements:
        if adeIDList.index(ID) != 0:   #Setting Point P1 and P2 of the ADEs except the First
            Side = adeNodes.connectionList[ID][0][1]
            Nodes = adeNodes.GetNodeOfElementSide(ID,Side)
            if adeNodes != None:
                P1=adeNodes.nodeDefinition[Nodes[0]]
                P2=adeNodes.nodeDefinition[Nodes[1]]

        else:                       
            Side = adeNodes.GetSideOfElement(ID,adeNodes.baseNodes)
            Nodes = adeNodes.GetNodeOfElementSide(ID,Side)
        #strokes = rotateMatrixRow(strokeADE[ID],Side-1)     # Rotating the strokes, so they correlate to the Sides AB,BC,CA
        
        cADE=CATC()
        setNodes = set(Nodes)
        setNodesOfID = set(adeNodes.elements[ID])
        notinList = setNodesOfID - setNodes
        if next(iter(notinList)) not in adeNodes.nodeDefinition:
            if type(adeMoveList) == df.adeMoveList:
                index = adeMoveList.GetIndexOfID(ID)
                strokeADE[ID] = adeMoveList.data[0,index+1:index+4]/10000+strokeOffset
                # strokeADE[ID] = [adeMoveList.data[0,index+1]/10000+strokeOffset]
                # strokeADE[ID] += [adeMoveList.data[0,index+2]/10000+strokeOffset]
                # strokeADE[ID] += [adeMoveList.data[0,index+3]/10000+strokeOffset]
                strokeADE[ID] = np.matrix(strokeADE[ID])
            else:
                strokeADE[ID] = np.matrix([0,0,0])+strokeOffset
            [ADE[ID],mbs]=ATC(P1,cADE,mbs,sideLengths=strokeADE[ID],P2Direction = P2,connectedSide = Side, ID=ID,nodeNumber=adeNodes.elements)
            # for localIndex, adeNode in enumerate(adeNodes.elements[ID]):
            #     mbsNodeID = ADE[ID]["cATC"].nodesATC[localIndex]
            connectors[ID] = ADE[ID]["connectors"]
        else:
            [ADE[ID],mbs]=ATC(adeNodes.nodeDefinition[Nodes[0]],cADE,mbs,P2 = adeNodes.nodeDefinition[Nodes[1]],P3 = adeNodes.nodeDefinition[next(iter(notinList))],connectedSide = Side, ID=ID,nodeNumber=adeNodes.elements)
            # for localIndex, adeNode in enumerate(adeNodes.elements[ID]):
            #     mbsNodeID = ADE[ID]["cATC"].nodesATC[localIndex]
                # nodeNumber = 14
                # if adeNode == nodeNumber:
                #     print('Node', nodeNumber, 'has node Number', mbsNodeID, 'in mbs')
            connectors[ID] = ADE[ID]["connectors"]

        if adeIDList.index(ID) == 0:   #Storing Points for First ADE
            if adeNodes.elements[ID][0] not in adeNodes.nodeDefinition:
                adeNodes.nodeDefinition[adeNodes.elements[ID][0]]= deepcopy(ADE[ID]["cATC"].SixBarLinkage[1].centerOfRotation)
                adeNodes.nodeDefinition[adeNodes.elements[ID][1]]= deepcopy(ADE[ID]["cATC"].SixBarLinkage[2].centerOfRotation)
                adeNodes.nodeDefinition[adeNodes.elements[ID][2]]= deepcopy(ADE[ID]["cATC"].SixBarLinkage[3].centerOfRotation)
            
            boundaryNodes=[ADE[ID].get('cATC').nodesATC[(Side-1)*2]]+[ADE[ID].get('cATC').nodesATC[(Side-1)*2+1]]
            
            if config.simulation.setMeshWithCoordinateConst:
                nGround=mbs.AddNode(PointGround(referenceCoordinates=ADE[ID].get('cATC').coordinatePointsATC[(Side-1)*12]+[0], visualization=VNodePointGround(show=True)))
                mCoordinateGround = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=nGround, coordinate=0)) #gives always 0 displacement
                
                cooridnateConst=[]
                
                for i in range(round(len(boundaryNodes)/2)):
                    boundaryNodes2=[boundaryNodes[0]]
                    boundaryNodes=[boundaryNodes[1]]
                    
                for i in boundaryNodes:
                    mCoordinateRigid7x = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=i, coordinate=0)) #x-cooridnate of node
                    mCoordinateRigid7y = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=i, coordinate=1)) #y-cooridnate of node
                    
                    if not config.simulation.massPoints2DIdealRevoluteJoint2D:
                        mCoordinateRigid7rot = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=i, coordinate=2)) #phi-cooridnate of node                  
                    
                    cooridnateConst += [mbs.AddObject(CoordinateConstraint(markerNumbers=[mCoordinateGround,mCoordinateRigid7x], offset = 0))]
                    cooridnateConst += [mbs.AddObject(CoordinateConstraint(markerNumbers=[mCoordinateGround,mCoordinateRigid7y], offset = 0))]
                    if not config.simulation.massPoints2DIdealRevoluteJoint2D:
                        cooridnateConst += [mbs.AddObject(CoordinateConstraint(markerNumbers=[mCoordinateGround,mCoordinateRigid7rot], offset = 0))]

                if config.simulation.massPoints2DIdealRevoluteJoint2D:
                    boundaryNodes2=boundaryNodes2[0]
                    # mCoordinateRigid7x = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=i, coordinate=0)) #x-cooridnate of node
                    mCoordinateRigid7y = mbs.AddMarker(MarkerNodeCoordinate(nodeNumber=boundaryNodes2, coordinate=1)) #y-cooridnate of node
                    cooridnateConst += [mbs.AddObject(CoordinateConstraint(markerNumbers=[mCoordinateGround,mCoordinateRigid7y], offset = 0))]








        else:                       #Storing Points for ADEs except the First
            Nodes = adeNodes.GetNodeOfElementSide(ID,adeNodes.connectionList[ID][0][1])
            setNodes = set(Nodes)
            setNodesOfID = set(adeNodes.elements[ID])
            notinList = setNodesOfID - setNodes
            if next(iter(notinList)) not in adeNodes.nodeDefinition:
                if Side == 1:
                    adeNodes.nodeDefinition[notinList.pop()]= deepcopy(ADE[ID]["cATC"].SixBarLinkage[3].centerOfRotation)
                elif Side ==2:
                    adeNodes.nodeDefinition[notinList.pop()]= deepcopy(ADE[ID]["cATC"].SixBarLinkage[2].centerOfRotation)
                elif Side == 3:
                    adeNodes.nodeDefinition[notinList.pop()]= deepcopy(ADE[ID]["cATC"].SixBarLinkage[1].centerOfRotation)
            
       # ########################################################################     
       #  for i in range(len(adeNodes.connectionList[ID])):
       #      if adeNodes.connectionList[ID][i][2] in ADE:
       #          ConnectADE(mbs,ADE,adeNodes.connectionList[ID][i])
       #  ########################################################################         
        # refLength = (config.lengths.L0-2*(config.lengths.L1+config.lengths.L2))
        for i in range(3):
            setActuatorLength([ADE[ID].get('springDamperDriveObjects')[i]], [ADE[ID].get('Lref')[i]], 1, mbs) # actuators to set, leftDrivesLref, factorMultipliedToLength Chi
        
        for Vertex in mbs.variables['Vertices']:
            mbs.variables['VertexLoads'] += [mbs.AddLoad(LoadForceVector(markerNumber=Vertex, loadVector=[0,0,0]))]
            
        for markerConn in mbs.variables['MarkersConnectors']:
            mbs.variables['MarkersConnectorsLoads'] += [mbs.AddLoad(LoadForceVector(markerNumber=markerConn, loadVector=[0,0,0]))]

    return ADE, mbs

def UserFunctionMouseDrag(mbs):
    if not mbs.variables['activateMouseDrag'] == True:
        return 
    p = SC.GetCurrentMouseCoordinates(True)

    mbs.SetNodeParameter(interactivNode,'referenceCoordinates',[p[0],p[1],0])
      
def UserFunctionkeyPress(key, action, mods):
    """
    Changing Position of interactiv Node with Key Press (i = Up, j = Left, k = down, l = right / needs new keys -  l also activates/deactivates visualization of load Vectors) 
    """
    #if chr(key) == 'D' and action == 1: #use capital letters for comparison!!! action 1 == press
    #    mbs.variables['activateMouseDrag'] = not mbs.variables['activateMouseDrag']
    if (chr(key) == 'I' or chr(key) == 'J' or chr(key) == 'K' or chr(key) == 'L') and action==2:
        p = mbs.GetNodeParameter(interactivNode,'referenceCoordinates')
        if chr(key) == 'I': #arrow Up
            p[1] = p[1]+config.simulation.interactivNodeSpeed
        if chr(key) == 'K': #arrow Down
            p[1] = p[1]-config.simulation.interactivNodeSpeed
        if chr(key) == 'J': #arrow Left
            p[0] = p[0]-config.simulation.interactivNodeSpeed
        if chr(key) == 'L': #arrow Right
            p[0] = p[0]+config.simulation.interactivNodeSpeed
        mbs.SetNodeParameter(interactivNode,'referenceCoordinates',p)
        #SC.RenderEngineZoomAll()

#SpringDamper user function
# kDrive = 1e8 #1e4 in ASME paper
# dDrive = 1e4  #1e2 in ASME paper
def ChangeActorStiffnessStartingNode(mbs,t, itemNumber, positionInteractivNode):
    """
    Function to change Actor(itemNumber) Stiffness by distance to interactiv Marker 
    """
    #if not mbs.variables['driveToPoint'] == True:
    #    return 

    
    markerNumbers = mbs.GetObjectParameter(itemNumber,'markerNumbers')
    
    positionInteractivNode = list(positionInteractivNode)[:2]
    # positionInteractivNode=list(mbs.GetSensorValues(mbs.variables['nodeToChangeActorStiffness']))
    # positionInteractivNode=mbs.GetSensorValues(ADE[3]['sensors'][0])
    # positionInteractivNode=[0.4597,0.3958,0]
    positionActor = mbs.GetNodeParameter(mbs.GetObjectParameter(mbs.GetMarkerParameter(markerNumbers[0],'bodyNumber'),'nodeNumber'),'referenceCoordinates')
    positionActor=positionActor[0:2]
    relativPosition = LA.norm(np.array(positionInteractivNode)-np.array(positionActor))
    
    stiffness = CalculateActorStiffnessStartingNode(relativPosition,mbs,itemNumber)
    
    oldStiffness=mbs.GetObjectParameter(itemNumber,'stiffness')
    mbs.SetObjectParameter(itemNumber,'stiffness',oldStiffness+stiffness)
    # print('actor:',itemNumber,'stiffness:',stiffness)
    
def CalculateActorStiffnessStartingNode(relativPosition,mbs,itemNumber):
    """
    Function to Calculate Actor Stiffness by relative Position and Actor Length 
    """
    stiffnessFactor = 1
    # refLength = (config.lengths.L0-2*(config.lengths.L1+config.lengths.L2))
    
    # config.simulation.massPoints2DIdealRevoluteJoint2D = True
    if config.simulation.massPoints2DIdealRevoluteJoint2D:
        refLength = (config.lengths.L0) 
    else:
        refLength = (config.lengths.L0-2*(config.lengths.L1+config.lengths.L2))  
        
        
    actorLength = mbs.GetObjectOutput(itemNumber,  exu.OutputVariableType.Distance) - refLength
        
    L1=config.lengths.L1
    L2=config.lengths.L2
    

    stiffness = stiffnessFactor*relativPosition**2
    if stiffness >= 1e4:
        stiffness = 1e4
    
    if config.simulation.constrainActorLength:
        Lmax = config.lengths.maxStroke
        Lmin = config.lengths.minStroke
        
        if (actorLength <= Lmin) or (actorLength >= Lmax ):
            stiffness = stiffness*1e4

    # print('stiffness:',stiffness)
    return stiffness



def ChangeActorStiffness(mbs,t, itemNumber):
    """
    Function to change Actor(itemNumber) Stiffness by distance to interactiv Marker 
    """
    #if not mbs.variables['driveToPoint'] == True:
    #    return 
    Lref = mbs.GetObjectParameter(itemNumber,'referenceLength')
    
    markerNumbers = mbs.GetObjectParameter(itemNumber,'markerNumbers')
    positionInteractivNode = mbs.GetNodeParameter(interactivNode,'referenceCoordinates')
    
    positionActor = mbs.GetNodeParameter(mbs.GetObjectParameter(mbs.GetMarkerParameter(markerNumbers[0],'bodyNumber'),'nodeNumber'),'referenceCoordinates')
    positionActor[2] = 0.0
    relativPosition = LA.norm(np.array(positionInteractivNode)-np.array(positionActor))
    
    stiffness = CalculateActorStiffness(relativPosition,Lref)
    
    mbs.SetObjectParameter(itemNumber,'stiffness',stiffness)




def ChangeConnectorStiffness(mbs,t, itemNumber):
    """
    Function to change Connector(itemNumber) Stiffness by distance to interactiv Marker 
    """
    #if not mbs.variables['driveToPoint'] == True:
    #    return 
    #LrefExt = mbs.GetObjectParameter(itemNumber,'referenceLength')
    #Lref = max(LrefExt+deltaL,0.095)
    
    markerNumbers = mbs.GetObjectParameter(itemNumber,'markerNumbers')
    positionInteractivNode = np.array(mbs.GetNodeParameter(interactivNode,'referenceCoordinates'))
    positionFirstMarker = np.array(mbs.GetNodeOutput(mbs.GetObjectParameter(mbs.GetMarkerParameter(markerNumbers[0],'bodyNumber'),'nodeNumber'),exu.OutputVariableType.Position))
    positionSecondMarker = np.array(mbs.GetNodeOutput(mbs.GetObjectParameter(mbs.GetMarkerParameter(markerNumbers[1],'bodyNumber'),'nodeNumber'),exu.OutputVariableType.Position))
    positionConnector = (positionFirstMarker+positionSecondMarker)/2
    positionConnector[2] = 0.0
    positionFirstMarker[2] = 0.0
    positionSecondMarker[2] = 0.0

    distanceConnectorMarker = LA.norm(positionFirstMarker-positionSecondMarker)
    if distanceConnectorMarker >= 0.05:
        mbs.SetObjectParameter(itemNumber,'activeConnector',False)
    else: 
        mbs.SetObjectParameter(itemNumber,'activeConnector',True)
    distance = LA.norm(positionInteractivNode-positionConnector)
    distanceX = abs(positionInteractivNode[0]-positionConnector[0])
    distanceY = abs(positionInteractivNode[1]-positionConnector[1])

    n=(positionInteractivNode-positionConnector)/LA.norm(positionInteractivNode-positionConnector)
    
    stiffness = CalculateConnectorStiffness(distance,distanceX,distanceY)

    mbs.SetObjectParameter(itemNumber,'stiffness',[stiffness,stiffness,stiffness])

def AddLoadOnVertices(mbs,t,loadNumber):
    """
    Function to Add Load on Vertices definded by loadNumber 
    """


    markerNumber = mbs.GetLoadParameter(loadNumber,'markerNumber')
    positionInteractivNode = np.array(mbs.GetNodeParameter(interactivNode,'referenceCoordinates'))
    positionVertex = np.array(mbs.GetNodeOutput(mbs.GetObjectParameter(mbs.GetMarkerParameter(markerNumber,'bodyNumber'),'nodeNumber'),exu.OutputVariableType.Position))


    n=(positionInteractivNode-positionVertex)/LA.norm(positionInteractivNode-positionVertex)
    distanceVertexInteractivMarker = LA.norm(positionInteractivNode-positionVertex)
    distanceX = abs(positionInteractivNode[0]-positionVertex[0])
    distanceY = abs(positionInteractivNode[1]-positionVertex[1])

    loadVector = CalculateLoadVector(n,distanceVertexInteractivMarker,distanceX,distanceY)

    mbs.SetLoadParameter(loadNumber,'loadVector',loadVector)

def phi(t):
    return 3*t**2 -2*t**3

def CalculateActorStiffness(relativPosition,L):
    """
    Function to Calculate Actor Stiffness by relative Position and Actor Length 
    """
    stiffnessFactor = 1e2
    L1 = ADE[1]["cATC"].SixBarLinkage[1].SixBarLinkageParameters.L1
    L2 = ADE[1]["cATC"].SixBarLinkage[1].SixBarLinkageParameters.L2

    Lmax = 0.330
    Lmin = 0.229
    L = L+2*(L1+L2)
    if (L <= Lmin + 0.010) and (L > Lmin):
        stiffness = stiffnessFactor*(relativPosition*phi((abs(L-Lmin))/(0.010)))
    elif (L >= Lmax -0.010) and ((L < Lmax )):
        stiffness = stiffnessFactor*(relativPosition*phi((abs(L-Lmax))/(0.010)))
    elif (L <= Lmin) or (L >= Lmax ):
        stiffness = stiffnessFactor*10000
    else: 
        stiffness = stiffnessFactor*(relativPosition+1)
    return stiffness

def CalculateConnectorStiffness(distance,distanceX,distanceY):
    """
    Function to Calculate Connector Stiffness by distance to interactiv Marker 
    """
    stiffnessFactor = 1e4

    if distance<=3*config.simulation.radiusInteractivMarker and distanceY<=0.3*config.simulation.radiusInteractivMarker:
        stiffness = stiffnessFactor
    if distance<=2*config.simulation.radiusInteractivMarker and distanceY<=0.2*config.simulation.radiusInteractivMarker:
        stiffness = stiffnessFactor/2
    if distance<=1*config.simulation.radiusInteractivMarker and distanceY<=0.1*config.simulation.radiusInteractivMarker:
        stiffness = stiffnessFactor/4
    else:
        stiffness = stiffnessFactor*2
    return stiffness

def CalculateLoadVector(n,distance,distanceX,distanceY):
    """
    Function to calculate Load Vector by distance to interactiv Marker and normal Vector n
    """
    loadFactor = 1e7
    distanceFactor = 5

    if distance<=distanceFactor*config.simulation.radiusInteractivMarker:
        loadVector = -loadFactor*phi((distanceFactor*config.simulation.radiusInteractivMarker-distance)/distanceFactor*config.simulation.radiusInteractivMarker)*n
        if distanceY <= config.lengths.L0/2000 and distanceY < distanceX:
            loadVector[0] = 0
        if distanceX <= config.lengths.L0/2000 and distanceX < distanceY:
            loadVector[0] = 0
    else:
        loadVector = [0,0,0]
    return loadVector

def UserFunctionDriveRefLen1(t, u1, u2):
    """
    Function to Drive Actors from u1 to u2
    """
    if config.simulation.SolveDynamic:
        # dT = abs(u2-u1)/(config.com.actorSpeed)      #end time
        dT = abs(u2-u1)/(config.com.actorSpeed)*1000
        # dT = config.simulation.endTime
    else:
        dT = 1
        
    t0 = 0      #start time
    l0 = u1      #initial length
    l1 = u2    #final length
    lDrive = l0     
    if t>t0:
        if t < t0+dT:
            lDrive = l0 + (l1-l0) * 0.5*(1-np.cos((t-t0)/dT*np.pi)) #t=[0 .. 25]
        else:
            lDrive = l1                 
    return lDrive

#user function called at beginning of every time step
def PreStepUserFunction(mbs, t):
    """
    Prestep User Function 
    """
    #print("called")
    #print("mbs=", mbs2)
    LrefExt = mbs.variables['LrefExt']
    Actors = mbs.variables['springDamperDriveObjects']
    Connectors = mbs.variables['Connectors']
    Vertices = mbs.variables['VertexLoads']
    TimeStamp = mbs.variables['TimeStamp']
    line = mbs.variables['line']

    #mbs.WaitForUserToContinue()

    ##+++++++++++++++++++++++++++++++++++++++
    ## Changing of Actor Stiffness if activated and Interactiv marker is activated
    if config.simulation.changeActorStiffness and config.simulation.addInteractivMarker:
        for itemNumber in Actors:
            ChangeActorStiffness(mbs,t,itemNumber)

    ##+++++++++++++++++++++++++++++++++++++++
    ## Changing of Connector Stiffness if activated and Interactiv marker is activated
    if config.simulation.changeConnectorStiffness and config.simulation.addInteractivMarker:
        for itemNumber in Connectors:
            ChangeConnectorStiffness(mbs,t,itemNumber)

    ##+++++++++++++++++++++++++++++++++++++++
    ## Adding load to Vertices of Mesh if activated and Interactiv marker is activated
    if config.simulation.addLoadOnVertices and config.simulation.addInteractivMarker:
        for markerNumber in Vertices:
            AddLoadOnVertices(mbs,t,markerNumber)


    SensorValuesTime[line][t]={}
    
    SensorValues = {} 
    for ID in adeIDs:
        SensorValues[ID] = np.asmatrix([mbs.GetSensorValues(ADE[ID]['sensors'][0])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][1])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][2])[0:2]])   
    
    SensorValuesTime[line][t]=SensorValues



    ##+++++++++++++++++++++++++++++++++++++++
    ## Driving ADEs acording to MoveList if Activated
    if config.simulation.useMoveList:
        refLength = (config.lengths.L0-2*(config.lengths.L1+config.lengths.L2))
        #nextTimeStamp = False
        for i in range(len(adeIDs)):
            index = adeMoveList.GetIndexOfID(adeIDs[i])
            if index != None:
                # Drive Actors:
                for j in range(3):
                    oSpringDamper = mbs.variables['springDamperDriveObjects'][i*3+j]
                    #u1 = mbs.GetObjectParameter(oSpringDamper,'referenceLength')-refLength
                    u1 = adeMoveList.data[cnt,index+j+1]/10000#-adeMoveList.data[0,index+j+1]/10000
                    #u1 = LrefExt[i*3+j]
                    u2 = adeMoveList.data[cnt+1,index+j+1]/10000#-adeMoveList.data[0,index+j+1]/10000
                    oSpringDamper = mbs.variables['springDamperDriveObjects'][i*3+j]
                    driveLen = UserFunctionDriveRefLen1(t, u1,u2)
                    mbs.SetObjectParameter(oSpringDamper,'referenceLength',driveLen+refLength)       
    return True #succeeded

def addContactSpheres(gContact,ADE,sidesOfNodes,sizeOfSpheres,contactStiffness,dContact,frictionMaterialIndex):    
    j=0
    for i in sidesOfNodes[::2]:     
        markerNumbersFirstADE = ADE[i].get('cATC').markersForConnectorsATC[(sidesOfNodes[1::2][j]-1)*2:(sidesOfNodes[1::2][j]-1)*2+2]
        j+=1
        
        gContact.AddSphereWithMarker(markerNumbersFirstADE[0], radius=sizeOfSpheres, contactStiffness=contactStiffness, contactDamping=dContact, frictionMaterialIndex=frictionMaterialIndex)
        gContact.AddSphereWithMarker(markerNumbersFirstADE[1], radius=sizeOfSpheres, contactStiffness=contactStiffness, contactDamping=dContact, frictionMaterialIndex=MaterialIndex) 
        


def DisconnectADE(mbs,ADE,connectionEntry):
    """
    Disconnecting ADE with old connection Entry 
    """

    markerNumbersFirstADE = ADE[connectionEntry[0]].get('cATC').markersForConnectorsATC[(connectionEntry[1]-1)*2:(connectionEntry[1]-1)*2+2]
    markerNumbersSecondADE = ADE[connectionEntry[2]].get('cATC').markersForConnectorsATC[(connectionEntry[3]-1)*2:(connectionEntry[3]-1)*2+2]

    mbs.SetObjectParameter(ADE[connectionEntry[0]]["connectors"][connectionEntry[1]],'activeConnector',False)
    mbs.SetObjectParameter(ADE[connectionEntry[2]]["connectors"][connectionEntry[3]],'activeConnector',False)


    mbs.SetObjectParameter(ADE[connectionEntry[2]]["connectors"][connectionEntry[3]],'markerNumbers',[markerNumbersFirstADE[0],markerNumbersFirstADE[1]])
    mbs.SetObjectParameter(ADE[connectionEntry[0]]["connectors"][connectionEntry[1]],'markerNumbers',[markerNumbersSecondADE[0],markerNumbersSecondADE[1]])

    if connectionEntry[2] == -1 and config.simulation.setMeshWithCoordinateConst==False:
        mbs.SetObjectParameter(ADE[connectionEntry[0]]["connectors"][connectionEntry[1]],'offset',[0,0,0])
        mbs.SetObjectParameter(ADE[connectionEntry[2]]["connectors"][connectionEntry[3]],'offset',[0,0,0])
   
def ConnectADE(mbs,ADE,connectionEntry):
    """
    Connecting ADE with new connection Entry 
    """

    markerNumbersFirstADE = ADE[connectionEntry[0]].get('cATC').markersForConnectorsATC[(connectionEntry[1]-1)*2:(connectionEntry[1]-1)*2+2]
    markerNumbersSecondADE = ADE[connectionEntry[2]].get('cATC').markersForConnectorsATC[(connectionEntry[3]-1)*2:(connectionEntry[3]-1)*2+2]

    mbs.SetObjectParameter(ADE[connectionEntry[2]]["connectors"][connectionEntry[3]],'markerNumbers',[markerNumbersFirstADE[0],markerNumbersSecondADE[1]])
    mbs.SetObjectParameter(ADE[connectionEntry[0]]["connectors"][connectionEntry[1]],'markerNumbers',[markerNumbersSecondADE[0],markerNumbersFirstADE[1]])
    
    mbs.SetObjectParameter(ADE[connectionEntry[0]]["connectors"][connectionEntry[1]],'activeConnector',True)
    mbs.SetObjectParameter(ADE[connectionEntry[2]]["connectors"][connectionEntry[3]],'activeConnector',True)

    if connectionEntry[2] <= -1 and config.simulation.setMeshWithCoordinateConst==False:
        #offset1 = (np.array(mbs.GetNodeParameter(mbs.GetObjectParameter(mbs.GetMarkerParameter(markerNumbersFirstADE[0], 'bodyNumber'), 'nodeNumber'),'referenceCoordinates'))-np.array(mbs.GetMarkerParameter(markerNumbersFirstADE[0], 'localPosition')))-np.array(mbs.GetMarkerOutput(markerNumbersSecondADE[1],exu.OutputVariableType.Position))
        #offset2 = np.array(mbs.GetMarkerOutput(markerNumbersSecondADE[0],exu.OutputVariableType.Position)) - (np.array(mbs.GetNodeParameter(mbs.GetObjectParameter(mbs.GetMarkerParameter(markerNumbersFirstADE[1], 'bodyNumber'), 'nodeNumber'),'referenceCoordinates'))+np.array(mbs.GetMarkerParameter(markerNumbersFirstADE[1], 'localPosition')))
        
        offset1 = np.array(mbs.GetMarkerOutput(markerNumbersFirstADE[0],exu.OutputVariableType.Position,configuration = exu.ConfigurationType.Current))-np.array(mbs.GetMarkerOutput(markerNumbersSecondADE[1],exu.OutputVariableType.Position,configuration = exu.ConfigurationType.Current))
        offset2 = np.array(mbs.GetMarkerOutput(markerNumbersSecondADE[0],exu.OutputVariableType.Position,configuration = exu.ConfigurationType.Current)) - np.array(mbs.GetMarkerOutput(markerNumbersFirstADE[1],exu.OutputVariableType.Position,configuration = exu.ConfigurationType.Current))
        
        offset1[2] = 0
        offset2[2] = 0
        mbs.SetObjectParameter(ADE[connectionEntry[2]]["connectors"][connectionEntry[3]],'offset',-offset1)
        mbs.SetObjectParameter(ADE[connectionEntry[0]]["connectors"][connectionEntry[1]],'offset',-offset2)

def AddBoundary(mbs,P1 =np.array([0,0]),P2 = np.array([1,0])):

    thisADE = {
        "cATC": CATC(),
        "Lref": [],
        "springDamperDriveObjects":[],
        # "viconMeasuringPoints": [s1.markersViconMeasuringPoints[0],s2.markersViconMeasuringPoints[1],s2.markersViconMeasuringPoints[0],s3.markersViconMeasuringPoints[1],s3.markersViconMeasuringPoints[0],s1.markersViconMeasuringPoints[1]],
        "viconMeasuringPoints": [],
        "connectors":[],
        "sensors": []
 
       }

    if len(P1) == 2:
        P1 = np.append(P1,[0])
    if len(P2) == 2:
        P2 = np.append(P2,[0])
    
    nGround=mbs.AddNode(PointGround(referenceCoordinates=P1, visualization=VNodePointGround(show=False)))
    nGround2=mbs.AddNode(PointGround(referenceCoordinates=P2, visualization=VNodePointGround(show=False)))
    

    mCoordinateGround = mbs.AddMarker(MarkerNodePosition(nodeNumber=nGround))
    mCoordinateGround2 = mbs.AddMarker(MarkerNodePosition(nodeNumber=nGround2))  
                
    

    jointsObjects=mbs.AddObject(CartesianSpringDamper(markerNumbers=[mCoordinateGround,mCoordinateGround2], visualization=VNodePointGround(show=False),activeConnector=False,stiffness=[config.lengths.stiffnessBoundary,config.lengths.stiffnessBoundary,0],damping=[config.lengths.dampingBoundary,config.lengths.dampingBoundary,0]))

    thisADE['cATC'].markersForConnectorsATC=[mCoordinateGround,mCoordinateGround2]
    thisADE['connectors'] = {1:jointsObjects}
    return thisADE, mbs

def findNodesInRectangle(adeNodes, P1 =np.array([0,0]), P2 = np.array([1,1])):
    """
    function to find Nodes in a defined Rectangle with bottom left P1 and top
    right P2
    ---------P2
    | node1  |
    |   |    |
    | node2  |
    P1-------
    Parameters
    ----------
    adeNodes : TYPE
        DESCRIPTION.
    P1 : np.array([0,0])
        bottom left point of rectangle. The default is np.array([0,0]).
    P2 : np.array([1,1])
        top right of rectangle. The default is np.array([1,1]).

    Returns
    -------
    nodesInRectangle : list() [node1, node2, ... ,nodeN]
        contains all nodes in the rectangle which lie within the points P1 and P2

    """
    nodesInRectangle=[]
    for ID in adeNodes.nodeDefinition:
        if adeNodes.nodeDefinition[ID][0] >= P1[0] and adeNodes.nodeDefinition[ID][0] <= P2[0] and adeNodes.nodeDefinition[ID][1] >= P1[1] and adeNodes.nodeDefinition[ID][1] <= P2[1]:
            nodesInRectangle += [ID]
    return nodesInRectangle


def getSidesOfNodes(adeNodes, nodes):
    """
    function to get all connecting sides between nodes

    Parameters
    ----------
    adeNodes : TYPE
        DESCRIPTION.
    nodes : [node1, node2, ... ,nodeN]
        contains all nodes

    Returns
    -------
    sides : list() [ID, Side, ..., IDn, SideN]
        Contains the ID number of the ATCs and the associated sides that are 
        defined by the nodes passed.

    """
    sides=[]
    for ID in adeNodes.elements: #go through each entry of the element list
        nodesIntersect = set(adeNodes.elements[ID]).intersection(set(nodes)) #check if element list conatin some of the given nodes, returns set{} if nothing is found, set{N1, N2} if two are the same and set{N1,N2,N3} if all nodes are found
        if len(nodesIntersect) == 2: #if nodesIntersect conatins two entries, return the side number between the two nodes
            side = adeNodes.GetSideOfElement(ID,list(nodesIntersect))
            # nodes = adeNodes.GetNodeOfElementSide(ID,side)
            sides+=[ID,side]
        elif len(nodesIntersect) == 3: #if nodesIntersect contains three entries, return all sides
              sides+=[ID,1,ID,2,ID,3]     
    return sides

def GenerateGridSimulation(nx,ny,adeNodes):
    # nx = 6
    # ny = 20

    # adeNodes = df.ElementDefinition()

    height = config.lengths.L0
    width = config.lengths.L0

    nodes = [*range(1,nx+2)]
    for j,node in enumerate(nodes):
            adeNodes.nodeDefinition[node] = np.array([width*j,0])

    for i in range(ny+1):
        nodes = [*range((nx+1)*(i+1)+1,(nx+1)*(i+2)+1)]
        for j,node in enumerate(nodes):
            adeNodes.nodeDefinition[node] = np.array([width*j,height*(i+1)])
        for j in range(nx):
            """
            N3------N4
            |       |
            |       |
            |       |
            N1------N2
            """
            N1 = (i*(nx+1))+j+1
            N2 = (i*(nx+1))+j+2
            N3 = (i+1)*(nx+1)+j+1
            N4 = (i+1)*(nx+1)+j+2
            ID1 = (i*(nx*2))+(j*2)+1
            ID2 = (i*(nx*2))+(j*2)+2
            if (i%2 == 0 and j%2 == 0) or (i%2 != 0 and j%2 != 0) :
                """
                N3------N4
                | \  ID2|
                |   \   |
                | ID1 \ |
                N1------N2
                """
                adeNodes.AddElementNodes(ID1,[N1,N2,N3],setLists = False)
                adeNodes.AddElementNodes(ID2,[N2,N4,N3],setLists = False)
            else:
                """
                N3------N4
                | ID1 / |
                |   /   |
                | / ID2 |
                N1------N2
                """
                adeNodes.AddElementNodes(ID1,[N1,N4,N3],setLists = False)
                adeNodes.AddElementNodes(ID2,[N1,N2,N4],setLists = False)
    adeNodes.SetListOfNodes()
    adeNodes.SetConnectionList()
    adeNodes.SetNodeNumbersToAde()
    adeNodes.SetADEPointsOfNode()
    # client.Put('adeNodes',pickle.dumps(adeNodes))
    # client.Put('StartSimulation',json.dumps(True))

def addBoundaryWithNodesInRectangle(adeNodes,nodesInRectangle):
    
    indexID=0
    for index in adeNodes.connectionList:
        for lenRange in range(len(adeNodes.connectionList[index])):
            minVal=min(adeNodes.connectionList[index][lenRange])
            if minVal<indexID:
                indexID=minVal
    indexID=indexID-1
    for i in range(len(nodesInRectangle)-1):
        IDSide=getSidesOfNodes(adeNodes, [nodesInRectangle[i],nodesInRectangle[i+1]])
        adeNodes.connectionList[IDSide[0]] += [[IDSide[0], IDSide[1], indexID, 1]]
        indexID=indexID-1
    
    
# if __name__ == "__main__":
#     ##+++++++++++++++++++++++++++++++++++++++
#     ## Initialization of Module
#     config.init()
#     logging.info('Starte Simulations Module')

#     client = wf.WebServerClient()
#     SC = exu.SystemContainer()
#     mbs = SC.AddSystem()

#     computeDynamic=True
    
#     ##simulation options
#     displaySimulation=config.simulation.displaySimulation
    
#     ##++++++++++++++++++++++++++++++
#     ##ANIMATIONS
#     ##make images for animations (requires FFMPEG):
    


#     startSimulation = False
#     while not startSimulation:
        
#         try:
#             ##+++++++++++++++++++++++++++++++++++++++
#             ## Checking if to end the Programm

#             if json.loads(client.Get('EndProgram')):
#                 os._exit(0)
#             ##+++++++++++++++++++++++++++++++++++++++
#             ## Initialization of Variables

#             mbs.variables['springDamperDriveObjects']=[] #all drives that need to be changed over time
#             mbs.variables['springDamperDriveObjectsFactor']=[] #factor for drives: depends on distance to neutral fiber
#             mbs.variables['LrefExt']=[]
#             mbs.variables['Connectors'] = []
#             mbs.variables['Vertices'] = []
#             mbs.variables['VertexLoads'] = []
#             mbs.variables['TimeStamp'] = 0
#             mbs.variables['nextTimeStamp'] = False
#             mbs.variables['Sensors'] = []
#             mbs.variables['activateMouseDrag'] = False
#             mbs.variables['driveToPoint'] = False
#             mbs.variables['line'] = 0
#             mbs.variables['MarkersConnectors']=[]
#             mbs.variables['MarkersConnectorsLoads']=[]
#             if config.simulation.activateWithKeyPress:
#                 mbs.variables['activateMouseDrag'] = False
            
#             mbs.variables['xValue']=[]

#             ##+++++++++++++++++++++++++++++++++++++++
#             ## Getting Variables from Webserver
#             adeNodes =  pickle.loads(client.Get('adeNodes'))
#             adeIDs = []

#             if type(adeNodes) == dict:
#                 for ID in adeNodes[0].elements:
#                         adeIDs.append(ID)
#             else:
#                 for ID in adeNodes.elements:
#                         adeIDs.append(ID)

#             startSimulation = json.loads(client.Get('StartSimulation'))



#             if config.simulation.useMoveList:
#                 adeMoveList = df.adeMoveList()
#                 adeMoveList.FromJson(client.Get('MoveList'))

#             ##+++++++++++++++++++++++++++++++++++++++
#             ## Starting preparations for Simulation

#             if startSimulation and (type(adeNodes) == df.ElementDefinition or type(adeNodes) == dict):
#                 if config.simulation.useMoveList: 
#                     ADE, mbs = GenerateMesh(adeNodes,mbs,adeMoveList = adeMoveList)         #Generating Mesh with Movelist
#                 else:
#                     ADE, mbs = GenerateMesh(adeNodes,mbs)                                   #Generatin Mesh without Movelist

#                 ##+++++++++++++++++++++++++++++++++++++++
#                 ## Setting up Interactiv Marker
#                 if config.simulation.addInteractivMarker:
#                     gCylinder = GraphicsDataCylinder(pAxis=[0,0,0],vAxis=[0,0,0.001], radius=config.simulation.radiusInteractivMarker,
#                                                       color= color4dodgerblue, nTiles=64)

#                     interactivNode = mbs.AddNode({'nodeType': 'PointGround',
#                                                     'referenceCoordinates':config.simulation.interactivMarkerPosition,
#                                                     'initialCoordinates': [0.0, 0.0, 0.0],
#                                                     'name': 'Interactiv Node'})
                    
                    
#                     interactivMassPoint = mbs.AddObject(MassPoint(physicsMass = 1, nodeNumber = interactivNode, 
#                                                     visualization=VObjectMassPoint(graphicsData=[gCylinder]),name = 'Interactiv Mass Point'))
#                     interactivMarker =  mbs.AddMarker({'markerType': 'NodePosition',
#                                                         'nodeNumber': interactivNode,
#                                                         'name': 'Interactiv Marker'})

#                     ##+++++++++++++++++++++++++++++++++++++++
#                     ## Adding Spring to interactiv Marker to drag the Mesh
#                     if config.simulation.addSpringToInteractivMarker:
#                             mbs.AddObject(CartesianSpringDamper(markerNumbers = [interactivMarker,ADE[3].get('cATC').SixBarLinkage[1].markers[-1]], 
#                                                         stiffness = [1e8,1e8,1e8],
#                                                         damping = [1e8,1e8,1e8],offset = [0,0,0]
#                                                         ))


#                 ##+++++++++++++++++++++++++++++++++++++++
#                 ## Simulation Settings

#                 simulationSettings = exu.SimulationSettings() #takes currently set values or default values
#                 # simulationSettings.timeIntegration.preStepPyExecute = 'MBSUserFunction()'
#                 T=0.002
#                 SC.visualizationSettings.connectors.defaultSize = T
#                 SC.visualizationSettings.bodies.defaultSize = [T, T, T]
#                 SC.visualizationSettings.nodes.defaultSize = 0.0025
#                 SC.visualizationSettings.markers.defaultSize = 0.005
#                 SC.visualizationSettings.loads.defaultSize = 0.015
#                 SC.visualizationSettings.general.autoFitScene=True
                
                
                
                
                
                
                
                
                
                
#                 SC.visualizationSettings.sensors.traces.showPositionTrace = True
                
                
                
                
                
                
                
                
                
                
                
#                 SC.visualizationSettings.nodes.show= True
#                 SC.visualizationSettings.markers.show= False
#                 SC.visualizationSettings.connectors.show= False
                
#                 SC.visualizationSettings.openGL.lineWidth=2 #maximum
#                 SC.visualizationSettings.openGL.lineSmooth=True
#                 SC.visualizationSettings.openGL.multiSampling = 4
#                 SC.visualizationSettings.general.drawCoordinateSystem = False
#                 SC.visualizationSettings.general.textSize = 16
#                 if not config.simulation.showLabeling:
#                     SC.visualizationSettings.general.showSolverTime = False
#                     SC.visualizationSettings.general.showSolverInformation = False
#                     SC.visualizationSettings.general.showSolutionInformation = False
                
#                 #SC.visualizationSettings.window.renderWindowSize=[1600,1024]
#                 SC.visualizationSettings.window.renderWindowSize=[1600,1000]
                
                
#                 ##++++++++++++++++++++++++++++++
#                 ##ANIMATIONS
#                 ##make images for animations (requires FFMPEG):
#                 ##requires a subfolder 'images'
#                 if config.simulation.animation:
#                     simulationSettings.solutionSettings.recordImagesInterval=0.04 #simulation.endTime/200
                
                
#                 if config.simulation.activateWithKeyPress:
#                     SC.visualizationSettings.window.keyPressUserFunction = UserFunctionkeyPress
                
#                 if config.simulation.saveImages:
#                     SC.visualizationSettings.exportImages.saveImageFileName = config.simulation.imageFilename
#                     SC.visualizationSettings.exportImages.saveImageFormat = "PNG"
#                     SC.visualizationSettings.exportImages.saveImageSingleFile=False
                    


                    
#                     if config.simulation.showJointAxes:
#                         SC.visualizationSettings.connectors.defaultSize = 0.002
#                         SC.visualizationSettings.connectors.showJointAxes = True
#                         # SC.visualizationSettings.connectors.jointAxesLength = 0.008#0.0015
#                         # SC.visualizationSettings.connectors.jointAxesRadius = 0.0025#0.0008
#                         SC.visualizationSettings.connectors.jointAxesLength = 0.0015
#                         SC.visualizationSettings.connectors.jointAxesRadius = 0.0008

                
            

#                 ##+++++++++++++++++++++++++++++++++++++++
#                 ## Assemble Mesh 
#                 mbs.Assemble()
#                 sysStateList = mbs.systemData.GetSystemState()

                
#                 if displaySimulation:
#                     exu.StartRenderer()
#                     mbs.WaitForUserToContinue()

#                 if type(adeNodes) == dict:
#                     for ID in adeNodes[0].elements:
#                         for i in range(len(adeNodes[0].connectionList[ID])):
#                             if adeNodes[0].connectionList[ID][i][2] in ADE:
#                                 ConnectADE(mbs,ADE,adeNodes[0].connectionList[ID][i])
#                 else:
#                     for ID in adeNodes.elements:
#                         for i in range(len(adeNodes.connectionList[ID])):
#                             if adeNodes.connectionList[ID][i][2] in ADE:
#                                 ConnectADE(mbs,ADE,adeNodes.connectionList[ID][i])
#                 mbs.AssembleLTGLists()
#                 mbs.systemData.SetSystemState(sysStateList,configuration = exu.ConfigurationType.Initial)

#                 ##+++++++++++++++++++++++++++++++++++++++
#                 ## Setting um SensorValues and Sending initial Sensor Values (Mesh Node Coordinates)
#                 SensorValuesTime = {} 
#                 SensorValuesTime[0] = {}  
                
#                 SensorValues = {} 
#                 SensorValues[0] = {}
#                 for ID in adeIDs:
#                     SensorValues[0][ID] = np.asmatrix([mbs.GetSensorValues(ADE[ID]['sensors'][0])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][1])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][2])[0:2]])
#                 client.Put('SensorValues',pickle.dumps(SensorValues))
    
#                 SensorValuesTime[0]=SensorValues.copy()

#                 # calculate in the first step the static solution
#                 simulationSettings.staticSolver.newton.numericalDifferentiation.relativeEpsilon = 1e-4
#                 newResidual = True
#                 if not(newResidual):
#                     simulationSettings.staticSolver.newton.relativeTolerance = 1e-10
#                     simulationSettings.staticSolver.newton.absoluteTolerance = 1e-12
#                 else:
#                     #new solver settings, work for all 2x7 configurations with eps=1e-4
#                     simulationSettings.staticSolver.newton.relativeTolerance = 1e-5
#                     simulationSettings.staticSolver.newton.absoluteTolerance = 1e-5
#                     simulationSettings.staticSolver.newton.newtonResidualMode = 1 #take Newton increment
                
#                 simulationSettings.staticSolver.stabilizerODE2term = 1
                
#                 simulationSettings.staticSolver.newton.numericalDifferentiation.relativeEpsilon = 1e-9
#                 simulationSettings.staticSolver.newton.maxIterations = 250
                    
#                 simulationSettings.linearSolverType = exu.LinearSolverType.EigenSparse
#                 simulationSettings.staticSolver.newton.weightTolerancePerCoordinate = True   
#                 simulationSettings.solutionSettings.solutionInformation = "PARTS_StaticSolution"

                
#                 numberOfLoadSteps=10
#                 simulationSettings.staticSolver.numberOfLoadSteps = numberOfLoadSteps
                
#                 exu.SolveStatic(mbs, simulationSettings=simulationSettings)

#                 if config.simulation.saveImages:
#                     SC.RedrawAndSaveImage()
                
#                 solODE2initial = mbs.systemData.GetODE2Coordinates(configuration=exu.ConfigurationType.Current)
#                 mbs.systemData.SetODE2Coordinates(solODE2initial, configuration=exu.ConfigurationType.Initial)  #set old solution as initial value for new solution
                 
#                 mbs.WaitForUserToContinue()       
                
#                 # mbs.SetPreStepUserFunction(PreStepUserFunction)

#                 # simulationSettings.timeIntegration.numberOfSteps = config.simulation.nrSteps
#                 # simulationSettings.timeIntegration.endTime = config.simulation.endTime

#                 simulationSettings.timeIntegration.newton.absoluteTolerance = 1e3
#                 simulationSettings.timeIntegration.newton.numericalDifferentiation.minimumCoordinateSize = 1
            
#                 simulationSettings.timeIntegration.verboseMode = 0
#                 #simulationSettings.timeIntegration.newton.useNumericalDifferentiation = True
#                 #simulationSettings.timeIntegration.newton.numericalDifferentiation.doSystemWideDifferentiation = True
                
#                 simulationSettings.timeIntegration.generalizedAlpha.computeInitialAccelerations = False
#                 simulationSettings.timeIntegration.newton.useModifiedNewton = False #JG
#                 simulationSettings.timeIntegration.generalizedAlpha.spectralRadius = 0.4
#                 simulationSettings.timeIntegration.adaptiveStep = True #disable adaptive step reduction

                
                
#                 ##############################################################
#                 # IMPORTANT!!!!!!!!!
#                 simulationSettings.linearSolverType = exu.LinearSolverType.EigenSparse #sparse solver !!!!!!!!!!!!!!!
#                 ##############################################################
#                 simulationSettings.displayStatistics = False       
#             #        simulationSettings.timeIntegration.preStepPyExecute = "UserFunction()\n" #NEU?
#                 if False: #JG
#                     simulationSettings.timeIntegration.generalizedAlpha.useIndex2Constraints = True
#                     simulationSettings.timeIntegration.generalizedAlpha.useNewmark= True


#                 if config.simulation.solutionViewer: #use this to reload the solution and use SolutionViewer
#                     SC.visualizationSettings.general.autoFitScene=False #if reloaded view settings
#                     simulationSettings.solutionSettings.solutionWritePeriod = config.simulation.solutionWritePeriod
#                     simulationSettings.solutionSettings.writeSolutionToFile = True
#                     simulationSettings.solutionSettings.coordinatesSolutionFileName = config.simulation.solutionViewerFile
#                     simulationSettings.solutionSettings.writeFileFooter = False
#                     simulationSettings.solutionSettings.writeFileHeader = True
#                     if os.path.isfile(solutionFilePath):
#                         os.remove(solutionFilePath)
#                     simulationSettings.solutionSettings.appendToFile = True



#                 ##+++++++++++++++++++++++++++++++++++++++
#                 ## Starting Simulation using Movelist
#                 if config.simulation.useMoveList:
#                     ##+++++++++++++++++++++++++++++++++++++++
#                     ## Checking and Setting Connections of ADEs
#                     TimeStamp = 0
                    

#                     for i in range(len(adeMoveList.data)-1):
#                         mbs.variables['line']=i+1
#                         SensorValuesTime[i+1]={}
                        
#                         if i == len(adeMoveList.data)-2 and config.simulation.solutionViewer:
#                             simulationSettings.solutionSettings.writeFileFooter = True
#                             simulationSettings.solutionSettings.writeFileHeader = False
#                             if i == 0:
#                                 simulationSettings.solutionSettings.writeFileHeader = True


#                         nextTimeStamp = False
#                         cnt=i
                   
#                         if cnt >= 0:
#                             ##+++++++++++++++++++++++++++++++++++++++
#                             ## Checking if Connections changed with the MoveList
#                             for j in range(len(adeIDs)):
#                                 index = adeMoveList.GetIndexOfID(adeIDs[j])
#                                 if index != None:
#                                     if config.simulation.connectionInfoInMoveList:
#                                         if adeMoveList.data[cnt,index+4] != adeMoveList.data[cnt+1,index+4] or adeMoveList.data[cnt,index+5] != adeMoveList.data[cnt+1,index+5] or adeMoveList.data[cnt,index+6] != adeMoveList.data[cnt+1,index+6]:
#                                             nextTimeStamp = True
#                                     else:
#                                         if  adeMoveList.data[cnt+1,index+4] != 0 or adeMoveList.data[cnt+1,index+5] != 0 or adeMoveList.data[cnt+1,index+6] != 0: #for 6ADEs
#                                             nextTimeStamp = True
                                    
#                             if nextTimeStamp == True:
#                                 TimeStamp += 1

#                                 if TimeStamp > 0:
#                                     #sysStateList = mbs.systemData.GetSystemState()
#                                     IDList = []
#                                     for ID in adeNodes[0].elements:
#                                         IDList += [ID]
#                                         ##+++++++++++++++++++++++++++++++++++++++
#                                         ## Disconnecting Connections that are no longer Needed
#                                         for k in range(len(adeNodes[TimeStamp-1].connectionList[ID])):
#                                             if adeNodes[TimeStamp-1].connectionList[ID][k] not in adeNodes[TimeStamp].connectionList[ID] and (adeNodes[TimeStamp-1].connectionList[ID][k][2] in IDList or adeNodes[TimeStamp-1].connectionList[ID][k][2] <= -1):
#                                                 DisconnectADE(mbs,ADE,adeNodes[TimeStamp-1].connectionList[ID][k])
#                                                 if config.simulation.debugConnectionInfo:
#                                                     logging.info("Disconnected" + str(adeNodes[TimeStamp-1].connectionList[ID][k]))
#                                         # # ##+++++++++++++++++++++++++++++++++++++++
#                                         # ## Connecting new Connections of ADEs
#                                         # for k in range(len(adeNodes[TimeStamp].connectionList[ID])):
#                                         #     if adeNodes[TimeStamp].connectionList[ID][k] not in adeNodes[TimeStamp-1].connectionList[ID] and (adeNodes[TimeStamp].connectionList[ID][k][2] in IDList or adeNodes[TimeStamp].connectionList[ID][k][2] <= -1):
#                                         #         ConnectADE(mbs,ADE,adeNodes[TimeStamp].connectionList[ID][k])
#                                         #         if config.simulation.debugConnectionInfo:
#                                         #             logging.info("Connected" + str(adeNodes[TimeStamp].connectionList[ID][k]))

#                                     mbs.AssembleLTGLists()
#                                     mbs.systemData.SetSystemState(sysStateList,configuration = exu.ConfigurationType.Initial)

#                         ##+++++++++++++++++++++++++++++++++++++++
#                         ## Simulation of Mesh                          
#                         mbs.SetPreStepUserFunction(PreStepUserFunction)
                        
#                         if computeDynamic:

#                             simulationSettings.solutionSettings.solutionInformation = "MoveList Line " +str(i+2)
                            
#                             if config.simulation.SolveDynamic:
#                                 simulationSettings.timeIntegration.numberOfSteps = int(config.simulation.nrSteps)
#                                 endTime=cf.CalcPause(adeMoveList,cnt+1,minPauseVal = 0.2)
#                                 config.simulation.endTime = endTime
#                                 simulationSettings.timeIntegration.endTime = endTime+config.simulation.idleTime
                                
#                                 # simulationSettings.timeIntegration.simulateInRealtime=True

#                                 exu.SolveDynamic(mbs, simulationSettings = simulationSettings,showHints=True)

#                                 # sysStateList = mbs.systemData.GetSystemState()
#                                 # mbs.systemData.SetSystemState(sysStateList,configuration = exu.ConfigurationType.Initial)
                                
#                                 # mbs.SetPreStepUserFunction(PreStepUserFunction)
                                
#                                 # simulationSettings.timeIntegration.endTime = 1
#                                 # exu.SolveDynamic(mbs, simulationSettings = simulationSettings,showHints=True)
                                
#                             else:
#                                 numberOfLoadSteps=10
#                                 simulationSettings.staticSolver.numberOfLoadSteps = numberOfLoadSteps
#                                 exu.SolveStatic(mbs, simulationSettings = simulationSettings,showHints=True)
                                
                                
#                             ## Sending new Sensor Values to Webserver
#                             SensorValuesTime[cnt+1][0]={}
#                             SensorValues[cnt+1] = {}
#                             for ID in adeIDs:
#                                 SensorValues[cnt+1][ID] = np.asmatrix([mbs.GetSensorValues(ADE[ID]['sensors'][0])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][1])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][2])[0:2]])
#                             client.Put('SensorValues',pickle.dumps(SensorValues))
    
#                             SensorValuesTime[cnt+1][0]=SensorValues[cnt+1]

                        
#                             print("stop flag=", mbs.GetRenderEngineStopFlag())
#                             if config.simulation.saveImages:
#                                 SC.RedrawAndSaveImage() 
#                             #mbs.WaitForUserToContinue() 
                        
#                         if not mbs.GetRenderEngineStopFlag():
#                             sysStateList = mbs.systemData.GetSystemState()
#                             mbs.systemData.SetSystemState(sysStateList,configuration = exu.ConfigurationType.Initial)

#                             ##+++++++++++++++++++++++++++++++++++++++
#                             ## Connecting new Connections of ADEs
#                             if nextTimeStamp:
#                                 nextTimeStamp = False
#                                 #sysStateList = mbs.systemData.GetSystemState()
#                                 IDList = []
#                                 for ID in adeNodes[0].elements:
#                                     IDList += [ID]
                                
#                                     ##+++++++++++++++++++++++++++++++++++++++
#                                     ## Connecting new Connections of ADEs
#                                     for k in range(len(adeNodes[TimeStamp].connectionList[ID])):
#                                         if adeNodes[TimeStamp].connectionList[ID][k] not in adeNodes[TimeStamp-1].connectionList[ID] and (adeNodes[TimeStamp].connectionList[ID][k][2] in IDList or adeNodes[TimeStamp].connectionList[ID][k][2] <= -1):
#                                             ConnectADE(mbs,ADE,adeNodes[TimeStamp].connectionList[ID][k])
#                                             if config.simulation.debugConnectionInfo:
#                                                 logging.info("Connected" + str(adeNodes[TimeStamp].connectionList[ID][k]))
#                                     mbs.AssembleLTGLists()
#                                     mbs.systemData.SetSystemState(sysStateList,configuration = exu.ConfigurationType.Initial)
#                         else:
#                             ##+++++++++++++++++++++++++++++++++++++++
#                             ## Resetting Simulation
#                             break
                        
                        
                
#                 ##+++++++++++++++++++++++++++++++++++++++
#                 ## Starting Simulation without Movelist
#                 else:
#                     if computeDynamic:

#                             simulationSettings.solutionSettings.solutionInformation = "Exudyn ADE Simulation"

#                             mbs.Assemble()
#                             mbs.SetPreStepUserFunction(PreStepUserFunction)
                            
#                             if displaySimulation:
#                                 exu.StartRenderer()
#                                 mbs.WaitForUserToContinue()

#                             if config.simulation.SolveDynamic:
#                                 simulationSettings.timeIntegration.numberOfSteps = config.simulation.nrSteps
#                                 endTime=cf.CalcPause(adeMoveList,cnt+1,minPauseVal = 0.2)
#                                 config.simulation.endTime = config.simulation.endTime
#                                 simulationSettings.timeIntegration.endTime = config.simulation.endTime
#                                 exu.SolveDynamic(mbs, simulationSettings = simulationSettings,showHints=True)
#                             else:
#                                 numberOfLoadSteps=10
#                                 simulationSettings.staticSolver.numberOfLoadSteps = numberOfLoadSteps
#                                 exu.SolveStatic(mbs, simulationSettings = simulationSettings,showHints=True)

#                             ##+++++++++++++++++++++++++++++++++++++++
#                             ## Sending new Sensor Values to Webserver
#                             SensorValues[1] = {}
#                             for ID in adeIDs:
#                                 SensorValues[1][ID] = np.asmatrix([mbs.GetSensorValues(ADE[ID]['sensors'][0])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][1])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][2])[0:2]])
#                             client.Put('SensorValues',pickle.dumps(SensorValues))
                            
                        
#                             print("stop flag=", mbs.GetRenderEngineStopFlag())
                    
                        
#                     if not mbs.GetRenderEngineStopFlag():
#                         sysStateList = mbs.systemData.GetSystemState()
#                         mbs.systemData.SetSystemState(sysStateList,configuration = exu.ConfigurationType.Initial)
#                     else: 
#                         break

#                 if displaySimulation and not mbs.GetRenderEngineStopFlag():  
#                     mbs.WaitForUserToContinue()      
#                     exu.StopRenderer() #safely close rendering window!
#                 try:
#                     if config.simulation.solutionViewer:
#                          sol = LoadSolutionFile(config.simulation.solutionViewerFile, safeMode=True)#, maxRows=100) 
#                          SolutionViewer(mbs,sol) #can also be entered in 
#                 except:
#                     logging.warning('Couldnt load Solution File')


#                 ##+++++++++++++++++++++++++++++++++++++++
#                 ## Saving Sensor Data to File
#                 if config.simulation.saveSensorValues:
#                      pickle.dump(  SensorValues,  open(config.simulation.sensorValuesFile,  'wb')  )
#                      pickle.dump(  SensorValuesTime,  open(config.simulation.sensorValuesTimeFile,  'wb')  )

#                 ##+++++++++++++++++++++++++++++++++++++++
#                 ## Resetting Simulation
#                 mbs.Reset()
#                 startSimulation = False
#                 client.Put('StartSimulation',json.dumps(startSimulation))
                
        
#         except Exception as ExeptionMessage: 
#             logging.warning('Error ocured during Simulation')
#             logging.warning(ExeptionMessage)
#             startSimulation = False
#             #client.Put('StartSimulation',json.dumps(False))
#             mbs.Reset()
#         time.sleep(config.simulation.looptime-time.time()%config.simulation.looptime)
