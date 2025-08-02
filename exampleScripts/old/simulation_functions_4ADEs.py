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

if __name__ == "__main__":
    if os.getcwd().split('\\')[-1].__contains__('exampleScripts'):
        os.chdir('..')
    if sys.path[0].split('\\')[-1].__contains__('exampleScripts'):
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
    #print("called")
    #print("mbs=", mbs2)q
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



##+++++++++++++++++++++++++++++++++++++++
## Initialization of Module
config.init()
logging.info('Starte Simulations Module')

# client = wf.WebServerClient()
SC = exu.SystemContainer()
mbs = SC.AddSystem()

computeDynamic=True


##simulation options
displaySimulation=True

##++++++++++++++++++++++++++++++
##ANIMATIONS
##make images for animations (requires FFMPEG):
my_dir = os.path.abspath('./functions')
# os.system('start /MIN cmd /C  python "'+os.path.join(my_dir, 'webserver_functions.py')+'"')    
# os.system('start /MIN cmd /C python "'+os.path.join(my_dir, 'measurement_functions.py')+'"')


#%% define system
"""
Simulation of four ADEs 
      5-----6
      / \   /
    /   \ /
    3-----4
  / \   /
  /   \ /
1-----2

"""
adeNodes = df.ElementDefinition()
adeNodes.AddElementNodes(ID= [4,1], Nodes = [[3,1,2],[3,2,4]])
adeNodes.AddElementNodes(ID= 4, Nodes = [3,1,2])
adeNodes.AddElementNodes(ID= 1, Nodes = [3,2,4])
adeNodes.AddElementNodes(ID= 2, Nodes = [3,4,5])
adeNodes.AddElementNodes(ID= 3, Nodes = [6,5,4])

adeMoveList = df.adeMoveList('data\moveList\Fahrt014ADE.txt')





timeTotal=0
numberOfSteps=0

startSimulation = False
while not startSimulation:
    try:
        ##+++++++++++++++++++++++++++++++++++++++
        ## Checking if to end the Programm
        ##+++++++++++++++++++++++++++++++++++++++
        ## Initialization of Variables
        mbs.variables['springDamperDriveObjects']=[] #all drives that need to be changed over time
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
        #     adeMoveList = df.adeMoveList()
        #     adeMoveList.FromJson(client.Get('MoveList'))


        ##+++++++++++++++++++++++++++++++++++++++
        ## Starting preparations for Simulation
        config.simulation.useRevoluteJoint2DTower=True #setFalse Springs ASME2020CI, True = revolute joints
        #Options for six-bar linkages, if all False RevoluteJoint2D are used for the six-bar linkages
        config.simulation.simplifiedLinksIdealRevoluteJoint2D = False #simplified model using only revoluteJoints at edges
        config.simulation.simplifiedLinksIdealCartesianSpringDamper = False
        
        config.simulation.useCompliantJoints=False          #False for ideal
        config.simulation.numberOfElementsBeam=8
        config.simulation.numberOfElements=16
    
        config.simulation.useCrossSpringPivot = False #use CrossSpringPivot
    
        config.simulation.cartesianSpringDamperActive = False #setTrue ASME2020CI
        config.simulation.connectorRigidBodySpringDamper = False 
        
        config.simulation.rigidBodySpringDamperNonlinearStiffness = True #use nonlinear compliance matrix then choose circularHinge and crossSpringPivots
        config.simulation.rigidBodySpringDamperNonlinearStiffnessCircularHinge=True
        config.simulation.rigidBodySpringDamperNonlinearStiffnessCrossSpringPivot=False
        config.simulation.rigidBodySpringDamperNeuralNetworkCircularHinge=False
        config.simulation.rigidBodySpringDamperNeuralNetworkCrossSprings=False


        if config.simulation.useCompliantJoints or config.simulation.useCrossSpringPivot:
            config.simulation.graphics = 'graphicsSTL'
        else:
            config.simulation.graphics = 'graphicsStandard'

        if startSimulation and (type(adeNodes) == df.ElementDefinition or type(adeNodes) == dict):
            if config.simulation.useMoveList: 
                ADE, mbs = GenerateMesh(adeNodes,mbs,adeMoveList = adeMoveList)         #Generating Mesh with Movelist
            else:
                ADE, mbs = GenerateMesh(adeNodes,mbs)                                   #Generatin Mesh without Movelist


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
            
            # simulationSettings.displayComputationTime= True
            ##++++++++++++++++++++++++++++++
            ##ANIMATIONS
            ##make images for animations (requires FFMPEG):
            ##requires a subfolder 'images'
            if config.simulation.animation:
                simulationSettings.solutionSettings.recordImagesInterval=0.04 #simulation.endTime/200
            
            

            ##+++++++++++++++++++++++++++++++++++++++
            ## Assemble Mesh 
            mbs.Assemble()
            sysStateList = mbs.systemData.GetSystemState()

            
            if displaySimulation:
                exu.StartRenderer()
                mbs.WaitForUserToContinue()

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
            for ID in adeIDs:
                SensorValues[0][ID] = np.asmatrix([mbs.GetSensorValues(ADE[ID]['sensors'][0])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][1])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][2])[0:2]])
            # client.Put('SensorValues',pickle.dumps(SensorValues))

            SensorValuesTime[0]=SensorValues.copy()
            
            ##############################################################
            # calculate in the first step the static solution
            # simulationSettings.solutionSettings.sensorsAppendToFile =False
            # simulationSettings.solutionSettings.sensorsWriteFileFooter=False
            # simulationSettings.solutionSettings.sensorsWriteFileHeader = False
            
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
            mbs.systemData.SetODE2Coordinates(solODE2initial, configuration=exu.ConfigurationType.Initial)  #set old solution as initial value for new solution
             
            mbs.WaitForUserToContinue()       
            ##############################################################          
            
            # mbs.SetPreStepUserFunction(PreStepUserFunction)

            ##############################################################
            simulationSettings.timeIntegration.numberOfSteps = int(config.simulation.nrSteps)
            simulationSettings.timeIntegration.endTime = config.simulation.endTime

            # simulationSettings.timeIntegration.newton.absoluteTolerance = 1e3
            # simulationSettings.timeIntegration.newton.numericalDifferentiation.minimumCoordinateSize = 1
        
            # simulationSettings.timeIntegration.verboseMode = 0
            # #simulationSettings.timeIntegration.newton.useNumericalDifferentiation = True
            # #simulationSettings.timeIntegration.newton.numericalDifferentiation.doSystemWideDifferentiation = True
            
            simulationSettings.timeIntegration.generalizedAlpha.computeInitialAccelerations = False
            
            # simulationSettings.timeIntegration.generalizedAlpha.spectralRadius = 0.4
            # simulationSettings.timeIntegration.adaptiveStep = True #disable adaptive step reduction
            ##############################################################

            
            timeTotal=timeTotal+mbs.sys['staticSolver'].timer.total
            numberOfSteps=numberOfSteps+1


            ##############################################################
            # IMPORTANT!!!!!!!!!
            simulationSettings.timeIntegration.newton.useModifiedNewton = True #JG
            # simulationSettings.linearSolverType = exu.LinearSolverType.EigenSparse #sparse solver !!!!!!!!!!!!!!!
            ##############################################################
            simulationSettings.displayStatistics = False     
            simulationSettings.displayComputationTime = False  
            # simulationSettings.timeIntegration.preStepPyExecute = "UserFunction()\n" #NEU?
            # if True: #JG
            #     simulationSettings.timeIntegration.generalizedAlpha.useIndex2Constraints = False
            #     simulationSettings.timeIntegration.generalizedAlpha.useNewmark= False


            if config.simulation.solutionViewer: #use this to reload the solution and use SolutionViewer
                SC.visualizationSettings.general.autoFitScene=False #if reloaded view settings
                simulationSettings.solutionSettings.solutionWritePeriod = config.simulation.solutionWritePeriod
                simulationSettings.solutionSettings.writeSolutionToFile = True
                simulationSettings.solutionSettings.coordinatesSolutionFileName = config.simulation.solutionViewerFile
                simulationSettings.solutionSettings.writeFileFooter = False
                simulationSettings.solutionSettings.writeFileHeader = True
                
                
                # simulationSettings.solutionSettings.sensorsWriteFileFooter=False
                # simulationSettings.solutionSettings.sensorsWriteFileHeader = False
                
                if os.path.isfile(solutionFilePath):
                    os.remove(solutionFilePath)
                simulationSettings.solutionSettings.appendToFile = True
                simulationSettings.solutionSettings.sensorsAppendToFile =True
                

            ##+++++++++++++++++++++++++++++++++++++++
            ## Starting Simulation using Movelist
            if config.simulation.useMoveList:
                ##+++++++++++++++++++++++++++++++++++++++
                ## Checking and Setting Connections of ADEs
                TimeStamp = 0
                

                for i in range(len(adeMoveList.data)-1):
                    mbs.variables['line']=i+1
                    SensorValuesTime[i+1]={}
                    # if i>0:
                    #     SensorValuesTime[i]={}
                    
                    # if i == len(adeMoveList.data)-2 and config.simulation.solutionViewer:
                    #     simulationSettings.solutionSettings.sensorsWriteFileFooter = False
                    #     simulationSettings.solutionSettings.sensorsWriteFileHeader = False
                    #     simulationSettings.solutionSettings.writeFileFooter = True
                    #     simulationSettings.solutionSettings.writeFileHeader = False
                    #     if i == 0:
                    #         simulationSettings.solutionSettings.writeFileHeader = False
                            # simulationSettings.solutionSettings.sensorsWriteFileHeader = False

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
                                #sysStateList = mbs.systemData.GetSystemState()
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
                    mbs.SetPreStepUserFunction(PreStepUserFunction)
                    
                    if computeDynamic:
                        simulationSettings.solutionSettings.solutionInformation = "MoveList Line " +str(i+2)
                        
                        if not config.simulation.SolveDynamic:
                            numberOfLoadSteps=10
                            simulationSettings.staticSolver.numberOfLoadSteps = numberOfLoadSteps
                            exu.SolveStatic(mbs, simulationSettings = simulationSettings,showHints=True)
                            
                            timeTotal=timeTotal+mbs.sys['staticSolver'].timer.total
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
                                        
                    
                    if not mbs.GetRenderEngineStopFlag():
                        sysStateList = mbs.systemData.GetSystemState()
                        mbs.systemData.SetSystemState(sysStateList,configuration = exu.ConfigurationType.Initial)

                    else:
                        ##+++++++++++++++++++++++++++++++++++++++
                        ## Resetting Simulation
                        break
            
            ##+++++++++++++++++++++++++++++++++++++++
            ## Starting Simulation without Movelist
            else:
                if computeDynamic:

                        simulationSettings.solutionSettings.solutionInformation = "Exudyn ADE Simulation"

                        mbs.Assemble()
                        
                        if displaySimulation:
                            exu.StartRenderer()
                            mbs.WaitForUserToContinue()

                        exu.SolveDynamic(mbs, simulationSettings = simulationSettings)

                        ##+++++++++++++++++++++++++++++++++++++++
                        ## Sending new Sensor Values to Webserver
                        SensorValues[1] = {}
                        for ID in adeIDs:
                            SensorValues[1][ID] = np.asmatrix([mbs.GetSensorValues(ADE[ID]['sensors'][0])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][1])[0:2],mbs.GetSensorValues(ADE[ID]['sensors'][2])[0:2]])
                        client.Put('SensorValues',pickle.dumps(SensorValues))
                        
                    
                        print("stop flag=", mbs.GetRenderEngineStopFlag())
                
                    
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
        
 
