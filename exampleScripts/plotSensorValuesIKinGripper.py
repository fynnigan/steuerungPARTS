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


# solutionFilePath="./solution/coordinatesSolution.txt"
# if os.path.isfile(solutionFilePath):
#     os.remove(solutionFilePath)

from functions import data_functions as df
from functions import com_functions as cf
from functions import marker_functions as mf
from functions import measurement_functions as mef
from functions import webserver_functions as wf
import matplotlib.ticker as ticker

import config

import logging
import matplotlib.pyplot as plt
import config
import numpy as np
from flask import jsonify
import json
import time
import cv2
import pickle

#%% Ploten von zuvor gespeicherten Sensorwerten:
plt.close('all')
fontSize=14
plt.rc('font', size=fontSize) 
plt.rc('axes', titlesize=fontSize)     # fontsize of the axes title
plt.rc('axes', labelsize=fontSize)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=fontSize)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fontSize)    # fontsize of the tick labels
plt.rc('legend', fontsize=fontSize)    # legend fontsize



from functions import visualization_functions as vf
from functions.control_functions import *

SensorValuesIdealCircle = pickle.load( open('output/iKinGripper/SensorValues.pickle',  'rb') ) # loading sensor values
SensorValuesIdealImportCircle = pickle.load( open('output/iKinGripper/SensorValuesTime.pickle',  'rb') ) # loading sensor values
#%% load sensor data
# SensorValuesIdeal = pickle.load( open('output/nonlinearComplianceGripper/SensorValuesIdeal.pickle',  'rb') ) # loading sensor values
# SensorValuesIdealImport = pickle.load( open('output/nonlinearComplianceGripper/SensorValuesTimeIdeal.pickle',  'rb') ) # loading sensor values
SensorValuesIdeal = pickle.load( open('output/iKinGripper/SensorValuesCrossSpringsCorr.pickle',  'rb') ) # loading sensor values
SensorValuesIdealImport = pickle.load( open('output/iKinGripper/SensorValuesTimeCrossSpringsCorr.pickle',  'rb') ) # loading sensor values

# SensorValuesTime[line][time][ID][Point]
# line: line from movelist e.g. 0-28
# time: time from simulation
# ID: ID of ADE
# Point: Point of ADE [0,1,2]

# SensorValues4ATCRBSD = pickle.load( open('output/example4ATCs/SensorValuesRBSD.pickle',  'rb') ) # loading sensor values
# SensorValues4ATCRBSDImport = pickle.load( open('output/example4ATCs/SensorValuesTimeRBSD.pickle',  'rb') ) # loading sensor values

# SensorValues4ATCRBSD = pickle.load( open('output/example4ATCs/SensorValuesCS.pickle',  'rb') ) # loading sensor values
# SensorValues4ATCRBSDImport = pickle.load( open('output/example4ATCs/SensorValuesTimeCS.pickle',  'rb') ) # loading sensor values


# SensorValues4ATCRBSD = pickle.load( open('output/example4ATCs/SensorValuesTest.pickle',  'rb') ) # loading sensor values
# SensorValues4ATCRBSDImport = pickle.load( open('output/example4ATCs/SensorValuesTimeTest.pickle',  'rb') ) # loading sensor values

SensorValues4ATCRBSD = pickle.load( open('output/iKinGripper/SensorValuesCrossSprings.pickle',  'rb') ) # loading sensor values
SensorValues4ATCRBSDImport = pickle.load( open('output/iKinGripper/SensorValuesTimeCrossSprings.pickle',  'rb') ) # loading sensor values


import matplotlib.ticker as ticker
# 'data//experiments//supportStructure//Image//7//MeshPoints.txt'
# 'data//experiments//4ADEs#//uncorrected//MeshPoints.txt'
# SensorValues4ATCMeasurement = {}
# with open('data//experiments//4ADEs#//uncorrected//MeshPointsVicon.txt') as f:
#     lines = f.readlines()
#     for i,line in enumerate(lines):
#         SensorValues4ATCMeasurement[i] = {}
#         line = line.split()
#         for j in range(0,len(line),7):
#             SensorValues4ATCMeasurement[i][int(line[j])] = {}
#             for k in range(3):
#                 SensorValues4ATCMeasurement[i][int(line[j])][k] = np.matrix(line[j+k*2+1:j+k*2+3],dtype=float)

SensorValues4ATCMeasurement = pickle.load( open('output/iKinGripper/SensorValuesIdeal.pickle',  'rb') ) # loading sensor values
SensorValues4ATCMeasurementTime = pickle.load( open('output/iKinGripper/SensorValuesTimeIdeal.pickle',  'rb') ) # loading sensor values
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
        




# ID=[6]
ID=[6,9]
Point=[2]

# #%% load moveList
# # adeMoveList = df.adeMoveList('data\moveList\Fahrt014ADE.txt')
# adeMoveList = df.adeMoveList('data\moveList\Fahrt014ADEDemo.txt')
# #%% define system
# """
# Simulation of four ADEs 
#       5-----6
#      / \   /
#     /   \ /
#    3-----4
#   / \   /
#  /   \ /
# 1-----2

# """

# adeNodes = df.ElementDefinition()
# adeNodes.AddElementNodes(ID= [4,1], Nodes = [[3,1,2],[3,2,4]])
# adeNodes.AddElementNodes(ID= 4, Nodes = [3,1,2])
# adeNodes.AddElementNodes(ID= 1, Nodes = [3,2,4])
# adeNodes.AddElementNodes(ID= 2, Nodes = [3,4,5])
# adeNodes.AddElementNodes(ID= 3, Nodes = [6,5,4])

# adeNodes = df.ElementDefinition()
# adeNodes.AddElementNodes(ID= 5, Nodes = [3,1,2])
# adeNodes.AddElementNodes(ID= 6, Nodes = [3,2,4])
# adeNodes.AddElementNodes(ID= 2, Nodes = [3,4,5])
# adeNodes.AddElementNodes(ID= 3, Nodes = [6,5,4])
# TimeStamp = 0
# adeNodes = {}



# adeNodes = df.ElementDefinition()
# adeNodes.ReadMeshFile('data//experiments//gripper//initial9_gripper')
# adeMoveList = df.adeMoveList('data//experiments//gripper//configuration_gripper.txt')



TimeStamp=0
adeNodes = {}
adeNodes = df.ElementDefinition()
adeNodes.ReadMeshFile('data//experiments//gripper//initial9_gripper')
adeMoveList = df.adeMoveList('data//experiments//gripper//configuration_gripper.txt')
# adeNodes.SetNodeNumbersToAde()

adeNodes.connectionList={}
adeNodes.SetListOfNodes()           
adeNodes.SetConnectionList()
adeNodes.SetNodeNumbersToAde()
adeNodes.SetADEPointsOfNode() 
# currentLine = df.adeMoveList()
# adeIDs = []
# nextTimeStamp = False
# for ID in adeNodes.elements:
#     adeIDs.append(ID)

# currentLine.data = adeMoveList.data[0,:]
# adeNodes.ConnectionListFromMoveListLine(adeIDs,currentLine)


# for i in range(1,adeMoveList.data.shape[0]):
#     for j in range(len(adeIDs)):
#         index = adeMoveList.GetIndexOfID(adeIDs[j])
#         if index != None:
#             if adeMoveList.data[i,index+4] != adeMoveList.data[i-1,index+4] or adeMoveList.data[i,index+5] != adeMoveList.data[i-1,index+5] or adeMoveList.data[i,index+6] != adeMoveList.data[i-1,index+6]:
#                 nextTimeStamp = True
#     if nextTimeStamp == True:
#         nextTimeStamp = False
#         TimeStamp += 1
#         currentLine.data = adeMoveList.data[i,:]
#         adeNodes[TimeStamp] = df.ElementDefinition()
#         adeNodes[TimeStamp].ConnectionListFromMoveListLine(adeIDs,currentLine)

# adeMoveList.Split(maxStepsize=200, keepConnectionInfo=True)    

# client.Put('MoveList',adeMoveList.ToJson())
# client.Put('adeNodes',pickle.dumps(adeNodes))
# client.Put('StartSimulation',json.dumps(True))





# #%% renewTime from simulationTime to continuous time
SensorValues4ATCRBSDTime=vf.fromSimulationTimeToContinuousTime(SensorValues4ATCRBSDImport)

# #%% AverageMeshPointsOverElementDefinitionTime
# SensorValues4ATCRBSDTime = AverageMeshPointsOverElementDefinitionTime(SensorValues4ATCRBSDTime,adeNodes,adeMoveList,elementDefinition=False)     
# SensorValues4ATCRBSD = AverageMeshPointsOverElementDefinition(SensorValues4ATCRBSD,adeNodes,adeMoveList,elementDefinition=False)
# SensorValues4ATCMeasurement = AverageMeshPointsOverElementDefinition(SensorValues4ATCMeasurement,adeNodes,adeMoveList,elementDefinition=False)


# #%% renewTime from simulationTime to continuous time
SensorValues4ATCIdealTimeRenewed=vf.fromSimulationTimeToContinuousTime(SensorValuesIdealImport)
# SensorValues4ATCRBSDTimeRenewed=vf.fromSimulationTimeToContinuousTime(SensorValues4ATCRBSDImport)


# #%% AverageMeshPointsOverElementDefinitionTime
SensorValues4ATCIdealTime = AverageMeshPointsOverElementDefinitionTime(deepcopy(SensorValues4ATCIdealTimeRenewed),adeNodes,adeMoveList,elementDefinition=False)
# SensorValues4ATCRBSDTime = AverageMeshPointsOverElementDefinitionTime(deepcopy(SensorValues4ATCRBSDTimeRenewed),adeNodes,adeMoveList,elementDefinition=False)     



#moveMeshPoints to origin
# vf.MoveToOrigin(SensorValues4ATCMeasurement,adeNodes,5,3,additionalRotation=1.5*np.pi/180*2)
# MoveToOrigin(meshPointsVicon,adeNodes,5,3)


#movePoints to first idealPoint ???
# lineToStart=4
# SensorValues4ATCMeasurement=vf.substractOffsetSensorValues(SensorValues4ATCMeasurement,SensorValuesIdeal,ID,Point,lineToStart=lineToStart)
# SensorValues4ATCRBSD=vf.substractOffsetSensorValues(SensorValues4ATCRBSD,SensorValuesIdeal,ID,Point,lineToStart=lineToStart)




# red - '^' ideal, deprecated
# orange - 'x' simulation, deprecated
# blue - 'o' measurement, deprecated
markerstyleListSelected=['^','x','o']
colorlineListSelected=['black','blue','orange']
lineStyleSelected=['-','--','-.']



#%% figure1 plot Trajectories
figName='Trajectories'
Fig3=plt.figure(figName)
ax=Fig3.gca() # get current axes

# vf.plotTrajectoriesTime(ID,Point,SensorValues4ATCIdealTime,adeMoveList, colorlineListSelected[0], lineStyleSelected[0], markerstyleListSelected[0],figName,label='ideal',printMarkerLabel=False,printLegend=False)
# vf.plotTrajectoriesTime(ID,Point,SensorValues4ATCRBSDTime,adeMoveList,  colorlineListSelected[1], lineStyleSelected[1], markerstyleListSelected[1],figName,label='model',printMarkerLabel=False,printLegend=False)

for i in ID:
    for j in Point:
        vf.PlotSensorValuePoint(SensorValuesIdealCircle,i,j,linecolor = 'black',lineStyle='--',markercolor = 'black',marker='', figureName = figName,label='',combinePlots=True,plotLine=True,plotMarker=False,time=False,markerLabel='',printMarkerLabel=False,everyTwoLabels=False,markerSize=100,linewidths=2)
        vf.PlotSensorValuePoint(SensorValues4ATCMeasurement,i,j,linecolor = 'black',lineStyle='',markercolor = 'black',marker='', figureName = figName,label='',combinePlots=True,plotLine=True,plotMarker=True,time=False,markerLabel='',printMarkerLabel=False,everyTwoLabels=False,markerSize=100,linewidths=2)
        vf.PlotSensorValuePoint(SensorValues4ATCRBSD,i,j,linecolor = colorlineListSelected[1],lineStyle='',markercolor = 'red' ,marker='x', figureName = figName,label='',combinePlots=True,plotLine=True,plotMarker=True,time=False,markerLabel='',printMarkerLabel=False,everyTwoLabels=False,markerSize=100,linewidths=2)
        vf.PlotSensorValuePoint(SensorValuesIdeal,i,j,linecolor = 'red',lineStyle='',markercolor = 'blue',marker='x', figureName = figName,label='',combinePlots=True,plotLine=True,plotMarker=True,time=False,markerLabel='',printMarkerLabel=True,everyTwoLabels=False,markerSize=100,linewidths=1)
 
        # offsetOfLine=vf.calculateOffset(SensorValuesIdeal,SensorValues4ATCMeasurement,0,i,j)
        # vf.substractOffset(SensorValues4ATCMeasurement,i,j,offsetOfLine=offsetOfLine,linecolor = 'orange',markercolor = 'orange', figureName = figName,label='',marker='x',combinePlots=True)



#%% figure1 mesh
lineStep=0 #first value of line 0
vf.plotMesh(SensorValues4ATCIdealTime,lineStep,plotFirstIndex=True,figureName=figName,labelATC=True,linestyle='-')

lineStep=0 #first value of line 0
vf.plotMesh(SensorValues4ATCIdealTime,lineStep,plotFirstIndex=False,figureName=figName,labelATC=False,linestyle='-')

lineStep=5 #first value of line 1
vf.plotMesh(SensorValues4ATCIdealTime,lineStep,plotFirstIndex=False,figureName=figName,labelATC=False,linestyle='-.')


# plt.legend()
plt.grid()
plt.axis('equal')
ax.grid(True, 'major', 'both')
ax.xaxis.set_major_locator(ticker.MaxNLocator(10)) 
ax.yaxis.set_major_locator(ticker.MaxNLocator(10)) 
plt.xlim(-0.35, 0.6)
ax.set_ylabel(r'nodes y in m')
ax.set_xlabel(r'nodes x in m')
plt.tight_layout()
plt.show()


saveFigure=True
if saveFigure:
    plt.savefig('output/iKinPlots/TrajectoriesGripper.pdf',format='pdf')
    plt.savefig('output/iKinPlots/TrajectoriesGripper.svg', format='svg')






figName='enlargedRepTraj'
Fig3=plt.figure(figName)
ax=Fig3.gca() # get current axes
#enlarged representation
# vf.plotTrajectoriesTime(ID,Point,SensorValues4ATCIdealTime,adeMoveList, colorlineListSelected[0], lineStyleSelected[0], markerstyleListSelected[0],figName,label='ideal',printMarkerLabel=False,printLegend=False)
# vf.plotTrajectoriesTime(ID,Point,SensorValues4ATCRBSDTime,adeMoveList,  colorlineListSelected[1], lineStyleSelected[1], markerstyleListSelected[1],figName,label='model',printMarkerLabel=False,printLegend=False)

for i in ID:
    for j in Point:
        vf.PlotSensorValuePoint(SensorValuesIdealCircle,i,j,linecolor = 'black',lineStyle='--',markercolor = 'black',marker='', figureName = figName,label='',combinePlots=True,plotLine=True,plotMarker=False,time=False,markerLabel='',printMarkerLabel=False)
        vf.PlotSensorValuePoint(SensorValues4ATCMeasurement,i,j,linecolor = 'black',lineStyle='',markercolor = 'black',marker='', figureName = figName,label='',combinePlots=True,plotLine=True,plotMarker=True,time=False,markerLabel='',printMarkerLabel=False)
        vf.PlotSensorValuePoint(SensorValues4ATCRBSD,i,j,linecolor = colorlineListSelected[1],lineStyle='',markercolor = 'red' ,marker='o', figureName = figName,label='',combinePlots=True,plotLine=True,plotMarker=True,time=False,markerLabel='',printMarkerLabel=False)
        vf.PlotSensorValuePoint(SensorValuesIdeal,i,j,linecolor = 'red',lineStyle='',markercolor = 'blue',marker='x', figureName = figName,label='',combinePlots=True,plotLine=True,plotMarker=True,time=False,markerLabel='',printMarkerLabel=True)
        
        # offsetOfLine=vf.calculateOffset(SensorValuesIdeal,SensorValues4ATCMeasurement,0,i,j)
        # vf.substractOffset(SensorValues4ATCMeasurement,i,j,offsetOfLine=offsetOfLine,linecolor = 'orange',markercolor = 'orange', figureName = figName,label='',marker='x',combinePlots=True)



#%% figure1 mesh
lineStep=0 #first value of line 0
vf.plotMesh(SensorValues4ATCIdealTime,lineStep,plotFirstIndex=True,figureName=figName,labelATC=True,linestyle='-')

lineStep=0 #first value of line 0
vf.plotMesh(SensorValues4ATCIdealTime,lineStep,plotFirstIndex=False,figureName=figName,labelATC=False,linestyle='-')

lineStep=5 #first value of line 1
vf.plotMesh(SensorValues4ATCIdealTime,lineStep,plotFirstIndex=False,figureName=figName,labelATC=False,linestyle='-.')


# plt.legend()
plt.grid()
plt.axis('equal')
ax.grid(True, 'major', 'both')
plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.12)
ax.xaxis.set_major_locator(ticker.MaxNLocator(5)) 
ax.yaxis.set_major_locator(ticker.MaxNLocator(5)) 
plt.xlim(-0.28, 0.)
plt.ylim(0.55, 0.78)
ax.set_ylabel(r'nodes y in m')
ax.set_xlabel(r'nodes x in m')
plt.tight_layout()
plt.show()


saveFigure=True
if saveFigure:
    plt.savefig('output/iKinPlots/enlargedRepTraj.pdf',format='pdf')
    plt.savefig('output/iKinPlots/enlargedRepTraj.svg', format='svg')
# #%% figure2 movement over time
# figName='movementOverTime'
# Fig4=plt.figure(figName)
# ax=Fig4.gca() # get current axes


# for i in ID:
#     for j in Point:
#         print('ade'+str(i)+'point'+str(j))
#         vf.plotComponentsOverTime(i,j,SensorValues4ATCIdealTime,colorLine='black',lineStyle='-',markerStyle='^',label='ideal',printLegend=True,printMarkerLabel=True)
#         vf.plotComponentsOverTime(i,j,SensorValues4ATCRBSDTime,colorLine='blue',lineStyle='--',markerStyle='x',label='model',printLegend=True,printMarkerLabel=False)
# # vf.plotComponentsOverTime(3,0,SensorValues4ATCRBSDTime,colorLine='black',lineStyle='-',markerStyle='^',label='ideal2',printLegend=True)
# # vf.plotComponentsOverTime(6,0,SensorValues4ATCRBSDTime,colorLine='black',lineStyle='-',markerStyle='^',label='ideal',printLegend=True)


# plt.legend()
# plt.grid()
# ax.grid(True, 'major', 'both')
# ax.xaxis.set_major_locator(ticker.MaxNLocator(10)) 
# ax.yaxis.set_major_locator(ticker.MaxNLocator(10)) 
# plt.xlim(-0.5, 20)
# ax.set_ylabel(r'x component of ADE '+str(ID)+' Point '+str(Point)+' in m')
# ax.set_xlabel(r'time t in s')
# plt.show()

# del SensorValuesIdeal[0][1]


#%% figure1 plot Trajectories
figName='error'
Fig3=plt.figure(figName)
ax=Fig3.gca() # get current axes

#%% figure3 plot and calculate offset/error
[errorModelIdealList,maxError,maxErrorMean]=vf.plotErrorOffset(ID,Point,SensorValues4ATCRBSD,SensorValues4ATCMeasurement,figureName='error',combinePlots=True,label='without error correction',linecolor='red',marker='o')
print('maxError: ',maxError,'maxErrorMean: ',maxErrorMean)

[errorModelIdealList,maxError,maxErrorMean]=vf.plotErrorOffset(ID,Point,SensorValuesIdeal,SensorValues4ATCMeasurement,figureName='error',combinePlots=True,label='with error correction',linecolor='blue',marker='x')
print('maxError: ',maxError,'maxErrorMean: ',maxErrorMean)

# plt.grid()
# plt.axis('equal')
ax.grid(True, 'major', 'both')
ax.xaxis.set_major_locator(ticker.MaxNLocator(10)) 
ax.set_ylabel(r'point with max error in mm')
# ax.yaxis.set_major_locator(ticker.MaxNLocator(10)) 
plt.xlim(-0.5, 9.5)
# plt.ylim(-0.001, 0.003)
plt.tight_layout()

if saveFigure:
    plt.savefig('output/iKinPlots/errorGripper.pdf',format='pdf')
    plt.savefig('output/iKinPlots/errorGripper.svg', format='svg')

# #%% figure4 error boxPlot
# for numberATCs in [6,8]:
#     if numberATCs==6:
#         ID2=[5,6,2,3]
#         Point2=[0,1,2]
#         offset=-0.2*0
#         facecolor='C0'
#         SensorValues=SensorValuesIdeal
#     elif numberATCs==8:
#         ID2=[5,6,2,3]
#         Point2=[0,1,2]
#         offset=0
#         facecolor='C2'
#         SensorValues=SensorValues4ATCRBSD

#     figName='boxPlot'
#     Fig5=plt.figure(figName)
#     ax=Fig5.gca() # get current axes
#     # fig, ax = plt.subplots()
    
#     [errorModelIdealList,maxError,maxErrorMean,errorModelIdealListValues]=vf.plotErrorOffsetBoxPlot(ID2,Point2,SensorValues,SensorValues4ATCMeasurement,figureName='boxPlot',label='4ATC',combinePlots=True,plotOverSteps=False,setOffset=offset,facecolor=facecolor)
#     import matplotlib.patches as mpatches
#     patch1 = mpatches.Patch(color='C0', label='IdealMeasurement')
#     patch2 = mpatches.Patch(color='C2', label='ModelMeasurement')
#     ax.legend(handles=[patch1,patch2])
    
#     plt.grid()
#     ax.grid(True, 'major', 'both')
#     ax.set_ylabel(r'error all steps in m')
#     ax.set_xlabel(r'points of ADEs')
#     plt.show()
    
    
    
#     #%% figure2 movement over time    
#     figName='boxPlot2'
#     Fig6=plt.figure(figName)
#     ax=Fig6.gca() # get current axes
    
#     [errorModelIdealList,maxError,maxErrorMean,errorModelIdealListValues]=vf.plotErrorOffsetBoxPlot(ID2,Point2,SensorValues,SensorValues4ATCMeasurement,figureName='boxPlot2',label='4ATC',combinePlots=True,plotOverSteps=True,setOffset=offset,facecolor=facecolor)
#     import matplotlib.patches as mpatches
#     patch1 = mpatches.Patch(color='C0', label='IdealMeasurement')
#     patch2 = mpatches.Patch(color='C2', label='ModelMeasurement')
#     ax.legend(handles=[patch1,patch2])
    
#     plt.grid()
#     ax.grid(True, 'major', 'both')
#     ax.set_ylabel(r'error of all points')
#     ax.set_xlabel(r'steps')
#     plt.show()    

















# #%% define some styles
# linestyleList=['-','--','-.',':','.',',','o','v','^','<','>','1','2','3','4','s','p','*','h','H','+','x','D','d','_']
# markerstyleList=['.',',','o','v','^','<','>','1','2','3','4','8','s','p','P','*','h','H','+','x','X','d','D']
# colorlineList=['b','g','r','c','m','y','k','b','g','r','c','m','y','k']

# linestyleListSelected=['x','o','^']



# ID=4
# Point=0
# #%% figure1 plot Trajectories
# figName='Trajectory'+'ID'+str(ID)+'Point'+str(Point)
# Fig3=plt.figure(figName)
# ax=Fig3.gca() # get current axes

# TimeStamp=0
# nextTimeStamp=False

# lineList=[0]

# SensorValues4ATCRBSDLine2=SensorValues4ATCRBSDTime[0].copy()
# SensorValues4ATCRBSDLine3=SensorValues4ATCRBSDTime[0].copy()

# # for ID in adeNodes[0].elements:
#     # for Point in range(3):
# for ID in [ID]:
#     for Point in [Point]:        
#         for line in SensorValues4ATCRBSDTime:
#             SensorValues4ATCRBSDLine2=SensorValues4ATCRBSDTime[line]
#             vf.PlotSensorValuePoint(SensorValues4ATCRBSDLine2,ID,Point,linecolor = colorlineList[ID] ,markercolor = colorlineList[ID],marker=linestyleListSelected[Point], figureName = figName,label='',combinePlots=True,plotMarker=False)

#             firstIndex=next(iter(SensorValues4ATCRBSDTime[line]))              
#             SensorValues4ATCRBSDLine3[0]=SensorValues4ATCRBSDTime[line][firstIndex]
#             vf.PlotSensorValuePoint(SensorValues4ATCRBSDLine3,ID,Point,linecolor = colorlineList[ID] ,markercolor = colorlineList[ID],marker=linestyleListSelected[Point], figureName = figName,label='',combinePlots=True,plotLine=False,plotMarker=True,time=True,markerLabel=str(line))
                   
#             if line == len(adeMoveList.data)-1: #last step
#                 lastIndex=list(SensorValues4ATCRBSDTime[len(adeMoveList.data)-1]).pop()       
#                 SensorValues4ATCRBSDLine3[0]=SensorValues4ATCRBSDTime[line][lastIndex]
#                 vf.PlotSensorValuePoint(SensorValues4ATCRBSDLine3,ID,Point,linecolor = colorlineList[ID] ,markercolor = colorlineList[ID],marker=linestyleListSelected[Point], figureName = figName,label='',combinePlots=True,plotLine=False,plotMarker=True,time=True,markerLabel=str(line))

# lineStep=0 #first value of line 0
# SensorValues4ATCRBSDLine4=SensorValues4ATCRBSDTime[0].copy()
# firstIndex=next(iter(SensorValues4ATCRBSDTime[lineStep]))
# SensorValues4ATCRBSDLine4[0]=SensorValues4ATCRBSDTime[lineStep][firstIndex].copy()
# vf.PlotSensorValues(SensorValues4ATCRBSDLine4,markercolor = 'grey',marker = '',figureName = figName,label='',combinePlots=True,labelATC = False)

# lineStep=1 #first value of line 1
# SensorValues4ATCRBSDLine4=SensorValues4ATCRBSDTime[0].copy()
# firstIndex=next(iter(SensorValues4ATCRBSDTime[lineStep]))
# SensorValues4ATCRBSDLine4[0]=SensorValues4ATCRBSDTime[lineStep][firstIndex].copy()
# vf.PlotSensorValues(SensorValues4ATCRBSDLine4,markercolor = 'grey',marker = '',figureName = figName,label='',combinePlots=True,labelATC = False)

# plt.legend()
# plt.grid()
# plt.axis('equal')
# ax.grid(True, 'major', 'both')
# ax.xaxis.set_major_locator(ticker.MaxNLocator(10)) 
# ax.yaxis.set_major_locator(ticker.MaxNLocator(10)) 
# plt.xlim(-0.5, 0.6)
# ax.set_ylabel(r'nodes y in m')
# ax.set_xlabel(r'nodes x in m')
# plt.show()



# #%% figure2 movement over time
# figName='movementOverTime'+'ID'+str(ID)+'Point'+str(Point)
# Fig4=plt.figure(figName)
# ax=Fig4.gca() # get current axes


# timeList=[]
# sensorValuesPointX=[]
# sensorValuesPointY=[]
# for line in SensorValues4ATCRBSDTime:
#     for time in SensorValues4ATCRBSDTime[line]:
#         timeList+=[time]
#         sensorValuesPointX+=[SensorValues4ATCRBSDTime[line][time][ID][Point][0,0]]
#         sensorValuesPointY+=[SensorValues4ATCRBSDTime[line][time][ID][Point][0,1]]
        
# plt.plot(timeList,sensorValuesPointX,label='X model')
# plt.plot(timeList,sensorValuesPointY,label='Y model')

# plt.legend()
# plt.grid()
# # plt.axis('equal')
# ax.grid(True, 'major', 'both')
# ax.xaxis.set_major_locator(ticker.MaxNLocator(10)) 
# ax.yaxis.set_major_locator(ticker.MaxNLocator(10)) 
# # plt.xlim(0, 20)
# # plt.ylim(-1, 1)
# ax.set_ylabel(r'x/y components of Point in m')
# ax.set_xlabel(r'time t in s')
# plt.show()



# #%% figure2 plot and calculate offset/error
# offsetIdeal=np.matrix([[0, 0]])
# offsetRBSD=np.matrix([[0, 0]])

# errorModelIdealList=[]
# # for ID in adeNodes.elements:
# #     for point in range(3):
# for ID in [ID]:
#     for point in [Point]:      
#         figName='ID:'+str(ID)+'Point:'+str(point)
#         Fig3=plt.figure(figName)
#         ax=Fig3.gca() # get current axes
        
        
#         errorModelIdeal=vf.plotErrorPoint(SensorValues4ATCRBSD,SensorValuesIdeal,
#                                           offsetRBSD,offsetIdeal,
#                                           ID,
#                                           point,
#                                           combinePlots=True,
#                                           figureName=figName,
#                                           linecolor = 'blue' ,
#                                           marker='o',
#                                           markercolor='blue',
#                                           label='model-ideal')
#         print('ID:'+str(ID)+'Point:'+str(point))
#         print('errorModelIdeal=',errorModelIdeal,'m')
    
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

# ##############
# meanErrorModelIdealList=np.mean(errorModelIdealList)
# print('meanError:')
# print('meanErrorModelIdeal=',meanErrorModelIdealList,'m')









#%% testCode - deprecated
# for line in lineList:        
#     # line=1 #line to plot
#     ##+++++++++++++++++++++++++++++++++++++++
#     ## Plotting Mesh over Measured Image   
#     currentAvgSensorValuesidealMesh7Lines = {}
#     currentAvgSensorValuesidealMesh7Lines[0] = SensorValues4ATCRBSDTime[line]    
    
#     ##plot mesh
#     # figName='step: '+str(line)
#     # Fig4=plt.figure(figName)
#     # ax=Fig4.gca() # get current axes

#     vf.PlotSensorValues(currentAvgSensorValuesidealMesh7Lines[0],markercolor = 'grey',marker = '',figureName = figName,label='',combinePlots=True,labelATC = False)    
#     # vf.PlotSensorValuesSteps(currentAvgSensoridealMesh,adeNodes,adeMoveList,markercolor = 'red',marker = '^',figureName = figName,label='RBSD',combinePlots=True,labelATC = True)
#     # vf.PlotSensorValuesSteps(currentAvgSensorValuesRBSD,adeNodes,adeMoveList,markercolor = 'orange',marker = 'x',figureName = figName,label='RBSD',combinePlots=True,labelATC = True)

#     plt.legend()
#     plt.grid()
#     plt.axis('equal')
#     ax.grid(True, 'major', 'both')
#     ax.xaxis.set_major_locator(ticker.MaxNLocator(10)) 
#     ax.yaxis.set_major_locator(ticker.MaxNLocator(10)) 
#     plt.xlim(-0.5, 0.6)
#     ax.set_ylabel(r'nodes y in m')
#     ax.set_xlabel(r'nodes x in m')
#     plt.show()

# figName='Trajectories'
# Fig3=plt.figure(figName)
# ax=Fig3.gca() # get current axes

# for nodeNumber in range(maxNodeNumber):
#     vf.PlotSensorValueNodeNumberOverTimeSteps(idealMesh,adeNodes,adeMoveList,nodeNumber+1,linecolor = colorlineList[nodeNumber],markercolor = colorlineList[nodeNumber],marker=markerstyleList[nodeNumber], figureName = 'Trajectories',label='ideal',combinePlots=True)
# # PlotSensorValueNodeNumberOverTimeSteps(SensorValues4ATCRBSD,adeNodes,adeMoveList,i+1,linecolor = 'orange',markercolor = 'orange',marker='x', figureName = 'Trajectories',label='simulation',combinePlots=True)

# # plt.plot(-10,0,color = 'red',linewidth=0,marker='^',label='ideal') #dummy for legend!
# # plt.plot(-10,0,color = 'orange',linewidth=0,marker='o',label='simulation') #dummy for legend!
# plt.legend()
# plt.axis('equal') 
# plt.grid()
# ax.grid(True, 'major', 'both')
# ax.xaxis.set_major_locator(ticker.MaxNLocator(10)) 
# ax.yaxis.set_major_locator(ticker.MaxNLocator(10)) 

# ax.set_ylabel(r'Trajectories of Nodes y in m')
# ax.set_xlabel(r'Trajectories of Nodes x in m')
# plt.xlim(-0.5, 0.6)
# plt.show()

# for nodeNumber in range(maxNodeNumber):
#     vf.PlotSensorValueNodeNumberOverTimeSteps(idealMesh,adeNodes,adeMoveList,nodeNumber+1,linecolor = colorlineList[nodeNumber],markercolor = colorlineList[nodeNumber],marker=markerstyleList[nodeNumber], figureName = 'Trajectories',label='ideal',combinePlots=True)

# markerstyleList=['^','o','x']

# for ID in adeNodes[0].elements:
#     for Point in range(3):
#         vf.PlotSensorValuePoint(idealMesh,ID,Point,linecolor = colorlineList[ID] ,markercolor = colorlineList[ID],marker=markerstyleList[Point], figureName = 'Trajectories',label='ADE'+str(ID)+' P'+str(Point),combinePlots=True)
# idealMesh2 = {}
# for line in range(int(((len(adeMoveList.data))/4-1)*2)):        
#     idealMesh2[line] = idealMesh[line]  
    
# idealMeshSplit7Lines2 = {}
# for line in lineList:        
#     idealMeshSplit7Lines2[line] = idealMeshSplit7Lines[line]  

# SensorValues4ATCRBSD7Lines2 = {}
# for line in lineList:        
#     SensorValues4ATCRBSD7Lines2[line] = SensorValues4ATCRBSD7Lines[line]  
    
# SensorValues4ATCRBSD2 = {}      
# for line in range(int(((len(adeMoveList.data))/4-1)*2)):
#     SensorValues4ATCRBSD2[line] = SensorValues4ATCRBSD[line]

# ID=2
# Point=0
# vf.PlotSensorValuePoint(idealMeshSplit7Lines2,ID,Point,linecolor = colorlineList[ID] ,markercolor = colorlineList[ID],marker=markerstyleList[Point], figureName = 'Trajectories',label='Ideal ADE'+str(ID)+' P'+str(Point),combinePlots=True,plotLine=False)
# vf.PlotSensorValuePoint(idealMesh2,ID,Point,linecolor = colorlineList[ID] ,markercolor = colorlineList[ID],marker=markerstyleList[Point], figureName = 'Trajectories',label='',combinePlots=True,plotMarker=False)
# vf.PlotSensorValuePoint(SensorValues4ATCRBSD7Lines2,ID,Point,linecolor = colorlineList[ID+1] ,markercolor = colorlineList[ID+1],marker=markerstyleList[Point], figureName = 'Trajectories',label='ADE'+str(ID)+' P'+str(Point),combinePlots=True,plotLine=False)
# vf.PlotSensorValuePoint(SensorValues4ATCRBSD2,ID,Point,linecolor = colorlineList[ID+1] ,markercolor = colorlineList[ID+1],marker=markerstyleList[Point], figureName = 'Trajectories',label='',combinePlots=True,plotMarker=False)


# ID=1
# Point=2
# vf.PlotSensorValuePoint(idealMeshSplit7Lines2,ID,Point,linecolor = colorlineList[ID] ,markercolor = colorlineList[ID],marker=markerstyleList[Point], figureName = 'Trajectories',label='ADE'+str(ID)+' P'+str(Point),combinePlots=True,plotLine=False)
# vf.PlotSensorValuePoint(idealMesh2,ID,Point,linecolor = colorlineList[ID] ,markercolor = colorlineList[ID],marker=markerstyleList[Point], figureName = 'Trajectories',label='',combinePlots=True,plotMarker=False)



# plt.plot(-10,0,color = 'red',linewidth=0,marker='^',label='ideal') #dummy for legend!
# plt.plot(-10,0,color = 'orange',linewidth=0,marker='o',label='simulation') #dummy for legend!
# plt.legend()
# plt.axis('equal') 
# plt.grid()
# ax.grid(True, 'major', 'both')
# ax.xaxis.set_major_locator(ticker.MaxNLocator(10)) 
# ax.yaxis.set_major_locator(ticker.MaxNLocator(10)) 

# ax.set_ylabel(r'Trajectories of Nodes y in m')
# ax.set_xlabel(r'Trajectories of Nodes x in m')
# plt.xlim(-0.5, 0.6)
# plt.show()

       














