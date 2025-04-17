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

if __name__ == "__main__":
    if os.getcwd().split('\\')[-1].__contains__('functions'):
        os.chdir('..')
    if sys.path[0].split('\\')[-1].__contains__('functions'):
        sys.path.append(os.getcwd())
    
import numpy as np
import logging
from functions import com_functions as cf
from functions import data_functions as df
from functions import marker_functions as mf
from functions import measurement_functions as mef
from functions import webserver_functions as wf

from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from copy import deepcopy
import config
import cv2
import time
import json

def PlotSensorValues(SensorValues,linecolor = 'black',linestyle='-',markercolor = 'red',marker='', figureName = '',label='',combinePlots=False,labelATC = False):
    i=0
    for line in SensorValues:
        if combinePlots:
            plt.figure(figureName)
        else:
            plt.figure(figureName + 'MoveList Line: '+str(line))



        for ID in SensorValues[line]:

            if i==0:
                label=label
                i=i+1
            else:
                label=''            

            A = np.asmatrix(SensorValues[line][ID][0])
            B = np.asmatrix(SensorValues[line][ID][1])
            C = np.asmatrix(SensorValues[line][ID][2])
            plt.scatter(A[0,0],A[0,1],color=markercolor, marker=marker,label='_nolegend_')
            plt.scatter(B[0,0],B[0,1],color=markercolor, marker=marker,label='_nolegend_')
            plt.scatter(C[0,0],C[0,1],color=markercolor, marker=marker,label='_nolegend_')
            mef.PlotTriangle( A,B,C, linecolor,label=label, linestyle=linestyle)
            if labelATC:
                mef.LabelADE(A,B,C,ID,scale = 1)

        plt.grid(True)
        plt.axis('equal')            

def PlotSensorValuePoint(SensorValues,ID,Point,linecolor = 'black',lineStyle='-',markercolor = 'red',marker='', figureName = '',label='',combinePlots=False,plotLine=True,plotMarker=True,time=False,markerLabel='',printMarkerLabel=True,everyTwoLabels=False,markerSize=100,linewidths=2):
    if combinePlots:
        plt.figure(figureName)
    else:
        plt.figure(figureName + 'Point' + str(ID) + str(Point))

    A = np.zeros((len(SensorValues),2))
    i = 0
    for line in SensorValues:
        A[i]=SensorValues[line][ID][Point]
        if plotMarker:
            if printMarkerLabel: 
                if not time:
                    if not everyTwoLabels:
                        txt=line
                    else:
                        if int(line) % 4 == 0:
                            txt=line
                        else:
                            txt=''
                else:
                    txt=markerLabel
            else:
                txt=''
            plt.scatter(A[i,0],A[i,1],color=markercolor, marker=marker, label='_nolegend_',s=markerSize,linewidths=linewidths)
            plt.text(A[i,0]-0.005,A[i,1]+0.005,str(txt),fontsize=12)
        i += 1
        
    if plotLine:
        plt.plot(A[:,0],A[:,1],color = linecolor,linestyle=lineStyle,linewidth=1,label='_nolegend_')
            
    plt.plot(-10,0,color = linecolor,linestyle=lineStyle,linewidth=0,marker=marker,label=label) #dummy for legend!
    
    plt.grid(True)
    plt.axis('equal')    


def plotErrorPoint(SensorValues1,SensorValues2,offsetOfLine1, offsetOfLine2,ID,Point,linecolor = 'black',markercolor = '', figureName = '',label='',marker='',combinePlots=False,plotFigures=True):
    if plotFigures:
        if combinePlots:
            plt.figure(figureName)
        else:
            plt.figure(figureName + 'Point' + str(Point))

    errorNorm=0
    errorMax=['0',0]
    errorList=[]
    for line in SensorValues1:        
        A = (SensorValues1[line][ID][Point]-offsetOfLine1) - (SensorValues2[line][ID][Point]-offsetOfLine2) 
    
        error=np.linalg.norm(A*1000)
        if error > errorMax[1]:
            errorMax[0]='line:'+str(line)
            errorMax[1]=error
            
        errorNorm+=error                
        errorList+=[error]
        if plotFigures:
            plt.scatter(line,error,color=markercolor, marker=marker, label='_nolegend_')
                   
    if plotFigures:
        plt.plot(-10,0,color = linecolor,linewidth=0,marker=marker,label=label) #dummy for legend!
    return [errorNorm/(line+1),errorMax,errorList]

def fromSimulationTimeToContinuousTime(SensorValues4ATCRBSDImport):
    #%% sortTime
    SensorValues4ATCRBSDTime = {}
    accumulatedTime=0
    for line in SensorValues4ATCRBSDImport:
        SensorValues4ATCRBSDTime[line]={}
        lastTime=0
        for newTime in SensorValues4ATCRBSDImport[line]:
            dT=newTime-lastTime    
            if dT<0:
                dT=0
            accumulatedTime+=dT        
            SensorValues4ATCRBSDTime[line][accumulatedTime]=SensorValues4ATCRBSDImport[line][newTime]
            lastTime=newTime
            
    return SensorValues4ATCRBSDTime

def plotTrajectoriesTime(ID,Point2,SensorValues4ATCRBSDTime, adeMoveList, colorlineList, lineStyle, markerstyleListSelected, figureName, label='', printMarkerLabel=True,printLegend=False):
    SensorValues4ATCRBSDLine2=deepcopy(SensorValues4ATCRBSDTime[0])
    SensorValues4ATCRBSDLine3=deepcopy(SensorValues4ATCRBSDTime[0])
    SensorValues4ATCRBSDLine4=deepcopy(SensorValues4ATCRBSDTime[0])
    for ID2 in ID:
        for Point in Point2:        
            for line in SensorValues4ATCRBSDTime:
                if printMarkerLabel:
                    markerLabel=str(line)
                else:
                    markerLabel=''
                SensorValues4ATCRBSDLine2=deepcopy(SensorValues4ATCRBSDTime[line])
                PlotSensorValuePoint(SensorValues4ATCRBSDLine2,ID2,Point,linecolor = colorlineList,lineStyle=lineStyle ,markercolor = colorlineList,marker=markerstyleListSelected, figureName = figureName,label='',combinePlots=True,plotMarker=False)
    
                firstIndex=next(iter(SensorValues4ATCRBSDTime[line]))              
                SensorValues4ATCRBSDLine3[0]=deepcopy(SensorValues4ATCRBSDTime[line][firstIndex])
                PlotSensorValuePoint(SensorValues4ATCRBSDLine3,ID2,Point,linecolor = colorlineList,lineStyle=lineStyle ,markercolor = colorlineList,marker=markerstyleListSelected, figureName = figureName,label='',combinePlots=True,plotLine=False,plotMarker=True,time=True,markerLabel=markerLabel)
                       
                if line == len(adeMoveList.data)-1: #last step
                    lastIndex=list(SensorValues4ATCRBSDTime[len(adeMoveList.data)-1]).pop()       
                    SensorValues4ATCRBSDLine4[0]=deepcopy(SensorValues4ATCRBSDTime[line][lastIndex])
                    PlotSensorValuePoint(SensorValues4ATCRBSDLine4,ID2,Point,linecolor = colorlineList,lineStyle=lineStyle ,markercolor = colorlineList,marker=markerstyleListSelected, figureName = figureName,label='',combinePlots=True,plotLine=False,plotMarker=True,time=True,markerLabel=markerLabel)
    
        if printLegend:
            plt.plot(-10,0,color = colorlineList,linewidth=0,marker=markerstyleListSelected,label=label) #dummy for legend!

def MoveToOrigin(sensorValues,adeNodes,ID,Side,additionalRotation=0):
        Nodes = adeNodes.GetNodeOfElementSide(ID,Side)
        index1 = adeNodes.elements[ID].index(Nodes[0])
        index2 = adeNodes.elements[ID].index(Nodes[1])
        P1=sensorValues[0][ID][index1]
        P2=sensorValues[0][ID][index2]
        
        if LA.norm(P2) < LA.norm(P1):
            P1,P2 = P2,P1

        Bound1=P1.T
        Bound2=P2.T#P1+(P2-P1)/LA.norm(P2-P1)*(B1-29)
        an = (P2-P1)/LA.norm(P2-P1) 
        
        ex = np.array([[1],[0]])
        bound = P2-P1
        alpha = -math.acos(an[0,0])+additionalRotation

        P1=P1.T
        
        for line in sensorValues:
            for ID in sensorValues[line]:
                for i in range(3):
                    sensorValues[line][ID][i] = np.asarray(np.matmul(mef.RotAz(alpha),np.matrix(sensorValues[line][ID][i].T-P1))).T

def plotMesh(SensorValues4ATCRBSDTime,lineStep,plotFirstIndex=True,figureName='',labelATC=False,linestyle='-'):
    SensorValues4ATCRBSDLine4=SensorValues4ATCRBSDTime[0].copy()
    if plotFirstIndex:
        firstIndex=next(iter(SensorValues4ATCRBSDTime[lineStep]))
    else:
        firstIndex=list(SensorValues4ATCRBSDTime[lineStep]).pop() 
        
    SensorValues4ATCRBSDLine4[0]=SensorValues4ATCRBSDTime[lineStep][firstIndex].copy()
    PlotSensorValues(SensorValues4ATCRBSDLine4,linecolor = 'black',linestyle=linestyle,markercolor = 'grey',marker = '',figureName = figureName,label='',combinePlots=True,labelATC = labelATC)    

def plotComponentsOverTime(ID,Point,SensorValuesTime,colorLine,lineStyle,markerStyle='x',label='',printLegend=False,printMarkerLabel=False,plotComponent=[-1]):
    timeList=[]
    sensorValuesPointX=[]
    sensorValuesPointY=[]
    
    sensorValuesPointX0=[]
    sensorValuesPointY0=[]
    timeList0=[]
    alreadyPlotted=False
    for line in SensorValuesTime:
        for time in SensorValuesTime[line]:
            timeList+=[time]
            sensorValuesPointX+=[SensorValuesTime[line][time][ID][Point][0,0]]
            sensorValuesPointY+=[SensorValuesTime[line][time][ID][Point][0,1]]
            
        if line < len(SensorValuesTime)-1:
            firstIndex=next(iter(SensorValuesTime[line]))
        else:
            firstIndex=next(iter(SensorValuesTime[line]))
            timeList0+=[firstIndex]
            sensorValuesPointX0+=[SensorValuesTime[line][firstIndex][ID][Point][0,0]]
            sensorValuesPointY0+=[SensorValuesTime[line][firstIndex][ID][Point][0,1]]             
            firstIndex=list(SensorValuesTime[line]).pop()
            
        timeList0+=[firstIndex]
        sensorValuesPointX0+=[SensorValuesTime[line][firstIndex][ID][Point][0,0]]
        sensorValuesPointY0+=[SensorValuesTime[line][firstIndex][ID][Point][0,1]]  
            
        if printMarkerLabel:
            if -1 in plotComponent or 0 in plotComponent:
                plt.text(timeList0[line]-0.005,sensorValuesPointX0[line]+0.005,str(line),fontsize=12)
            if -1 in plotComponent or 1 in plotComponent:
                plt.text(timeList0[line]-0.005,sensorValuesPointY0[line]+0.005,str(line),fontsize=12)

    if -1 in plotComponent or 0 in plotComponent:
        plt.plot(timeList,sensorValuesPointX,color = colorLine,linestyle=lineStyle,label='_nolegend_')
        plt.scatter(timeList0,sensorValuesPointX0,color = colorLine, marker=markerStyle,label='_nolegend_')
    if -1 in plotComponent or 1 in plotComponent:
        plt.plot(timeList,sensorValuesPointY,color = colorLine,linestyle=lineStyle,label='_nolegend_')
        plt.scatter(timeList0,sensorValuesPointY0,color = colorLine, marker=markerStyle,label='_nolegend_')

    if printLegend:
        plt.plot(-10,0,color=colorLine,linestyle=lineStyle,linewidth=1,marker=markerStyle,label=label) #dummy for legend!    
    # plt.plot(timeList,sensorValuesPointY,color = colorLine,lineStyle=lineStyle,label='Y '+label)

def PlotElementDefinition(adeNodes,linecolor = 'black',markercolor = 'red',marker='o', figureName = ''):

    if adeNodes.nodeDefinition:
        plt.figure(figureName)
        plt.axis('equal')
        plt.grid()
        for ID in adeNodes.elements:
            Nodes = adeNodes.elements[ID]

            A=np.asmatrix(adeNodes.nodeDefinition[Nodes[0]])
            B=np.asmatrix(adeNodes.nodeDefinition[Nodes[1]])
            C=np.asmatrix(adeNodes.nodeDefinition[Nodes[2]])

            plt.scatter(A[0,0],A[0,1],color=markercolor, marker=marker)
            plt.scatter(B[0,0],B[0,1],color=markercolor, marker=marker)
            plt.scatter(C[0,0],C[0,1],color=markercolor, marker=marker)
            mef.PlotTriangle( A,B,C, linecolor )
            mef.LabelADE(A,B,C,ID,scale = 1)
            mef.IdealMesh2idealMeasurement(A,B,C,True,color='green')
    else:
        logging.warning('Node Definition not Set. Use simulation/Generate Mesh or Stroke2desiredMesh to fill Node Definition')


def calculateOffset(SensorValues1,SensorValues2,lineToStart,ID,Point):
    A1=SensorValues1[lineToStart][ID][Point] 
    A2=SensorValues2[lineToStart][ID][Point] 
    return A2-A1


def substractOffsetSensorValues(SensorValues,SensorValues2,ID,Point,lineToStart=0):
    
    SensorValuesSubstractOffset=deepcopy(SensorValues)
    
    for k in ID:
        for j in Point:
            
            offsetOfLine=calculateOffset(SensorValues2,SensorValues,lineToStart,k,j)
          
            A = np.zeros((len(SensorValues),2))
            i = 0      
            
            for line in SensorValues:
                A[i]=SensorValues[line][k][j]-offsetOfLine
                SensorValuesSubstractOffset[line][k][j]=A[i]
                i += 1
 
    return SensorValuesSubstractOffset



def substractOffset(SensorValues,ID,Point,offsetOfLine=0,linecolor = 'black',markercolor = '', figureName = '',label='',marker='',combinePlots=False):
    if combinePlots:
        plt.figure(figureName)
    else:
        plt.figure(figureName + 'Point' + str(ID) + str(Point))    
    
    SensorValuesSubstractOffset=deepcopy(SensorValues)
    
    A = np.zeros((len(SensorValues),2))
    i = 0
       
    for line in SensorValues:
        A[i]=SensorValues[line][ID][Point]-offsetOfLine
        SensorValuesSubstractOffset[line][ID][Point]=A[i]
        plt.scatter(A[i,0],A[i,1],color=markercolor, marker=marker, label='_nolegend_')
        plt.text(A[i,0]-0.005,A[i,1]+0.005,str(line),fontsize=12)
        i += 1
    plt.plot(A[:,0],A[:,1],color = linecolor,linewidth=1,marker=marker,label=label)
            
    plt.grid()
    plt.axis('equal')    

    return SensorValuesSubstractOffset

def plotError(SensorValues1,SensorValues2,ID,Point,offsetOfLine1, offsetOfLine2,linecolor = 'black',markercolor = 'black', figureName = '',label='',marker='',combinePlots=False):
    if combinePlots:
        plt.figure(figureName)
    else:
        plt.figure(figureName + 'Point' + str(ID) + str(Point))    
  
    errorNorm=0
    i=1
    A = np.zeros((len(SensorValues1),2))
    errors = np.zeros((len(SensorValues1),1))

    for lineNr,line in enumerate(SensorValues1):
        A = (SensorValues1[line][ID][Point]-offsetOfLine1) - (SensorValues2[line][ID][Point]-offsetOfLine2)

        error=np.linalg.norm(A)
        errors[lineNr] = error
        errorNorm+=error
        i+=1
        plt.scatter(lineNr,error,color=markercolor, marker=marker, label='_nolegend_')

    #plt.plot(-10,0,color = linecolor,linewidth=0,marker=marker,label=label) #dummy for legend!


    

def plotErrorOffset(ID,Point,SensorValues4ATCRBSD,SensorValuesIdeal,figureName='',combinePlots=False,label='model-ideal',linecolor='blue',marker='o'):
    if combinePlots:
        plt.figure(figureName)
    else:
        plt.figure(figureName + 'Point' + str(Point))  
        
    offsetIdeal=np.matrix([[0, 0]])
    offsetRBSD=np.matrix([[0, 0]])
    
    errorModelIdealList=[]
    
    maxError=['0',0]
    maxErrorMean=['0',0]
    # for ID in adeNodes[0].elements:
    #     for point in range(3):
    for ID in ID:
        for point in Point:        
            if not combinePlots:
                figureName='ID:'+str(ID)+'Point:'+str(point)
                
            Fig3=plt.figure(figureName)
            ax=Fig3.gca() # get current axes
            
            
            [errorModelIdeal, errorMaxVal, errorMax]=plotErrorPoint(SensorValues4ATCRBSD,SensorValuesIdeal,
                                              offsetRBSD,offsetIdeal,
                                              ID,
                                              point,
                                              combinePlots=True,
                                              figureName=figureName,
                                              linecolor = linecolor ,
                                              marker=marker,
                                              markercolor=linecolor,
                                              label=label)
            
            if maxError[1] < errorMaxVal[1]:
                maxError[0]=errorMaxVal[0]+' ID: '+str(ID)+' Point: '+str(point)
                maxError[1]=errorMaxVal[1]

            if maxErrorMean[1] < errorModelIdeal:
                maxErrorMean[0]='ID: '+str(ID)+' Point: '+str(point)
                maxErrorMean[1]=errorModelIdeal
            
            print(str(errorMaxVal[0])+' ID:'+str(ID)+' Point:'+str(point)+' maxError=',errorMaxVal[1],'m')
            print('ID:'+str(ID)+' Point:'+str(point)+' meanError=',errorModelIdeal,'m')

        
            plt.legend()
            plt.grid()
            ax.grid(True, 'major', 'both')
            ax.xaxis.set_major_locator(ticker.MaxNLocator(10)) 
            ax.yaxis.set_major_locator(ticker.MaxNLocator(10)) 
            
            plt.xlim(-1, len(SensorValues4ATCRBSD))
            
            plt.axhline(y = errorModelIdeal, xmin = 0, xmax = len(SensorValues4ATCRBSD), color=linecolor, linestyle='--')
            
            # ax.set_ylabel(r'point P6 error in m; mean error ='+str(round(errorModelIdeal*1e3,3))+'mm')
            ax.set_ylabel(r'point with max error in m')
            ax.set_xlabel(r'steps')
            plt.show()
            
            errorModelIdealList+=['ID: '+str(ID)+' Point: '+str(point),errorModelIdeal]       
    return [errorModelIdealList,maxError,maxErrorMean]


def plotErrorOffsetBoxPlot(ID,Point,SensorValues4ATCRBSD,SensorValuesIdeal,figureName='',combinePlots=False,label='ATC6',linecolor='blue',marker='o',plotOverSteps=False,setOffset=0,facecolor='C0'):
    if combinePlots:
        # plt.figure(figureName)
        Fig4=plt.figure(figureName)
        ax=Fig4.gca() # get current axes
    else:
        plt.figure(figureName + 'Point' + str(Point))  
        
    offsetIdeal=np.matrix([[0, 0]])
    offsetRBSD=np.matrix([[0, 0]])
    
    errorModelIdealList=[]
    errorModelIdealListValues={}
    
    maxError=['0',0]
    maxErrorMean=['0',0]
    # for ID in adeNodes[0].elements:
    #     for point in range(3):
    for ID in ID:
        errorModelIdealListValues[ID]={}
        for point in Point:    
            
            errorModelIdealListValues[ID][point]={}
            # if not combinePlots:
            #     figureName='ID:'+str(ID)+'Point:'+str(point)
                
            # Fig3=plt.figure(figureName)
            # ax=Fig3.gca() # get current axes
            
            
            [errorModelIdeal, errorMaxVal,errorList]=plotErrorPoint(SensorValues4ATCRBSD,SensorValuesIdeal,
                                              offsetRBSD,offsetIdeal,
                                              ID,
                                              point,
                                              combinePlots=True,
                                              figureName='test',
                                              linecolor = linecolor ,
                                              marker=marker,
                                              markercolor=linecolor,
                                              label='test',
                                              plotFigures=False)
            
            if maxError[1] < errorMaxVal[1]:
                maxError[0]=errorMaxVal[0]+' ID: '+str(ID)+' Point: '+str(point)
                maxError[1]=errorMaxVal[1]

            if maxErrorMean[1] < errorModelIdeal:
                maxErrorMean[0]='ID: '+str(ID)+' Point: '+str(point)
                maxErrorMean[1]=errorModelIdeal
            
            print(str(errorMaxVal[0])+' ID:'+str(ID)+' Point:'+str(point)+' maxError=',errorMaxVal[1],'m')
            print('ID:'+str(ID)+' Point:'+str(point)+' meanError=',errorModelIdeal,'m')
  
            errorModelIdealList+=['ID: '+str(ID)+' Point: '+str(point),errorModelIdeal]
            
            errorModelIdealListValues[ID][point]=errorList

    if not plotOverSteps:
        listOfData=[]
        listNumber=[]
        listLabel=[]
        i=1  
        for ID in errorModelIdealListValues:
            for point in errorModelIdealListValues[ID]:
                listOfData+=[errorModelIdealListValues[ID][point]]
                listNumber+=[i+setOffset]
                i+=1
                listLabel+=['ID'+str(ID)+'point'+str(point)]
        
        bp1 = plt.boxplot(listOfData, positions=listNumber, notch=False, widths=0.3, 
                         patch_artist=True, boxprops=dict(facecolor=facecolor))

        plt.xticks(listNumber, listLabel)
        plt.xticks(rotation=45)

    else:
        listOfData=[]
        listNumber=[]
        listLabel=[]
        firstIndex=next(iter(errorModelIdealListValues)) 
        for step in range(len(errorModelIdealListValues[firstIndex][0])):
            listOfDataPerStep=[]
            for ID in errorModelIdealListValues:
                for point in errorModelIdealListValues[ID]:
                    listOfDataPerStep+=[errorModelIdealListValues[ID][point][step]]
                    
            
            listLabel+=[str(step)]        
            listOfData+=[listOfDataPerStep]  
            listNumber+=[step+1+setOffset]
            
        bp1 = plt.boxplot(listOfData, positions=listNumber, notch=False, widths=0.3, 
                         patch_artist=True, boxprops=dict(facecolor=facecolor))
        
        plt.xticks(listNumber, listLabel)

    return [errorModelIdealList,maxError,maxErrorMean,errorModelIdealListValues]


def PlotSensorValueNodeNumberOverTimeSteps(SensorValues,adeNodes,adeMoveList,nodeNumber,linecolor = 'black',markercolor = 'red',marker='', figureName = '',label='',combinePlots=False):
    if combinePlots:
        plt.figure(figureName)
    else:
        plt.figure(figureName + 'NodeNumber' + str(nodeNumber))

    TimeStamp = 0
    adeIDs = []
    
    A = np.zeros((len(SensorValues),2))  
    xList=[]
    yList=[]
    
    if type(adeNodes) == dict:
        for ID in adeNodes[0].elements:
                adeIDs.append(ID)
    else:
        for ID in adeNodes.elements:
                adeIDs.append(ID)
    
    for line in range(len(adeMoveList.data)):
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


        for ADE in adeNodes[TimeStamp].elements:
            if nodeNumber in adeNodes[TimeStamp].elements[ADE]:
                ID=ADE
                Point=adeNodes[TimeStamp].elements[ID].index(nodeNumber)  
                
                # Point=adeNodes[TimeStamp].nodeNumbersToAdeIndex[ID][index]


                A[line]=SensorValues[line][ID][Point]
                xList+=[A[line,0]]
                yList+=[A[line,1]]
                plt.scatter(A[line,0],A[line,1],color=markercolor, marker=marker, label='_nolegend_')
                plt.text(A[line,0]-0.005,A[line,1]+0.005,'N'+str(nodeNumber)+'L'+str(line),fontsize=12)
                break

        if nextTimeStamp == True:
            TimeStamp += 1

    # plt.plot(A[:,0],A[:,1],color = linecolor,linewidth=1,label='_nolegend_')
    # plt.plot(xList,yList,color = linecolor,linewidth=1,label='_nolegend_')
    plt.plot(xList,yList,color = linecolor,linewidth=1,label='N'+str(nodeNumber))  
    # plt.plot(-10,0,color = 'red',linewidth=0,marker='^',label='ideal')

def plotErrorNodeNumberOverTimeSteps(SensorValues1,SensorValues2,offsetOfLine1, offsetOfLine2,adeNodes,adeMoveList,nodeNumber,linecolor = 'black',markercolor = '', figureName = '',label='',marker='',combinePlots=False):
    if combinePlots:
        plt.figure(figureName)
    else:
        plt.figure(figureName + 'NodeNumber' + str(nodeNumber))
  
    
    TimeStamp = 0
    adeIDs = []
    
    errorNorm=0
    # A = np.zeros((len(SensorValues),2))  
    # xList=[]
    # yList=[]
    
    if type(adeNodes) == dict:
        for ID in adeNodes[0].elements:
                adeIDs.append(ID)
    else:
        for ID in adeNodes.elements:
                adeIDs.append(ID)
    
    for line in range(len(adeMoveList.data)):
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


        for ADE in adeNodes[TimeStamp].elements:
            if nodeNumber in adeNodes[TimeStamp].elements[ADE]:
                ID=ADE
                Point=adeNodes[TimeStamp].elements[ID].index(nodeNumber)              
                # Point=adeNodes[TimeStamp].nodeNumbersToAdeIndex[ID][index]
                
                A = (SensorValues1[line][ID][Point]-offsetOfLine1) - (SensorValues2[line][ID][Point]-offsetOfLine2)
                error=np.linalg.norm(A)

                error=np.linalg.norm(A)
                errorNorm+=error                
                
                plt.scatter(line,error,color=markercolor, marker=marker, label='_nolegend_')
                
                # A[line]=SensorValues[line][ID][Point]
                # xList+=[A[line,0]]
                # yList+=[A[line,1]]
                # plt.scatter(A[line,0],A[line,1],color=markercolor, marker=marker, label='_nolegend_')
                # plt.text(A[line,0]-0.005,A[line,1]+0.005,'N'+str(nodeNumber)+'L'+str(line),fontsize=12)
                break

        if nextTimeStamp == True:
            TimeStamp += 1  
    
  
    plt.plot(-10,0,color = linecolor,linewidth=0,marker=marker,label=label) #dummy for legend!
    return errorNorm/(line+1)




if __name__ == "__main__":
    #plt.ion()
    config.init()
    logging.info('Startet Visualization Module')

    client = wf.WebServerClient()
    plt.figure("ADE Points")
    plt.grid()
    plt.ylim([0,config.measurement.yCord])
    plt.xlim([0,config.measurement.xCord])

    while True:
        
        try:
            ##+++++++++++++++++++++++++++++++++++++++
            ## Checking if to end the Programm

            if json.loads(client.Get('EndProgram')):
                os._exit(0)

            plt.figure("ADE Points")
            plt.grid()
            plt.ylim([0,config.measurement.yCord])
            plt.xlim([0,config.measurement.xCord])
            MeasuredCords = df.NumpyFromJson(client.Get('MeasuredCords'))
            mef.SortPointsADEBoundary(MeasuredCords, 1,"ADE Points")
            
            plt.savefig('html/img/detectetMarkerPlot.jpg')
            plt.show(block=False)
        except:
            continue
        plt.pause(config.visualization.looptime-time.time()%config.visualization.looptime)
        plt.clf()