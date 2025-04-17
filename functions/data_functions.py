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
import csv
import numpy as np
import logging


import pyvista as pv
from vtk import VTK_TRIANGLE, VTK_TETRA

from collections import deque

# from numpy.lib.arraysetops import 


import config
from flask import jsonify
from copy import deepcopy
import json
import pickle

config.init()

class ElementDefinition:
    """
    Class to generate a Mesh of ADEs
    
    """
    def __init__(self):
        self.elements = {}        
        self.elementMarkerPositionEstimate = {}
        self.elementMarkerPosition = {}
        self.nodeDefinition = {}
        self.numberOfNodes=0
        self.SetListOfNodes()
        self.SetConnectionList()
        self.SetBaseNodes()
        self.nodeNumbersToAde = {}
        self.nodeNumbersToAdeIndex = {}
        self.adePointsOfNode = {}
    def AddElementNodes(self,ID,Nodes,setLists = True):
        """
        Add an Element to the Mesh
        """
        if type(ID)==int:
            if len(Nodes) != 3:
                logging.ERROR('Wrong Number of Nodes ('+str(len(Nodes))+')! ')
                raise ValueError
            self.elements[ID] = Nodes
        elif len(ID)==len(Nodes):
            for i in range(len(ID)):
                if len(Nodes[i]) != 3:
                    logging.ERROR('Wrong Number of Nodes ('+str(len(Nodes))+')! ')
                    raise ValueError
                self.elements[ID[i]] = Nodes[i]
        if setLists == True:
            self.SetListOfNodes()
            self.SetConnectionList()
            self.SetNodeNumbersToAde()
            self.SetADEPointsOfNode()

    def ElementsFromConnectionList(self):

        for element in self.connectionList:
            if not self.elements:
                    self.AddElementNodes(element,[1,2,3],setLists=False)
            else:
                Nodes = [0,0,0]
                for connectionEntry in self.connectionList[element]:
                    if connectionEntry[2] in self.elements:
                        auxNodes = self.GetNodeOfElementSide(connectionEntry[2],connectionEntry[3])
                        if connectionEntry[1] == 1:
                            Nodes[0] = auxNodes[1]
                            Nodes[1] = auxNodes[0]
                        if connectionEntry[1] == 2:
                            Nodes[2] = auxNodes[1]
                            Nodes[0] = auxNodes[0]
                        if connectionEntry[1] == 3:
                            Nodes[1] = auxNodes[1]
                            Nodes[2] = auxNodes[0]
                if min(Nodes) == 0:
                    self.SetListOfNodes()
                    if Nodes[0] == 0:
                        self.numberOfNodes += 1
                        Nodes[0] = self.numberOfNodes
                    if Nodes[1] == 0:
                        self.numberOfNodes += 1
                        Nodes[1] = self.numberOfNodes
                    if Nodes[2] == 0:
                        self.numberOfNodes += 1
                        Nodes[2] = self.numberOfNodes
                self.AddElementNodes(element,Nodes,setLists=False)
                        
    def SetBaseNodes(self,Nodes = [1,2]):
        """
        Set the base Nodes
        """
        self.baseNodes = Nodes

    def SetListOfNodes(self):
        """
        Set the List of Nodes
        """
        listOfNodes = []
        for ID in self.elements:
            setListOfNodes = set(listOfNodes)
            setNodesOfID = set(self.elements[ID])
            notinList = setNodesOfID - setListOfNodes
            listOfNodes.extend(notinList)
        self.listOfNodes = listOfNodes
        self.numberOfNodes = len(listOfNodes)

    def SetConnectionList(self):
        """
        Set the connection List from the added Elements
        """
        connectionList = {}
        secondaryElementsList = deepcopy(self.elements)
        for ID in self.elements:
            connectionList[ID] = []
            setFirstElement = set(self.elements[ID])
            #del secondaryElementsList[ID]
            #####
            ## Check for Intersection Nodes between elements 
            for ID2 in secondaryElementsList:
                if ID == ID2:
                    continue
                intersection = list(setFirstElement.intersection(self.elements[ID2]))
                if len(intersection) == 2:
                    connectionList[ID].append([ID,self.GetSideOfElement(ID,intersection),ID2,self.GetSideOfElement(ID2,intersection)])

            ####
            ## Check for Intersection with Boundary Nodes 
            intersectionBoundary = list(setFirstElement.intersection(self.baseNodes))
            if len(intersectionBoundary) == 2:
                connectionList[ID].append([ID,self.GetSideOfElement(ID,intersectionBoundary),-1,1])
        self.connectionList = connectionList

    def ConnectionListFromMoveListLine(self,adeIDs,adeMoveList):
        connectionList = {}
        for ID in adeIDs:
            connectionList[ID] = []
            index = adeMoveList.GetIndexOfID(ID)
            if index != None:
                if adeMoveList.data[0,index+4] != 0:
                    firstSide = 1
                    if adeMoveList.data[0,index+4] <= -1:
                        secondSide = 1
                    else:
                        index2 = adeMoveList.GetIndexOfID(adeMoveList.data[0,index+4])
                        if adeMoveList.data[0,index2+4] == ID:
                            secondSide = 1
                        elif adeMoveList.data[0,index2+5] == ID:
                            secondSide = 2
                        elif adeMoveList.data[0,index2+6] == ID:
                            secondSide = 3
                    connectionList[ID].append([ID,firstSide,adeMoveList.data[0,index+4],secondSide])
                if adeMoveList.data[0,index+5] != 0:
                    firstSide = 2
                    if adeMoveList.data[0,index+5] <= -1:
                        secondSide = 1
                    else:
                        index2 = adeMoveList.GetIndexOfID(adeMoveList.data[0,index+5])
                        if adeMoveList.data[0,index2+4] == ID:
                            secondSide = 1
                        elif adeMoveList.data[0,index2+5] == ID:
                            secondSide = 2
                        elif adeMoveList.data[0,index2+6] == ID:
                            secondSide = 3
                    connectionList[ID].append([ID,firstSide,adeMoveList.data[0,index+5],secondSide])
                if adeMoveList.data[0,index+6] != 0:
                    firstSide = 3
                    if adeMoveList.data[0,index+6] <= -1:
                        secondSide = 1
                    else:
                        index2 = adeMoveList.GetIndexOfID(adeMoveList.data[0,index+6])
                        if adeMoveList.data[0,index2+4] == ID:
                            secondSide = 1
                        elif adeMoveList.data[0,index2+5] == ID:
                            secondSide = 2
                        elif adeMoveList.data[0,index2+6] == ID:
                            secondSide = 3
                    connectionList[ID].append([ID,firstSide,adeMoveList.data[0,index+6],secondSide])

            # Boundary Connection, if not in MoveList
            if not config.simulation.boundaryConnectionInfoInMoveList and ID in self.elements:
                setFirstElement = set(self.elements[ID])
                intersectionBoundary = list(setFirstElement.intersection(self.baseNodes))
                if len(intersectionBoundary) == 2:
                    connectionList[ID].append([ID,self.GetSideOfElement(ID,intersectionBoundary),-1,1])
        self.connectionList = connectionList

    def GetSideOfElement(self,ID,Nodes):
        """
        Return the Side of an Element between two Nodes
        """
        if len(Nodes) != 2:
            logging.ERROR('Wrong Number of Nodes ('+str(len(Nodes))+')! ')
            raise ValueError
        if all(item in self.elements[ID] for item in Nodes):
            FirstIndex = self.elements[ID].index(Nodes[0])
            SecondIndex = self.elements[ID].index(Nodes[1])

            if (FirstIndex == 0 and SecondIndex == 1) or (FirstIndex == 1 and SecondIndex == 0):
                ElementSide = 1
            elif (FirstIndex == 1 and SecondIndex == 2) or (FirstIndex == 2 and SecondIndex == 1):
                ElementSide = 3
            elif (FirstIndex == 0 and SecondIndex == 2) or (FirstIndex == 2 and SecondIndex == 0):
                ElementSide = 2
        else:
            logging.ERROR('Nodes not in Element! ')
            raise ValueError
        return ElementSide

    def SetNodeNumbersToAde(self):
        # adeList=adeNodes.elements
        adeList=self.elements
        GetNodeNumbersToAde = {}
        listOfNodesNew=[]
        for ADEs in adeList.keys():
            for nodes in adeList[ADEs]:    
                if ADEs in GetNodeNumbersToAde and nodes not in listOfNodesNew:
                    GetNodeNumbersToAde[ADEs].append(nodes)
                else:
                    if ADEs not in GetNodeNumbersToAde:
                        GetNodeNumbersToAde[ADEs] = []
                    if nodes not in listOfNodesNew:
                        GetNodeNumbersToAde[ADEs].append(nodes)
                listOfNodesNew+=[nodes]
        self.nodeNumbersToAde = GetNodeNumbersToAde
        
        nodeNumbersToAdeIndex2={} 
        for pointOfADEToPlot in self.nodeNumbersToAde:
            nodeNumbersToAdeIndex2[pointOfADEToPlot] = []
            for nodeNumber in self.nodeNumbersToAde[pointOfADEToPlot]:  
                nodeNumbersToAdeIndex2[pointOfADEToPlot].append(self.elements[pointOfADEToPlot].index(nodeNumber))
        self.nodeNumbersToAdeIndex = nodeNumbersToAdeIndex2        
        
    def SetADEPointsOfNode(self):
        self.adePointsOfNode = {}
        for node in self.listOfNodes:
            self.adePointsOfNode[node]= {}
            for ID in self.elements:
                if node in self.elements[ID]:
                    self.adePointsOfNode[node][ID] = self.elements[ID].index(node)

    def GetNodeOfElementSide(self,ID,Side):
        """
        Return the Nodes of an Element Side
        """
        if Side == 1:
            Nodes = [self.elements[ID][0],self.elements[ID][1]]
        if Side == 3:
            Nodes = [self.elements[ID][1],self.elements[ID][2]]
        if Side == 2:
            Nodes = [self.elements[ID][2],self.elements[ID][0]]
        return Nodes

    def ReadAbaqusInputFile(self, filename):
         inpFile = open(filename)
         nodes = False
         elements = False

         nodesDict = {}
         elementsDict = {}
         for line in inpFile:
            line = line.strip()
            if line =="*Node":
                 nodes = True
            if line == "*Element, type=CPS3":
                nodes = False
                elements = True
            if line == "*End Part":
                elements = False
                # self.SetBaseNodes(self.elements[1][0:2])
                self.SetConnectionList()
                self.SetListOfNodes()
                return
            if nodes == True and line !="*Node":
                aux = np.array(list(map(str.strip, line.split(',')))).astype(float)
                self.nodeDefinition[int(aux[0])] = aux [1:]
            if elements == True and line != "*Element, type=CPS3":
                aux = list(map(int, map(str.strip, line.split(','))))
                self.AddElementNodes(aux[0],aux[1:],setLists=False)

    def ReadMeshFile(self,file):
        with open(os.path.join(os.getcwd(),file), newline= '') as meshFile: 
            reader = csv.reader(meshFile,delimiter=' ',skipinitialspace=True)
            data = list(reader)
        #read nodes
        nNodes = int(data[0][0])
        if len(data[1])==2 or not data[1][2]:
             nDim = 2
        else:
             nDim = 3
        nodes = np.array([el[0:nDim] for el in data[1:nNodes+1]]).astype(float)
        #nodes*=1.3
        #nodeX = nodeCoordinates[:,0]
        #nodeY = nodeCoordinates[:,1]
        #read trigs
        offsetCells = nNodes+1
        nCells = int(data[offsetCells][0])
        #first element is not needed
        nodesPerCell = len(data[offsetCells+1])
        ATCs = np.array([el[1:nodesPerCell] for el in data[offsetCells+1 :offsetCells+nCells+1]]).astype(int)
        
        #adapt indices to comply with python indexing (starting with 0)
        ATCs = ATCs -1 

        #read edges     
        #offsetEdges = nNodes+nTrigs+2
        #nEdges = int(data[nNodes+nTrigs+2][0])
        #edges = np.array([el[1:3] for el in data[offsetEdges+1 :offsetEdges+nEdges+1]]).astype(int)   
        celltypes = np.empty(nCells, dtype=np.uint8)
        cells = np.insert(ATCs,0,len(ATCs[0]),axis=1).ravel()
        #create cells for mesh generation if 2D
        nodeCoordinates = np.insert(nodes,2,0,axis=1)
        #each cell is a VTK_TETRA
        celltypes[:] = VTK_TRIANGLE
        #create mesh 
        mesh = pv.UnstructuredGrid(cells, celltypes, nodeCoordinates) 
           
        #compute topology
        # as np.array
        # topology = np.empty((0,3),int)
        topology=[]
        for cell in range(mesh.n_cells):
            neighbors = getCellNeighbors(mesh, cell)
            topology.append(neighbors)
            #topology=np.vstack([topology,neighbors])   
        rootATC=0    
        ATCsorted =[rootATC]
        shiftIndex = topology[rootATC].index(-1)
        topology[rootATC] = topology[rootATC][shiftIndex:] +topology[rootATC][:shiftIndex]
        ATCs2process = deque([rootATC])
        while ATCs2process:
            nextATC = ATCs2process.popleft()
            # check right and left children
            rightChildATC = topology[nextATC][1]
            leftChildATC = topology[nextATC][2]
            childATCs=[rightChildATC, leftChildATC]
            for childATC in childATCs:
                if childATC!=-1: #no surface
                    if childATC not in ATCsorted: # ATC already processed                
                        # append to processing queue
                        ATCs2process.append(childATC)
                        ATCsorted.append(childATC)
                        # rearange topology
                        ParentIndex = topology[childATC].index(nextATC)
                        topology[childATC] = topology[childATC][ParentIndex:]+topology[childATC][:ParentIndex]
                        if topology[childATC][0]==-1:
                            print(topology[childATC])
                            raise ValueError ("wrong topology order of ATC {:}!".format(childATC))
        # for nodeID in range(parts.nNodes):
        #     self.nodeDefinition[nodeID]=parts.nodes[nodeID]
        
        # for i in range(parts.nATCs):
        #     self.AddElementNodes(ID= i, Nodes = list(parts.ATCs[i]))
        #  adeIDs = range(parts.nATCs)     
        
        scalingFactor=config.lengths.L0
        for nodeID in range(nNodes):
            self.nodeDefinition[nodeID+1]=nodes[nodeID]*scalingFactor

        for i in range(len(ATCs)):
            self.AddElementNodes(ID= int(i)+1, Nodes = list(ATCs[i]+1),setLists=False)
        # adeIDs = list(range(1,parts.nATCs+1))   
        adeIDs = list(np.array(ATCs) +1)
        
        baseNode1 = ATCs[0][0]+1
        baseNode2 = ATCs[0][1]+1
        self.SetBaseNodes(Nodes=[baseNode1,baseNode2])
        self.SetConnectionList()
        self.SetListOfNodes()
        self.SetADEPointsOfNode()
         
class adeMoveList:
    """
    Class that contains the ADE Movelistdata, as well as functions to read, save and edit it
    """
    def __init__(self,fileName = None):
        if fileName != None:
            self.ReadFile(fileName)
        else:
            self.data = np.matrix([])

    def ReadFile(self,fileName):
        """
        read the movelist from a save file
        """
        self.data = np.asmatrix(np.genfromtxt(fileName,delimiter = ' ',dtype = float))

    def FillADE(self,row,adeLst):
        """
        fill the Actor and Connector settings of the ades defined in adeLst
        """
        for ade in adeLst:
            for n in range(int(self.data.shape[1]/7)):
                index = 1+n*7

                if ade.ID == self.data[row,index]:
                    ade.S1 = self.data[row,index+1]
                    ade.S2 = self.data[row,index+2]
                    ade.S3 = self.data[row,index+3]
                    ade.M1 = self.data[row,index+4]
                    ade.M2 = self.data[row,index+5]
                    ade.M3 = self.data[row,index+6]

    def Refresh(self):
        """
        refresh the indexing of the movelist
        """
        for i in range(len(self.data[:,1])):
            self.data[i,1] = i+1

    def GetIndexOfID(self,ID):
        for i in range(int((self.data.shape[1]-1)/7)):
                if self.data[0,int(i*7)+1] == ID:
                    return (int(i*7)+1)

    def Open(self, openValue = 20):
        lastLine = self.data[-1,:]
        newLine = lastLine.copy()
        newLine.data[0,0] = lastLine[0,0] +1
        for n in range(int(lastLine.shape[1]/7)):
            index = 1+n*7
            newLine.data[0,index+1] = min(lastLine[0,index+1] + openValue,1000)
            newLine.data[0,index+2] = min(lastLine[0,index+2] + openValue,1000)
            newLine.data[0,index+3] = min(lastLine[0,index+3] + openValue,1000)

        self.data = np.r_[self.data,newLine]

    def Close(self, closeValue = 20):
        lastLine = self.data[-1,:]
        newLine = lastLine.copy()
        newLine.data[0,0] = lastLine[0,0] +1
        for n in range(int(lastLine.shape[1]/7)):
            index = 1+n*7
            newLine.data[0,index+1] = max(lastLine[0,index+1] - closeValue,0)
            newLine.data[0,index+2] = max(lastLine[0,index+2] - closeValue,0)
            newLine.data[0,index+3] = max(lastLine[0,index+3] - closeValue,0)
        self.data = np.r_[self.data,newLine]
            
    def ToJson(self):
        returndata = deepcopy(self)
        returndata.data = returndata.data.tolist()
        return json.dumps(returndata.data)

    def FromJson(self,data):
        self.data = np.asmatrix(json.loads(data))

    def ConvertConnectionInfoToMagnetCommands(self):
        data2 = deepcopy(self.data)
        for j in range(1,data2.shape[1],7):
            for k in range(4,7):
                self.data[0,j+k] = 0
        for i in range(1,data2.shape[0]):
            for j in range(1,data2.shape[1],7):
                for k in range(4,7):
                    if data2[i,j+k] == data2[i-1,j+k]:
                        self.data[i,j+k] = 0
                    elif data2[i,j+k] == 0 and data2[i-1,j+k] > 0:
                        self.data[i,j+k] = 2
                    elif data2[i,j+k] > 0:
                        #self.data[i-1,j+k] = 1
                        self.data[i,j+k] = 1
                    elif data2[i,j+k] < 0 or data2[i-1,j+k] < 0:
                        self.data[i,j+k] = 0

    def Split(self,maxStepsize = 200, keepConnectionInfo = False, variableStepsize=True):
        import numpy as np
        A = deepcopy(self.data)
        

        AUpdate = []
        for m in range(1, A.shape[0]):
            maxSteps = round(np.max(np.abs(self.data[m,:]-self.data[m-1,:])/maxStepsize))
            if maxSteps != 0:
                ANew = np.zeros((maxSteps, A.shape[1]),dtype=int)
            else:
                ANew = np.zeros((1, A.shape[1]),dtype=int)
                ANew[0,:] = A[m-1,:]
                AUpdate.append(ANew)
                continue
            ANew[0,:] = A[m-1,:]
            #LengthOfAnew = np.zeros((1, A.shape[1]))
            for i in range(1,A.shape[1],7):
                for j in range(1,4):
                    if variableStepsize:
                        stepsize =  round(np.abs((A[m,i+j]-A[m-1,i+j]))/maxSteps)
                    else:
                        stepsize = maxStepsize
                    for n in range(1, maxSteps):
                        ANew[n, i] = A[m-1, i]
                        if (A[m, i+j] - A[m-1, i+j]) != 0:
                            #steps = round(abs(A[m, i+j] - A[m-1, i+j]) / stepsize)
                            
                            if (A[m, i+j] - A[m-1, i+j]) > 0:
                                if A[m, i+j] > ANew[n-1, i+j] + stepsize:
                                    ANew[n, i+j] = ANew[n-1, i+j] + stepsize
                                else: 
                                    ANew[n, i+j] = A[m, i+j]
                            else:
                                if A[m, i+j] < ANew[n-1, i+j] - stepsize:
                                    ANew[n, i+j] = ANew[n-1, i+j] - stepsize
                                else: 
                                    ANew[n, i+j] = A[m, i+j]
                            #LengthOfAnew[0, i+j] = maxSteps+1
                        else:
                            #LengthOfAnew[0, i+j] = 1
                            ANew[n, i+j] = A[m, i+j]
                if keepConnectionInfo:
                    for j in range(4,7):
                        for n in range(1,maxSteps):
                            if A[m, i+j] == 0 and A[m-1, i+j] >= 0:
                                ANew[n, i+j] = A[m, i+j]
                            else:
                                ANew[n, i+j] = A[m-1, i+j]
                        #ANew[maxSteps-1,i+j] = A[m, i+j]


            #for i in range(0,A.shape[1]-1,7):
            #    for j in range(3):
            #        if (LengthOfAnew[0, i+j] > 0):
            #            ANew[int(LengthOfAnew[i+j]):, i+j] = A[m, i+j]
            
            AUpdate.append(ANew)
        AUpdate.append(A[-1,:])
        AUpdate = np.concatenate(AUpdate)

        for i in range(len(AUpdate)):
            AUpdate[i, 0] = i+1
            
                
        self.data = np.asmatrix(AUpdate)



def NumpyToJson(data):
    data = data.tolist()
    return json.dumps(data)

def NumpyFromJson(data):
    return np.asmatrix(json.loads(data))

def ImageToJson(data):
    #data = data.tolist()
    return pickle.dumps(data)

def ImageFromJson(data):
    return pickle.loads(data)

class structure:
    pass

def getCellNeighbors(mesh, cellID):
    """!get neighbor cell IDs
    @param mesh mesh as pyvista.core.pointset.UnstructuredGrid
    @param cellID ID of cell for retrieving neighbors
    @returns neighbors of cell with @b cellID as list
    """
    cell = mesh.GetCell(cellID)
    nodeIDs = pv.vtk_id_list_to_array(cell.GetPointIds())
    #ToDo: implement 3D neighbors?
    neighbors = []
    for nodeIndex in range(len(nodeIDs)):        
        neighbor = set(mesh.extract_points(nodeIDs[nodeIndex])["vtkOriginalCellIds"]).intersection(mesh.extract_points(nodeIDs[nodeIndex-1])["vtkOriginalCellIds"])
        #for 3D
        if mesh.cells[0]==4:
            neighbor = neighbor.intersection(mesh.extract_points(nodeIDs[nodeIndex-2])["vtkOriginalCellIds"])
        neighbor.discard(cellID)
        if len(neighbor)==1:
            neighbors.append(neighbor.pop())
        elif len(neighbor)==0:
            neighbors.append(-1)
        else:
            raise ValueError('Set of neighbors is to large!')
    #return as np.array
    #return np.array(neighbors)
    return neighbors


if __name__ == "__main__":
    import doctest     # importing doctest module
    doctest.testmod()  # activating test mode
