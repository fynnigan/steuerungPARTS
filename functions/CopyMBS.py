#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is an EXUDYN example
#
# Details:  Slider crank test model
#
# Author:   Johannes Gerstmayr
# Date:     2019-11-01
#
# Copyright:This file is part of Exudyn. Exudyn is free software. You can redistribute it and/or modify it under the terms of the Exudyn license. See 'LICENSE.txt' for more details.
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# needed for copying (analysing) structures
# import inspect
# import collections


import exudyn as exu
from exudyn.itemInterface import *


# mbs2 = SC.AddSystem()

# save mbs into dict and load from pickled file; how far work the python functions
def GetMBSCopy(mbs1, SC): 
    mbs2 = SC.AddSystem()
    
    # nodes
    nNodes = mbs1.systemData.NumberOfNodes()
    nLoads = mbs1.systemData.NumberOfLoads()
    nObjects = mbs1.systemData.NumberOfObjects()
    nSensors = mbs1.systemData.NumberOfSensors()
    nMarkers = mbs1.systemData.NumberOfMarkers()
    flagWarningVisualization = False
    
    for i in range(nNodes): 
        # print(mbs1.GetNode(i))
        node_i = mbs1.GetNode(i)
        if node_i['name'][0:4] == 'node': 
           node_i['name'] = '' #  node_i.pop('name') 
        mbs2.AddNode(node_i)
            # getattr(exudyn.itemInterface, node_i['nodeType'])( 
            #           referenceCoordinates=node_i['referenceCoordinates'], 
            #           name=node_i['name'], 
            #           visualization=getattr(exudyn.itemInterface, 'V' + node_i['nodeType'])(
            #               show =node_i['Vshow'], drawSize=node_i['VdrawSize'], color=node_i['Vcolor']
                          # )))
                     
                     # Vshow=node_i['Vshow'],
                      # VdrawSize=node_i['VdrawSize'],
                     # Vcolor=node_i['Vcolor']))
        
    for i in range(nLoads): 
        load_i = mbs1.GetLoad(i)
        if load_i['name'][0:4] == 'load': 
           load_i['name'] = ''
        mbs2.AddLoad(load_i)
    
    for i in range(nObjects): 
        object_i = mbs1.GetObject(i, addGraphicsData=False)
        if object_i['name'][0:6] == 'object': 
           object_i['name'] = ''
        # object_i['VgraphicsData'] = ''+
        keys = list(object_i.keys())
        flagWarningVisualization = False
        for key in keys:
            # print()
            if key[0] == 'V': 
                # visulization does not work yet
                object_i.pop(key)
                flagWarningVisualization = True
        mbs2.AddObject(object_i, )
        

        
    for i in range(nSensors):
        sensor_i = mbs1.GetSensor(i)
        if sensor_i['name'][0:6] == 'sensor': 
           sensor_i['name'] = ''
        mbs2.AddSensor(sensor_i)
        
    for i in range(nMarkers): 
        marker_i = mbs1.GetMarker(i)
        if marker_i['name'][0:6] == 'marker': 
           marker_i['name'] = ''
        mbs2.AddMarker(marker_i)

    if flagWarningVisualization: 
        print('\n******************************')
        print('Warning: copying visulization is currently not supported')
        print('******************************\n')
        
    mbs2.variables = mbs1.variables
    # ToDo: do UserFunctions work? what are the limits of the functionality?
    mbs2.Assemble()
    state_sys1 = mbs1.systemData.GetSystemState()
    mbs2.systemData.SetSystemState(state_sys1)
    # mbs2.systemData.Info() # print info on all system elements
    return mbs2

if __name__ == '__main__': 
    # print(mbs2)
    mbs2 = GetMBSCopy(mbs, SC)
    # print(mbs2)
    # 
    mbsList = [mbs2]
    for i in range(1): 
        mbsList += [GetMBSCopy(mbsList[i], SC)]
    
    
    simulationSettings.solutionSettings.writeSolutionToFile = 1
    m1 = mbs.GetObjectParameter(1, 'physicsMass')
    exu.SolveDynamic(mbs, simulationSettings)
    m1_ = mbs.GetObjectParameter(1, 'physicsMass')
    solution1 = np.loadtxt('coordinatesSolution.txt', delimiter=',')
    print('last value of solution1: ', solution1[-1,-1])
    m2 = mbs2.GetObjectParameter(1, 'physicsMass')
    exu.SolveDynamic(mbsList[-1], simulationSettings)
    m2_ = mbs2.GetObjectParameter(1, 'physicsMass')
    solution2 = np.loadtxt('coordinatesSolution.txt', delimiter=',')
    print('last value of solution2: ', solution1[-1,-1])
    print('m1 goes from {} to {}. '.format(m1, m1_))
    print('m2 goes from {} to {}. '.format(m2, m2_))
    
    if np.linalg.norm(solution1-solution2) == 0: 
        print('copySystem correct!')
        
    # 56 ms ± 264 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    # 66.3 ms ± 168 µs per loop (mean ± std. dev. of 7 runs, 10 loops each) # --> using copied system...
    # 122 ms ± 1.79 ms per loop (mean ± std. dev. of 7 runs, 10 loops each) # copying system 10 times
    # without user funcction there is no noticable difference
    # original: 36.7 ms ± 600 µs | copied 34.9 ms ± 439 µs per loop
    import sys
    sys.exit()
    
    exu.StartRenderer()
    mbs.WaitForUserToContinue()
    
    #++++++++++++++++++++++++++++++++++++++++++
    #solve generalized alpha / index3:
    exu.SolveDynamic(mbs, simulationSettings)
    
    SC.WaitForRenderEngineStopFlag()
    exu.StopRenderer() #safely close rendering window!
    
    
    # ##animate solution
    # #fileName = 'coordinatesSolution.txt'
    # #solution = LoadSolutionFile('coordinatesSolution.txt')
    # #AnimateSolution(mbs, solution, 10, 0.05)
    
