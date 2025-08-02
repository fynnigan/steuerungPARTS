<<<<<<< HEAD
﻿#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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


import numpy as np
import matplotlib
import logging
import cv2

class structure():
    pass

def init():
    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Lengths Config 
    global lengths
    lengths = structure()

    lengths.L0 = 229*1e-3     #Unextended Length of ADEs
    lengths.L1=38*1e-3         #Length L1 between P0 (Hinge) and P1 #37.7418 mm
    lengths.L2=29*1e-3         #Length L2 between P1 and P2
    lengths.L3=8*1e-3          #Length L3 between P1 and P3 
    lengths.L4=10*1e-3         #Length L4 between P1 and MarkerPoint towards P0
    lengths.L5=1*1e-3          #Length L5 between P1 and MarkerPoint 90° rotated towards P0 
    lengths.L100=329*1e-3      #Extended Length of ADEs
    
    lengths.weightATC = 0.365  #wight of one ATC in kg
    
    #weight and inertia of RB0 and RB5
    b=40e-3 # width of actuator
    h=60e-3 # length of actuator    
    
    lengths.inertiaActuator=1/12*(b*h**3)
    lengths.weightActuator = lengths.weightATC/6
    
    #weight and inertia of RB1 - RB4
    #C-shape profile B=3.5mm, b=2mm , H=7mm, h=1mm (Ix)
    B=3.5e-3 #m
    b=2e-3 #m
    H=7e-3 #m
    h=1e-3 #m
    
    b3=B-b
    h4=H-2*h
    Ix=(B*H**3-b3*h4**3)/12
    lengths.inertiaBeams=Ix *2   #mm^4 #2 six-bar linkages (top bottom) 
    lengths.weightBeams=1.0e-3 *2 #2 six-bar linkages (top bottom)
    
    
    
    
    lengths.boundaryXdistance = 0*1e-3    #X Distance from Boundary Marker to P0
    lengths.boundaryYdistance = 9.35*1e-3 #Y Distance from Boundary Marker to P0
    
    #spring for prismaticJoint (Actuator); initial stiffness and damping
    lengths.stiffnessPrismaticJoints = 1e5
    lengths.dampingPrismaticJoints = 5e3
    
    #values for springs for connectors between two elements
    lengths.stiffnessConnectors=6e3
    lengths.dampingConnectors=2e4
    
    lengths.stiffnessBoundary=4e6 #4e4
    lengths.dampingBoundary=3e3  
    
    lengths.maxStroke = 0.100
    lengths.minStroke = 0.0
    
    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Webserver Config
    global webserver

    webserver = structure()
    webserver.pauseBetweenMovementCommands = 10     #Pause Value in s between MovementCommands when running a Movelist from Server 


    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Communication config
    global com
    com = structure()

    com.actorSpeed = 25 #mm/s

    com.timeoutReceive = 20         #Time to wait for ADE Battery Status in s
    com.timeoutSend = 1             #Time to wait for TCPSocket while Sending in s
    com.pauseVal = 0              #Pause Value between Move List rows in s
    com.looptime = 0.1              #Minimum Time each loop of the Com-module takes in s 


    com.adeIDList = [2,3,5,6]           #ADE ID list

    com.adeNameIDallocation = {     # Dictionary allocating ADE-Names to IDs
        1:'-',
        2:'WEMOS_ID_2',
        3:'ESP-3A10F5',
        4:'WEMOS_ID_4',
        5:'ESP-3A0D49',
        6:'ESP-7DCC03',
        9:'ESP-3A05D7'}

    #com.adeIPIDallocation = {       # Dictionary allocating ADE-IPs to IDs
    #    1:'-',
    #    2:'-',
    #    3:'192.168.43.26',
    #    4:'-',
    #    5:'192.168.43.28',
    #    6:'192.168.43.45',
    #    9:'192.168.43.252'}

    
    com.adeIPIDallocation = {       # Dictionary allocating ADE-IPs to IDs
        1:'192.168.1.1',
        2:'192.168.1.2',
        3:'192.168.1.3',
        4:'192.168.1.4',
        5:'192.168.1.5',
        6:'192.168.1.6',
        7:'192.168.1.7',
        8:'192.168.1.8',
        9:'192.168.1.9'}
    # Data config
    global data
    data = structure()

    data.ademoveListLocation =  'data/moveList/Fahrt014ADE.txt'   # Location of Save file Bsp.:['data\movelst\demo_ADE3.txt','data\movelst\save_4ADE_1Punkt_Fahrt2.txt','data\movelst\save_01.txt']

    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++s
    # Camera config
    global camera
    camera = structure()

    camera.available = False    # Set False if no Camera is Available (ex. Testing outside of Lab)
    
    # 'data//experiments//4ADEs//uncorrected//'
    # 'data//experiments//supportStructure//Image//7//'
    # 'data//experiments//supportStructure//Image//6//2//'
    # 'data//experiments//rollingGait//Image//'
    # 'data//experiments//bridge//7ATC//unSplit//'
    # 'data//experiments//loopIntegration//Image//'
    # 'data//experiments//walker//Image//'
    # 'data//experiments//rollingGait//Image//'

    camera.imageFolder = 'data/experiments/rollingGait/Image/'  # Folder where Images are Read from if Camera isnt Available


    camera.ID = 0


    camera.codec = 0x47504A4D   # MJPG
    camera.API = cv2.CAP_DSHOW  # Camera API
    camera.videoFormat = [2592,1994]    # Videoformat [Width,Heigth] in Pixel


    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Measurement config
    global measurement
    measurement = structure()
    measurement.looptime = 0.01        #Minimum Time each loop takes in s


    measurement.PointsPerAde = 3        # Number of Measurement Points per ADE
    measurement.BlueLower = np.array([110, 45, 0])         # Lower limit of Coordinate Marker Color
    measurement.BlueUpper = np.array([125, 255, 100])      # Upper limit of Coordinate Marker Color

            
    # Use TrackingmitTrackbar.py to get HSV Values
    if not camera.available:

        #measurement.markerColorLower = np.array([50, 70, 50])      # Lower limit of Measurement Point Marker Color np.array([56, 30, 30])  
        #measurement.markerColorUpper = np.array([80, 255, 255])    # Upper limit of Measurement Point Marker Color np.array([95, 150, 255])
        #measurement.markerColorLower = np.array([50, 70, 20])      # Lower limit of Measurement Point Marker Color np.array([56, 30, 30])  
        #measurement.markerColorUpper = np.array([90, 255, 255])    # Upper limit of Measurement Point Marker Color np.array([95, 150, 255])
        measurement.markerColorLower = np.array([55, 45, 25])      # rolling Gait
        measurement.markerColorUpper = np.array([80, 255, 255])
        #measurement.markerColorLower = np.array([60, 50, 20])      # loopIntegration
        #measurement.markerColorUpper = np.array([80, 255, 255])
        #measurement.markerColorLower = np.array([55, 70, 45])      # SupportStructure 7
        #measurement.markerColorUpper = np.array([80, 255, 255])
        #measurement.markerColorLower = np.array([60, 80, 50])      # walker
        #measurement.markerColorUpper = np.array([80, 255, 255])
        #measurement.markerColorLower = np.array([60, 60, 20])      # gripper
        #measurement.markerColorUpper = np.array([80, 255, 255])
        #measurement.markerColorLower = np.array([60, 60, 50])      # 4ADEs
        #measurement.markerColorUpper = np.array([80, 255, 255])

    else:
        measurement.markerColorLower = np.array([60, 80, 20])      # Lower limit of Measurement Point Marker Color np.array([56, 30, 30])  
        measurement.markerColorUpper = np.array([80, 190, 130])    # Upper limit of Measurement Point Marker Color np.array([95, 150, 255])

    measurement.markerRadius = 7.5*1e-3 #Marker Radius in mm
    measurement.radiusRange = 3  # inPixels

    measurement.xCord = 1080*1e-3     # x Distance of the Coordinate Markers in mm
    measurement.yCord = 940*1e-3      # y Distance of the Coordinate Markers in mm


    #measurement.xCord = 1080     # x Distance of the Coordinate Markers in mm
    #measurement.yCord = 520      # y Distance of the Coordinate Markers in mm

    measurement.calibrationFolder = 'data/calibrationData/'#'data/calibration3DNear/'#'data/allCalibrationData/'#'data/calibration3DNear/'#

    
    # measurement.ret = np.genfromtxt(measurement.calibrationFolder+'ret.txt',delimiter = ',',dtype = float)  #Calibration Data
    # measurement.mtx = np.genfromtxt(measurement.calibrationFolder+'mtx.txt',delimiter = ',',dtype = float)
    # measurement.dist = np.genfromtxt(measurement.calibrationFolder+'dist.txt',delimiter = ',',dtype = float)

    global gesture 
    gesture = structure()
    gesture.looptime = 0.1                          #Minimum Time each loop takes in s 
    gesture.minimalTimeBetweenMovemendCommands = 1  #Minimum Time between each Movemend Command in s
    gesture.gestureModeCheckTime = 2                #Minimum Time between each Gesture in s
    gesture.cameraID = 2                            #Camera ID of Gesture Camera, Use GetCameraID.py to show ID of each Camera

    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Visualization config
    global visualization
    visualization = structure()
    visualization.looptime = 0.01                    #Minimum Time each loop takes in s 

    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Simulation Config 

    global simulation
    simulation = structure()

    simulation.looptime = 0.1 # in s
    
    simulation.idleTime = 0

    #Options, for visualization
    simulation.graphics = 'graphicsStandard' #'graphicsStandard', 'graphicsSimplifed', 'graphicsSTL'

    #Options, if True RevoluteJoint2D are used for the rectangularBuildingBlock
    simulation.useRevoluteJoint2DTower=True #setFalse Springs ASME2020CI, True = revolute joints
    
    #Options if MoveList contains Connection Information
    simulation.useMoveList = True
    simulation.useInverseKinematic = False
    
    simulation.constrainActorLength = True

    simulation.connectionInfoInMoveList = False
    simulation.debugConnectionInfo = False

    
    
    #Options for Boundary
    simulation.numberOfBoundaryPoints = 20
    simulation.boundaryConnectionInfoInMoveList = False
    simulation.setMeshWithCoordinateConst = True
    
    # if simulation.setMeshWithCoordinateConst:
    #     simulation.numberOfBoundaryPoints = 0 #change here number of boundary 
    
    
    
    # ursprünglich die nächsten 3 falsch
    #Options for six-bar linkages, if all False RevoluteJoint2D are used for the six-bar linkages
    simulation.simplifiedLinksIdealRevoluteJoint2D = True #simplified model using only revoluteJoints at edges
    simulation.massPoints2DIdealRevoluteJoint2D = True
    simulation.simplifiedLinksIdealCartesianSpringDamper = True
    
    simulation.useCompliantJoints=False          #False for ideal
    simulation.numberOfElementsBeam=8
    simulation.numberOfElements=16

    simulation.useCrossSpringPivot = False #use CrossSpringPivot

    simulation.cartesianSpringDamperActive = False #setTrue ASME2020CI
    simulation.connectorRigidBodySpringDamper = False 
    
    simulation.rigidBodySpringDamperNonlinearStiffness = True #use nonlinear compliance matrix then choose circularHinge and crossSpringPivots
    simulation.rigidBodySpringDamperNonlinearStiffnessCircularHinge=False
    simulation.rigidBodySpringDamperNonlinearStiffnessCrossSpringPivot=False
    
    simulation.rigidBodySpringDamperNeuralNetworkCircularHinge=True
    simulation.rigidBodySpringDamperNeuralNetworkCrossSprings=False
    simulation.neuralnetworkJIT=True
    
    #friction force (norm); acts against velocity
    simulation.zeroZoneFriction = 1e-3 #zero-zone for velocity in friction
    simulation.fFriction = 0.07*0 #0.107*0.8 #*0.65 #*0.7



    #Options for Solver
    simulation.SolveDynamic = False #set False to SolveStatic
    simulation.endTime = 10 #used only if interactive marker True
    simulation.nrSteps = int(1e2)



    #Options for output
    simulation.saveSensorValues = True
    simulation.sensorValuesFile = 'output/SensorValues.pickle'
    # sensor_values_path = os.path.join(output_dir, 'SensorValues.pickle')

    simulation.sensorValuesFile2 = 'output/SensorValues2.pickle'
    simulation.sensorValuesTimeFile2 = 'output/SensorValuesTime2.pickle'
    
    simulation.saveImages = False
    simulation.imageFilename = 'output/SimulationImages/Simulation'
    
    
    ### settings for interactive marker
    simulation.towerHight=10
    simulation.towerLength=5
    
    simulation.speedOfInteractiveNode = -0.2
    simulation.loadValueInteractiveMode=1 #in N
    
    simulation.changeActorStiffness = False        #Use UserFunction for SpringDampers/ADE Aktuators
    simulation.stiffValueMin=1e4
    simulation.stiffValueMax=1e6
    
    simulation.LminForStiffness=0.229-0.02
    simulation.LmaxForStiffness=0.329
    
    
    simulation.addLoadOnVertices = False           #Adds Load onto Vertecies of the ADEs
    simulation.addInteractivMarker = False         #Flag if Interactiv Marker is Activated
    simulation.radiusInteractivMarker = 0.03       #Radius of the interacvit Marker in m
    
    
    simulation.rangeFactorSearchNode=5
    simulation.rangeFactorCloseNodes=5
    
    simulation.showRangeFactors=False



    simulation.changeConnectorStiffness = False    #Use UserFunction for SpringDampers/ADE Connectors
    simulation.activateWithKeyPress = False
    simulation.addSpringToInteractivMarker = False              #Add Spring between Interactiv Marker and a Given ADE Marker
    simulation.interactivNodeSpeed = 0.05 #for UserFunctionkeyPress
    simulation.interactivMarkerPosition = [0.229 ,0.229,0]  #Initial Position of the Interactiv Marker
    # change between simulation.graphics by used elements
    if simulation.useCompliantJoints or simulation.useCrossSpringPivot:
        simulation.graphics = 'graphicsSTL'
    else:
        simulation.graphics = 'graphicsStandard'
    
    
    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #Options for Visualization
    simulation.frameList = True
    simulation.showJointAxes=True

    simulation.animation=False
    
    simulation.showLabeling = True #shows nodeNumbers, ID and sideNumbers, simulationTime

    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #Options for Solution Viewer
    simulation.displaySimulation = False
    simulation.solutionViewer = True
    simulation.solutionViewerFile = 'solution/coordinatesSolution.txt'
    simulation.solutionWritePeriod = 0.01

    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Control Config
    global control
    control = structure()

    control.pauseValue = 0
    control.looptime = 0.1

    


    # Output config:
    global output
    output = structure()

    output.debug = True

    output.plot = False
    output.folder = 'output/30_06_21/ADEFahrt1/'
    output.coordinateFile = 'output/coordinates.txt'

    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%m/%d/%Y %H:%M:%S ')
    logging.getLogger('matplotlib.font_manager').disabled = True
=======
﻿#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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


import numpy as np
import matplotlib
import logging
import cv2

class structure():
    pass

def init():
    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Lengths Config 
    global lengths
    lengths = structure()

    lengths.L0 = 229*1e-3     #Unextended Length of ADEs
    lengths.L1=38*1e-3         #Length L1 between P0 (Hinge) and P1 #37.7418 mm
    lengths.L2=29*1e-3         #Length L2 between P1 and P2
    lengths.L3=8*1e-3          #Length L3 between P1 and P3 
    lengths.L4=10*1e-3         #Length L4 between P1 and MarkerPoint towards P0
    lengths.L5=1*1e-3          #Length L5 between P1 and MarkerPoint 90° rotated towards P0 
    lengths.L100=329*1e-3      #Extended Length of ADEs
    
    lengths.weightATC = 0.365  #wight of one ATC in kg
    
    #weight and inertia of RB0 and RB5
    b=40e-3 # width of actuator
    h=60e-3 # length of actuator    
    
    lengths.inertiaActuator=1/12*(b*h**3)
    lengths.weightActuator = lengths.weightATC/6
    
    #weight and inertia of RB1 - RB4
    #C-shape profile B=3.5mm, b=2mm , H=7mm, h=1mm (Ix)
    B=3.5e-3 #m
    b=2e-3 #m
    H=7e-3 #m
    h=1e-3 #m
    
    b3=B-b
    h4=H-2*h
    Ix=(B*H**3-b3*h4**3)/12
    lengths.inertiaBeams=Ix *2   #mm^4 #2 six-bar linkages (top bottom) 
    lengths.weightBeams=1.0e-3 *2 #2 six-bar linkages (top bottom)
    
    
    
    
    lengths.boundaryXdistance = 0*1e-3    #X Distance from Boundary Marker to P0
    lengths.boundaryYdistance = 9.35*1e-3 #Y Distance from Boundary Marker to P0
    
    #spring for prismaticJoint (Actuator); initial stiffness and damping
    lengths.stiffnessPrismaticJoints = 1e5
    lengths.dampingPrismaticJoints = 5e3
    
    #values for springs for connectors between two elements
    lengths.stiffnessConnectors=6e3*1e5
    lengths.dampingConnectors=2e4
    
    lengths.stiffnessBoundary=4e6 #4e4
    lengths.dampingBoundary=3e3  
    
    lengths.maxStroke = 0.101
    lengths.minStroke = -0.001
    
    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Webserver Config
    global webserver

    webserver = structure()
    webserver.pauseBetweenMovementCommands = 10     #Pause Value in s between MovementCommands when running a Movelist from Server 


    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Communication config
    global com
    com = structure()

    com.actorSpeed = 25 #mm/s

    com.timeoutReceive = 20         #Time to wait for ADE Battery Status in s
    com.timeoutSend = 1             #Time to wait for TCPSocket while Sending in s
    com.pauseVal = 0              #Pause Value between Move List rows in s
    com.looptime = 0.1              #Minimum Time each loop of the Com-module takes in s 


    com.adeIDList = [2,3,5,6]           #ADE ID list

    com.adeNameIDallocation = {     # Dictionary allocating ADE-Names to IDs
        1:'-',
        2:'WEMOS_ID_2',
        3:'ESP-3A10F5',
        4:'WEMOS_ID_4',
        5:'ESP-3A0D49',
        6:'ESP-7DCC03',
        9:'ESP-3A05D7'}

    #com.adeIPIDallocation = {       # Dictionary allocating ADE-IPs to IDs
    #    1:'-',
    #    2:'-',
    #    3:'192.168.43.26',
    #    4:'-',
    #    5:'192.168.43.28',
    #    6:'192.168.43.45',
    #    9:'192.168.43.252'}

    
    com.adeIPIDallocation = {       # Dictionary allocating ADE-IPs to IDs
        1:'192.168.1.1',
        2:'192.168.1.2',
        3:'192.168.1.3',
        4:'192.168.1.4',
        5:'192.168.1.5',
        6:'192.168.1.6',
        7:'192.168.1.7',
        8:'192.168.1.8',
        9:'192.168.1.9'}
    # Data config
    global data
    data = structure()

    data.ademoveListLocation =  'data\moveList\Fahrt014ADE.txt'   # Location of Save file Bsp.:['data\movelst\demo_ADE3.txt','data\movelst\save_4ADE_1Punkt_Fahrt2.txt','data\movelst\save_01.txt']

    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++s
    # Camera config
    global camera
    camera = structure()

    camera.available = False    # Set False if no Camera is Available (ex. Testing outside of Lab)
    
    # 'data//experiments//4ADEs//uncorrected//'
    # 'data//experiments//supportStructure//Image//7//'
    # 'data//experiments//supportStructure//Image//6//2//'
    # 'data//experiments//rollingGait//Image//'
    # 'data//experiments//bridge//7ATC//unSplit//'
    # 'data//experiments//loopIntegration//Image//'
    # 'data//experiments//walker//Image//'
    # 'data//experiments//rollingGait//Image//'

    camera.imageFolder = 'data//experiments//rollingGait//Image//'  # Folder where Images are Read from if Camera isnt Available


    camera.ID = 0


    camera.codec = 0x47504A4D   # MJPG
    camera.API = cv2.CAP_DSHOW  # Camera API
    camera.videoFormat = [2592,1994]    # Videoformat [Width,Heigth] in Pixel


    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Measurement config
    global measurement
    measurement = structure()
    measurement.looptime = 0.01        #Minimum Time each loop takes in s


    measurement.PointsPerAde = 3        # Number of Measurement Points per ADE
    measurement.BlueLower = np.array([110, 45, 0])         # Lower limit of Coordinate Marker Color
    measurement.BlueUpper = np.array([125, 255, 100])      # Upper limit of Coordinate Marker Color

            
    # Use TrackingmitTrackbar.py to get HSV Values
    if not camera.available:

        #measurement.markerColorLower = np.array([50, 70, 50])      # Lower limit of Measurement Point Marker Color np.array([56, 30, 30])  
        #measurement.markerColorUpper = np.array([80, 255, 255])    # Upper limit of Measurement Point Marker Color np.array([95, 150, 255])
        #measurement.markerColorLower = np.array([50, 70, 20])      # Lower limit of Measurement Point Marker Color np.array([56, 30, 30])  
        #measurement.markerColorUpper = np.array([90, 255, 255])    # Upper limit of Measurement Point Marker Color np.array([95, 150, 255])
        measurement.markerColorLower = np.array([55, 45, 25])      # rolling Gait
        measurement.markerColorUpper = np.array([80, 255, 255])
        #measurement.markerColorLower = np.array([60, 50, 20])      # loopIntegration
        #measurement.markerColorUpper = np.array([80, 255, 255])
        #measurement.markerColorLower = np.array([55, 70, 45])      # SupportStructure 7
        #measurement.markerColorUpper = np.array([80, 255, 255])
        #measurement.markerColorLower = np.array([60, 80, 50])      # walker
        #measurement.markerColorUpper = np.array([80, 255, 255])
        #measurement.markerColorLower = np.array([60, 60, 20])      # gripper
        #measurement.markerColorUpper = np.array([80, 255, 255])
        #measurement.markerColorLower = np.array([60, 60, 50])      # 4ADEs
        #measurement.markerColorUpper = np.array([80, 255, 255])

    else:
        measurement.markerColorLower = np.array([60, 80, 20])      # Lower limit of Measurement Point Marker Color np.array([56, 30, 30])  
        measurement.markerColorUpper = np.array([80, 190, 130])    # Upper limit of Measurement Point Marker Color np.array([95, 150, 255])

    measurement.markerRadius = 7.5*1e-3 #Marker Radius in mm
    measurement.radiusRange = 3  # inPixels

    measurement.xCord = 1080*1e-3     # x Distance of the Coordinate Markers in mm
    measurement.yCord = 940*1e-3      # y Distance of the Coordinate Markers in mm


    #measurement.xCord = 1080     # x Distance of the Coordinate Markers in mm
    #measurement.yCord = 520      # y Distance of the Coordinate Markers in mm

    measurement.calibrationFolder = 'data/calibrationData/'#'data/calibration3DNear/'#'data/allCalibrationData/'#'data/calibration3DNear/'#

    
    # measurement.ret = np.genfromtxt(measurement.calibrationFolder+'ret.txt',delimiter = ',',dtype = float)  #Calibration Data
    # measurement.mtx = np.genfromtxt(measurement.calibrationFolder+'mtx.txt',delimiter = ',',dtype = float)
    # measurement.dist = np.genfromtxt(measurement.calibrationFolder+'dist.txt',delimiter = ',',dtype = float)

    global gesture 
    gesture = structure()
    gesture.looptime = 0.1                          #Minimum Time each loop takes in s 
    gesture.minimalTimeBetweenMovemendCommands = 1  #Minimum Time between each Movemend Command in s
    gesture.gestureModeCheckTime = 2                #Minimum Time between each Gesture in s
    gesture.cameraID = 2                            #Camera ID of Gesture Camera, Use GetCameraID.py to show ID of each Camera

    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Visualization config
    global visualization
    visualization = structure()
    visualization.looptime = 0.01                    #Minimum Time each loop takes in s 

    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Simulation Config 

    global simulation
    simulation = structure()

    simulation.looptime = 0.1 # in s
    
    simulation.idleTime = 0

    #Options, for visualization
    simulation.graphics = 'graphicsStandard' #'graphicsStandard', 'graphicsSimplifed', 'graphicsSTL'

    #Options, if True RevoluteJoint2D are used for the rectangularBuildingBlock
    simulation.useRevoluteJoint2DTower=True #setFalse Springs ASME2020CI, True = revolute joints
    
    #Options if MoveList contains Connection Information
    simulation.useMoveList = True
    simulation.useInverseKinematic = False
    
    simulation.constrainActorLength = True

    simulation.connectionInfoInMoveList = False
    simulation.debugConnectionInfo = False

    
    
    #Options for Boundary
    simulation.numberOfBoundaryPoints = 20
    simulation.boundaryConnectionInfoInMoveList = False
    simulation.setMeshWithCoordinateConst = True
    
    # if simulation.setMeshWithCoordinateConst:
    #     simulation.numberOfBoundaryPoints = 0 #change here number of boundary 
    
    
    
    
    #Options for six-bar linkages, if all False RevoluteJoint2D are used for the six-bar linkages
    simulation.simplifiedLinksIdealRevoluteJoint2D = False #simplified model using only revoluteJoints at edges
    simulation.massPoints2DIdealRevoluteJoint2D = False
    simulation.simplifiedLinksIdealCartesianSpringDamper = False
    
    simulation.useCompliantJoints=True          #False for ideal
    simulation.numberOfElementsBeam=8
    simulation.numberOfElements=16

    simulation.useCrossSpringPivot = False #use CrossSpringPivot

    simulation.cartesianSpringDamperActive = False #setTrue ASME2020CI
    simulation.connectorRigidBodySpringDamper = False 
    
    simulation.rigidBodySpringDamperNonlinearStiffness = False #use nonlinear compliance matrix then choose circularHinge and crossSpringPivots
    simulation.rigidBodySpringDamperNonlinearStiffnessCircularHinge=False
    simulation.rigidBodySpringDamperNonlinearStiffnessCrossSpringPivot=False
    
    simulation.rigidBodySpringDamperNeuralNetworkCircularHinge=False
    simulation.rigidBodySpringDamperNeuralNetworkCrossSprings=False
    
    
    #friction force (norm); acts against velocity
    simulation.zeroZoneFriction = 1e-3 #zero-zone for velocity in friction
    simulation.fFriction = 0.07*0 #0.107*0.8 #*0.65 #*0.7



    #Options for Solver
    simulation.SolveDynamic = False #set False to SolveStatic
    simulation.endTime = 10 #used only if interactive marker True
    simulation.nrSteps = int(1e3) 



    #Options for output
    simulation.saveSensorValues = True
    simulation.sensorValuesFile = 'output/SensorValues.pickle'
    simulation.sensorValuesTimeFile = 'output/SensorValuesTime.pickle'
    
    simulation.saveImages = False
    simulation.imageFilename = 'output/SimulationImages/Simulation'
    
    
    ### settings for interactive marker
    simulation.towerHight=10
    simulation.towerLength=5
    
    simulation.speedOfInteractiveNode = -0.2
    simulation.loadValueInteractiveMode=1 #in N
    
    simulation.changeActorStiffness = False        #Use UserFunction for SpringDampers/ADE Aktuators
    simulation.stiffValueMin=1e4
    simulation.stiffValueMax=1e6
    
    simulation.LminForStiffness=0.229-0.02
    simulation.LmaxForStiffness=0.329
    
    
    simulation.addLoadOnVertices = False           #Adds Load onto Vertecies of the ADEs
    simulation.addInteractivMarker = False         #Flag if Interactiv Marker is Activated
    simulation.radiusInteractivMarker = 0.03       #Radius of the interacvit Marker in m
    
    
    simulation.rangeFactorSearchNode=5
    simulation.rangeFactorCloseNodes=5
    
    simulation.showRangeFactors=False



    simulation.changeConnectorStiffness = False    #Use UserFunction for SpringDampers/ADE Connectors
    simulation.activateWithKeyPress = False
    simulation.addSpringToInteractivMarker = False              #Add Spring between Interactiv Marker and a Given ADE Marker
    simulation.interactivNodeSpeed = 0.05 #for UserFunctionkeyPress
    simulation.interactivMarkerPosition = [0.229 ,0.229,0]  #Initial Position of the Interactiv Marker
    # change between simulation.graphics by used elements
    if simulation.useCompliantJoints or simulation.useCrossSpringPivot:
        simulation.graphics = 'graphicsSTL'
    else:
        simulation.graphics = 'graphicsStandard'
    
    
    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #Options for Visualization
    simulation.frameList = True
    simulation.showJointAxes=True

    simulation.animation=False
    
    simulation.showLabeling = True #shows nodeNumbers, ID and sideNumbers, simulationTime

    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #Options for Solution Viewer
    simulation.displaySimulation = False
    simulation.solutionViewer = True
    simulation.solutionViewerFile = 'solution/coordinatesSolution.txt'
    simulation.solutionViewerFileIKIN = 'solution/coordinatesSolutionIKIN.txt'
    simulation.solutionWritePeriod = 0.01

    #%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Control Config
    global control
    control = structure()

    control.pauseValue = 0
    control.looptime = 0.1

    


    # Output config:
    global output
    output = structure()

    output.debug = True

    output.plot = False
    output.folder = 'output/30_06_21/ADEFahrt1/'
    output.coordinateFile = 'output\coordinates.txt'

    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%m/%d/%Y %H:%M:%S ')
    logging.getLogger('matplotlib.font_manager').disabled = True
>>>>>>> ebc144b (added angle to gripper and changed filepaths to work platform-independent)
