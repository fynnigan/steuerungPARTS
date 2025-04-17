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
    

from functions import webserver_functions as wf
from functions import data_functions as df

import numpy as np
import logging
import config
import socket
import time
import json

from copy import deepcopy

class ADE:
    """
    Class for ADEs which includes the ADE ID, ADE IP and the Actor and Connector settings, as well as the ADE-Parameters
    also contains Functions to communicate with the ade
    """
    S1 = 0
    S2 = 0
    S3 = 0
    M1 = 0
    M2 = 0
    M3 = 0
    battery = 0
    runADE = False

    def __init__(self, ID,connect=True):
        """�
        initialize ADE, which creates an an TCPIP Socket and also connects to the ADE and sends the ADE Parameters

        :param ID: Identification Number od the ADE
        :type ID: int
        """
        self.ID = ID
        self.tcpsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcpsocket.settimeout(config.com.timeoutSend) #Timeout before sending/Connection is aborted
        #self.name = config.com.adeNameIDallocation[self.ID]
        self.IP = config.com.adeIPIDallocation[self.ID]#self.GetIPfromHostName()

        self.ReadParameter()
        if connect:
            self.Connect()
            self.SendParameter()

    def ReadParameter(self):
        """     
        Read Parameter File
        """     
        try:
            self.parameter = np.genfromtxt('data\parameters\ADE_'+str(self.ID)+'.txt',delimiter = ',')
            logging.info(f' ADE {self.ID:2d}: Parameters found')
        except:
            self.parameter = np.array([[1000.0,1000.0,1000.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[1.0,1.0,1.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0],[1.0,1.0,1.0],[0.0,0.0,0.0]])
            logging.info(f' ADE {self.ID:2d}: Parameters not found')

    def SendParameter(self):
        """
        Send Parameters to ade
        """
        try:
            for parameterNumber in range(3):
                data = 'p {} {:f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f} '.format((parameterNumber+1),self.parameter[0,parameterNumber],self.parameter[1,parameterNumber],self.parameter[2,parameterNumber],self.parameter[3,parameterNumber],self.parameter[4,parameterNumber],self.parameter[5,parameterNumber],self.parameter[6,parameterNumber],self.parameter[7,parameterNumber],self.parameter[8,parameterNumber])
                self.tcpsocket.send(data.encode()) # encode Data and send
        except:
            logging.warning(f' ADE {self.ID:2d}: Sending Failed. Try Reconecting')


    def Connect(self):
        """
        connect to ade
        """
        try:
            if self.IP == 'NoIPfound':
                raise ValueError
            self.tcpsocket.connect((self.IP,50000))
            logging.info(f' ADE {self.ID:2d}: Connection Successful')
        except:
            logging.warning(f' ADE {self.ID:2d}: Connection Failed')

    def Disconnect(self):
        """
        Disconnect ade
        """
        try:
            self.tcpsocket.shutdown()
            self.tcpsocket.close()
            logging.info(f' ADE {self.ID:2d}: Disconnected')
        except:
            logging.warning(f' ADE {self.ID:2d}: Was not Connected')


    def GoTo(self):
        """
        send actor lengths and connector commands to ade
        """
        try:
            data = 'g '+str(self.S1)+' '+str(self.S2)+' '+str(self.S3)+' '+str(self.M1)+' '+str(self.M2)+' '+str(self.M3)+' '
            self.tcpsocket.send(data.encode()) # encode Data and send
        except:
            logging.warning(f' ADE {self.ID:2d}: Sending Failed. Try Reconecting')

    def ReadBattery(self):
        """
        read the Battery-Voltage and write it into the Battery parameter of the ADE
        """
        try:
            totalData = ''
            timeout = time.time() + config.com.timeoutReceive #set timeout value
            while True:
                recvdata = self.tcpsocket.recv(1)       # recieve Data
                totalData += recvdata.decode("utf-8")   # decode Data

                if totalData[-1:] == '\n':              # look for Enter Symbol, which indicates End of Data
                    break
                elif time.time() > timeout:             # check for Timeout
                    raise TimeoutError
            self.battery = float(totalData[:-1])        # write Battery status


        except:
            logging.info(f' ADE {self.ID:2d}: No Packages Received')

    def GetIPfromHostName(self):
        try:
            IPAddress = socket.gethostbyname(self.name)
            return IPAddress
        except:
            logging.warning(' ADE not in Network')
            return 'NoIPfound'

def Run(adeMoveList,adeList,line = 1):
    """
    Run the ADE Movelist

    :param adeMoveList: List containing the ADE Movements
    :type adeMoveList: np.array

    :param adeList: List containing the ADE's
    :type adeList: list of ADEs
    """
    for i in range(adeMoveList.data.shape[0]):
        logging.info(f'________________________________________')
        logging.info(f'Zeile: {line}')
        logging.info(f'Befehle: ')
        adeMoveList.FillADE(i,adeList)    #Loads Current MoveList Line into the ADEs
        for ade in adeList:
            if not ade.runADE:
                continue
            ade.GoTo()      #sent movement command to ades
            logging.info(time.strftime("%H:%M:%S", time.localtime())+ ': '+f'ADE {ade.ID:^5d} Aktuatoren: {ade.S1:5d}{ade.S2:5d}{ade.S3:^10d} Magneten: {ade.M1:5d}{ade.M2:5d}{ade.M3:5d}')
        logging.info(f'Batteriestatus:\n')
        for ade in adeList:
            if not ade.runADE:
                continue
            ade.ReadBattery()   # read battery status of ades
            logging.info(f' ADE {ade.ID:2d}: {ade.battery :^8}')
        time.sleep(CalcPause(adeMoveList,i)) # wait for ades to move
        
def CalcPause(adeMoveList,i,minPauseVal=0):
    """
    Calculates the Pause to ensure the ADEs had time to move

    :param adeMoveList: List containing the ADE Movements
    :type adeMoveList: np.array
    
    :param i: line Index of MoveLst
    :type i: int

    """
    if i >= 1:
        # pause = np.max(np.abs(adeMoveList.data[i,:]-adeMoveList.data[i-1,:])/(config.com.actorSpeed*10))+config.com.pauseVal
        pause = np.max(np.abs(adeMoveList.data[i,:]-adeMoveList.data[i-1,:])/(config.com.actorSpeed*10))+config.com.pauseVal
    else:
        pause = config.com.pauseVal
    if pause <= minPauseVal:
        pause = minPauseVal
    return pause
      
def FillADE(adeList, MoveCmds):
    """
    fill the Actor and Connector settings of the ades defined in adeList
    """
    for ade in adeList:
        for n in range(int(len(MoveCmds.data[1,:])/7-1)):
            index = 1+n*7

            if ade.ID == MoveCmds.data[index]:
                ade.S1 = MoveCmds.data[index+1]
                ade.S2 = MoveCmds.data[index+2]
                ade.S3 = MoveCmds.data[index+3]
                ade.M1 = MoveCmds.data[index+4]
                ade.M2 = MoveCmds.data[index+5]
                ade.M3 = MoveCmds.data[index+6]

if __name__ == "__main__":
    ##+++++++++++++++++++++++++++++++++++++++
    ## Initialize Module
    config.init()
    moveCmds = df.adeMoveList()
    newmoveCmds = df.adeMoveList()
    logging.info('Starting Com Module')
    client = wf.WebServerClient()
    adeList = []
    adeIDList = []
    while True:
        try:

            ##+++++++++++++++++++++++++++++++++++++++
            ## Checking if to end the Programm
            if json.loads(client.Get('EndProgram')):
                os._exit(0)

            ##+++++++++++++++++++++++++++++++++++++++
            ## Check the ADE ID List and add them to the ID List if not allready added
            newadeIDList = json.loads(client.Get('adeIDList'))
            if adeIDList != newadeIDList:
                ##+++++++++++++++++++++++++++++++++++++++
                ## Connect New ADEs
                for ID in newadeIDList:
                    if ID not in adeIDList:
                        adeIDList.append(ID)
                        adeList.append(ADE(ID))
            ##+++++++++++++++++++++++++++++++++++++++
            ## Disconnect Old ADEs
                for ade in reversed(adeList):
                    if ade.ID not in newadeIDList:
                        adeIDList.remove(ade.ID)
                        ade.Disconnect()
                        adeList.remove(ade)

            ##+++++++++++++++++++++++++++++++++++++++
            ## Check if to Run ADEs
            runADE = json.loads(client.Get('runADE'))
            for ID in adeIDList:
                if str(ID) in runADE.keys():
                    index = adeIDList.index(ID)
                    adeList[index].runADE = runADE[str(ID)]

            if any(runADE.values()):
                ##+++++++++++++++++++++++++++++++++++++++
                ## Check if MovementCommands Changed
                newmoveCmds.FromJson(client.Get('MovementCommands'))
                line = json.loads(client.Get('Lines'))
                if newmoveCmds.data.size != 0 :
                    if not (moveCmds.data.sum() == newmoveCmds.data.sum()):
                        ##+++++++++++++++++++++++++++++++++++++++
                        ## Send MovementCommands to ADEs
                        moveCmds = deepcopy(newmoveCmds)
                        Run(moveCmds,adeList,line)
        except Exception as ExeptionMessage:
            logging.warning(ExeptionMessage)
        time.sleep(config.com.looptime-time.time()%config.com.looptime)
