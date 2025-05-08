#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 13:43:44 2025

@author: fynnheidenreich
"""

import numpy as np

class PTPTrajectory:
    def __init__(self, startLengths, targetLengths, vMax, aMax):
        self.startLengths = np.array(startLengths)
        self.targetLengths = np.array(targetLengths)
        self.vMax = vMax
        self.aMax = aMax
        self.numActuators = len(startLengths)
        self.computeTiming()

    def computeTiming(self):
        # calc length difference, direction and distance
        self.deltas = self.targetLengths - self.startLengths
        self.distances = np.abs(self.deltas)
        self.directions = np.sign(self.deltas)
        
        # initialize time parameters for every actuator
        self.t_accel = np.zeros(self.numActuators)
        self.t_total = np.zeros(self.numActuators)
        
        # true = Dreieck, false = Trapez
        self.triangleProfile = np.zeros(self.numActuators, dtype=bool)

        for i in range(self.numActuators):
            # acceleration time for max velocity
            t_a = self.vMax / self.aMax
            # distance during accerlation
            s_a = 0.5 * self.aMax * t_a**2

            # amax is not reached -> Dreieckprofil
            if self.distances[i] < 2 * s_a:
                self.triangleProfile[i] = True
                self.t_accel[i] = np.sqrt(self.distances[i] / self.aMax)
                self.t_total[i] = 2 * self.t_accel[i]
            else:   # amax is reached -> Trapezprofil
                self.triangleProfile[i] = False
                self.t_accel[i] = t_a
                s_const = self.distances[i] - 2 * s_a
                t_const = s_const / self.vMax
                self.t_total[i] = 2 * t_a + t_const

    # returns the length for all actuators for a specific time stamp
    def evaluate(self, t):
        lengths = np.zeros(self.numActuators)
        for i in range(self.numActuators):
            # Dreieck
            if self.triangleProfile[i]:
                if t < self.t_accel[i]:
                    # acceleration phase
                    s = 0.5 * self.aMax * t**2
                elif t < self.t_total[i]:
                    # deceleration phase
                    t_dec = t - self.t_accel[i]
                    s = (0.5 * self.aMax * self.t_accel[i]**2 +
                         self.aMax * self.t_accel[i] * t_dec -
                         0.5 * self.aMax * t_dec**2)
                else:
                    # target reached
                    s = self.distances[i]
            else:
                s_a = 0.5 * self.aMax * self.t_accel[i]**2
                t_const = self.t_total[i] - 2 * self.t_accel[i]
                if t < self.t_accel[i]:
                    # acceleration phase
                    s = 0.5 * self.aMax * t**2
                elif t < self.t_accel[i] + t_const:
                    # constant velocity phase
                    s = s_a + self.vMax * (t - self.t_accel[i])
                elif t < self.t_total[i]:
                    # deceleration phase
                    t_dec = t - self.t_accel[i] - t_const
                    s = s_a + self.vMax * t_const + self.vMax * t_dec - 0.5 * self.aMax * t_dec**2
                else:
                    # target reached
                    s = self.distances[i]

            lengths[i] = self.startLengths[i] + self.directions[i] * s

        return lengths
    
class syncPTPTrajectory:
    def __init__(self, startLengths, targetLengths, vMax, aMax):
        self.startLengths = np.array(startLengths)
        self.targetLengths = np.array(targetLengths)
        self.vMax = vMax
        self.aMax = aMax
        self.numActuators = len(startLengths)
        self.computeTiming()

    def computeTiming(self):
        # calc length difference, direction and distance
        self.deltas = self.targetLengths - self.startLengths
        self.distances = np.abs(self.deltas)
        self.directions = np.sign(self.deltas)
        
        # find max distance
        self.sMax = np.max(self.distances)
        
        # initialize time parameters
        # acceleration time for max velocity
        self.t_accel = self.vMax / self.aMax
        # distance during accerlation
        self.s_accel = 0.5 * self.aMax * self.t_accel**2
        
        # true = Dreieck, false = Trapez
        self.triangleProfile = np.zeros(self.numActuators, dtype=bool)
        
        # amax is not reached -> Dreieckprofil
        if self.sMax < 2 * self.s_accel:
            self.triangleProfile = True
            self.t_accel = np.sqrt(self.sMax / self.aMax)
            self.t_total_sync = 2 * self.t_accel
        else: # amax is reached -> Trapezprofil
            self.triangleProfile = False
            self.t_accel = self.t_accel
            s_const = (self.sMax - 2 * self.s_accel)
            t_const = s_const / self.vMax
            self.t_total_sync = 2 * self.t_accel + t_const
            self.t_total = self.t_total_sync
            
        self.aMaxVec = np.zeros(self.numActuators)
        self.vMaxVec = np.zeros(self.numActuators)
        
        for i in range(self.numActuators):
            self.aMaxVec[i] = 4 * self.distances[i] / (self.t_total_sync**2)
            self.vMaxVec[i] = self.aMaxVec[i] * (self.t_total_sync / 2)
        

    # returns the length for all actuators for a specific time stamp
    def evaluate(self, t):
        lengths = np.zeros(self.numActuators)
    
        for i in range(self.numActuators):
            a = self.aMaxVec[i]
            v = self.vMaxVec[i]
            s = self.distances[i]
            dir = self.directions[i]
            t_a = v / a  # individuelle Beschleunigungszeit
            t_total = self.t_total_sync
    
            if self.triangleProfile:
                if t < t_a:
                    pos = 0.5 * a * t**2
                elif t < t_total:
                    t_dec = t - t_a
                    pos = 0.5 * a * t_a**2 + a * t_a * t_dec - 0.5 * a * t_dec**2
                else:
                    pos = s
            else:
                s_a = 0.5 * a * t_a**2
                t_const = t_total - 2 * t_a
                if t < t_a:
                    pos = 0.5 * a * t**2
                elif t < t_a + t_const:
                    pos = s_a + v * (t - t_a)
                elif t < t_total:
                    t_dec = t - (t_a + t_const)
                    pos = s_a + v * t_const + v * t_dec - 0.5 * a * t_dec**2
                else:
                    pos = s
    
            lengths[i] = self.startLengths[i] + dir * pos

        return lengths