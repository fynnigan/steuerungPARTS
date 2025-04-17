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