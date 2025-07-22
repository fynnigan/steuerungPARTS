#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 13:43:44 2025

@author: fynnheidenreich
"""

import numpy as np
from scipy.interpolate import CubicHermiteSpline
import matplotlib.pyplot as plt


class PTPTrajectory:
    def __init__(self, startLengths, targetLengths, vMax, aMax, sync=True):
        self.startLengths = np.array(startLengths)
        self.targetLengths = np.array(targetLengths)
        self.vMax = vMax
        self.aMax = aMax
        self.sync = sync
        self.numActuators = len(startLengths)
        self.computeTiming()

    def computeTiming(self):
        self.deltas = self.targetLengths - self.startLengths
        self.distances = np.abs(self.deltas)
        self.directions = np.sign(self.deltas)

        self.triangleProfile = np.zeros(self.numActuators, dtype=bool)

        if self.sync:
            self.sMax = np.max(self.distances)

            t_a = self.vMax / self.aMax
            s_a = 0.5 * self.aMax * t_a**2

            if self.sMax < 2 * s_a:
                self.triangleProfile[:] = True
                self.t_accel = np.sqrt(self.sMax / self.aMax)
                self.t_total = 2 * self.t_accel
            else:
                self.triangleProfile[:] = False
                self.t_accel = t_a
                s_const = self.sMax - 2 * s_a
                t_const = s_const / self.vMax
                self.t_total = 2 * t_a + t_const

            self.aMaxVec = np.zeros(self.numActuators)
            self.vMaxVec = np.zeros(self.numActuators)

            for i in range(self.numActuators):
                if self.distances[i] == 0 or self.t_total == 0:
                    self.aMaxVec[i] = 0
                    self.vMaxVec[i] = 0
                else:
                    self.aMaxVec[i] = 4 * self.distances[i] / (self.t_total**2)
                    self.vMaxVec[i] = self.aMaxVec[i] * (self.t_total / 2)

        else:  # async
            self.t_accel = np.zeros(self.numActuators)
            self.t_single = np.zeros(self.numActuators)

            for i in range(self.numActuators):
                t_a = self.vMax / self.aMax
                s_a = 0.5 * self.aMax * t_a**2

                if self.distances[i] < 2 * s_a:
                    self.triangleProfile[i] = True
                    self.t_accel[i] = np.sqrt(self.distances[i] / self.aMax)
                    self.t_single[i] = 2 * self.t_accel[i]
                else:
                    self.triangleProfile[i] = False
                    self.t_accel[i] = t_a
                    s_const = self.distances[i] - 2 * s_a
                    t_const = s_const / self.vMax
                    self.t_single[i] = 2 * t_a + t_const
            self.t_total = max(self.t_single)

    def evaluate(self, t):
        lengths = np.zeros(self.numActuators)

        for i in range(self.numActuators):
            if self.distances[i] == 0:
                lengths[i] = self.startLengths[i]
                continue

            if self.sync:
                a = self.aMaxVec[i]
                v = self.vMaxVec[i]
                t_a = v / a if a != 0 else 0
                t_total = self.t_total
            else:
                a = self.aMax
                v = self.vMax
                t_a = self.t_accel[i]
                t_total = self.t_single[i]

            if self.triangleProfile[i]:
                if t < t_a:
                    s = 0.5 * a * t**2
                elif t < t_total:
                    t_dec = t - t_a
                    s = 0.5 * a * t_a**2 + a * t_a * t_dec - 0.5 * a * t_dec**2
                else:
                    s = self.distances[i]
            else:
                s_a = 0.5 * a * t_a**2
                t_const = t_total - 2 * t_a
                if t < t_a:
                    s = 0.5 * a * t**2
                elif t < t_a + t_const:
                    s = s_a + v * (t - t_a)
                elif t < t_total:
                    t_dec = t - (t_a + t_const)
                    s = s_a + v * t_const + v * t_dec - 0.5 * a * t_dec**2
                else:
                    s = self.distances[i]

            lengths[i] = self.startLengths[i] + self.directions[i] * s

        return lengths
    
def relativeToAbsolutePoints(relativePoints):
    absolutePoints = [relativePoints[0]]  # Startpunkt als erster absoluter Punkt
    for i in range(1, len(relativePoints)):
        newPoint = absolutePoints[-1] + relativePoints[i]
        absolutePoints.append(newPoint)
    return np.array(absolutePoints)
    
def plotTrajAufgabenraum(spline_x, spline_y, points=None):
    t_vals = np.linspace(0, 1, 200)
    x_vals = spline_x(t_vals)
    y_vals = spline_y(t_vals)

    plt.figure()
    plt.plot(x_vals, y_vals, label='Spline-Trajektorie')
    if points is not None:
        plt.plot(points[:, 0], points[:, 1], 'o', label='Stützpunkte')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.title("Trajektorie im Aufgabenraum")
    plt.show()


def targetToNodeListCoords(pos, angle, TCPOffsets):

    # Rotationsmatrix
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    
    # Rotierte Zielkoordinaten berechnen relativ zu hauptTCP
    rotatedTargets = [pos + R @ offset for offset in TCPOffsets]

    # neue Koordinaten relativ zu ihrer vorherigen Position
    relativeTargetOffsets = np.array(rotatedTargets) - np.array(TCPOffsets)

    # alle zielpunkte anfuegen
    target = [pos[0], pos[1]]
    for i in relativeTargetOffsets:
        target.append(i[0])
        target.append(i[1])
        
    return target



class CPTrajectory:
    def __init__(self, poseVec, resolution, ikObj, ikineNodeList, context, refLength):
        self.lengths = []
        self.times = np.linspace(0, 1, resolution)
        
        # Positionen und Richtungsvektoren extrahieren
        points = np.array([pos for pos, angle in poseVec])
        angles_rad = np.array([angle for pos, angle in poseVec])
        
        # Tangenten als Einheitsvektoren
        tangents = np.column_stack((np.cos(angles_rad), np.sin(angles_rad)))
                
        # Hermite-spline in die punkte legen
        t_stuetz = np.linspace(0, 1, len(poseVec))
        spline_x = CubicHermiteSpline(t_stuetz, points[:, 0], tangents[:, 0])
        spline_y = CubicHermiteSpline(t_stuetz, points[:, 1], tangents[:, 1])
        spline_angle = CubicHermiteSpline(t_stuetz, angles_rad, np.zeros_like(angles_rad))
        
        absPoints = relativeToAbsolutePoints(points)
        plotTrajAufgabenraum(spline_x, spline_y, points=absPoints)
        
        # spline an resolution Punkten interpolieren und IK berechnen
        for t in self.times:
            pos = [spline_x(t), spline_y(t)]
            angle = spline_angle(t)

            # Zielkoordinaten für alle TCPs berechnen
            posVec = targetToNodeListCoords(pos, angle, context['TCPOffsets'])

            # IK berechnen
            ikObj.InverseKinematicsRelative(None, np.array(posVec), ikineNodeList)
            actuatorLengths = np.array(ikObj.GetActuatorLength())
            self.lengths.append(np.array((actuatorLengths - refLength) * 10000))

        self.lengths = np.array(self.lengths)

            
        
    def evaluate(self, t):
        # Zeit clamping
        t_clamped = np.clip(t, 0, 1)
        
        # lineare Interpolation für gegebene Zeit t (zwischen zwei IK-Zuständen)
        idx_float = t_clamped * (len(self.times) - 1)
        idx_low = int(np.floor(idx_float))
        idx_high = min(idx_low + 1, len(self.times) - 1)
        alpha = idx_float - idx_low

        return (1 - alpha) * self.lengths[idx_low] + alpha * self.lengths[idx_high]
        
        
        
        
    
    
    