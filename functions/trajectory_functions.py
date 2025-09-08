#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 13:43:44 2025

@author: fynnheidenreich
"""

import numpy as np
from scipy.interpolate import splprep, splev, interp1d
import matplotlib
matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["STIX Two Text"]
})
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

def absoluteToRelativePoints(absolutePoints):
    relativePoints = [absolutePoints[0]]  # erster Punkt als Startpunkt
    for i in range(1, len(absolutePoints)):
        rel = absolutePoints[i] - absolutePoints[i - 1]
        relativePoints.append(rel)
    return np.array(relativePoints)


    
def plotTrajAufgabenraum(xy_array, points=None):
    plt.figure()
    plt.plot(xy_array[:, 0], xy_array[:, 1], label='Trajektorie')
    if points is not None:
        # points = absoluteToRelativePoints(points)
        plt.plot(points[:, 0], points[:, 1], 'o', label='Stützpunkte')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    

def rotatePointsAroundTCP(offsets, angle):
    """
    Rotiert eine Liste von 2D-Offsets um den TCP (also um den Ursprung).

    Parameters:
    - offsets: Liste von 2D-Vektoren (z. B. relative Punkte zum TCP)
    - angle: Winkel in Radiant (im mathematisch positiven Sinn)

    Returns:
    - Liste von rotierten 2D-Vektoren
    """
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    
    return [R @ offset for offset in offsets]


def targetToNodeListCoords(pos, angle, relOffsetsToTCP, previousRelOffsets=None):
    """
    Berechnet die Zielkoordinaten für die Node-Liste relativ zu ihrer vorherigen Position.

    Parameters:
    - pos: aktuelle Verschiebung des TCP
    - angle: aktuelle Rotation (im Bogenmaß)
    - relOffsetsToTCP: ursprüngliche relative Positionen der Punkte zum TCP
    - previousRelOffsets: relative Positionen der Punkte aus dem vorherigen Schritt (optional)

    Returns:
    - target: Liste mit [x0, y0, dx1, dy1, dx2, dy2, ...]
    - rotatedRelToTCP: aktuelle (gedrehte) Positionen, zur Weitergabe für den nächsten Schritt
    """
    # Punkte um den TCP drehen
    rotatedRelToTCP = rotatePointsAroundTCP(relOffsetsToTCP, angle)

    # Differenz zum vorherigen Schritt oder zur Grundkonfiguration
    if previousRelOffsets is not None:
        deltaOffsets = [rot - prev for rot, prev in zip(rotatedRelToTCP, previousRelOffsets)]
    else:
        deltaOffsets = [rot - orig for rot, orig in zip(rotatedRelToTCP, relOffsetsToTCP)]

    # Zielpunkte = TCP-Position + relative Änderung
    relativeTargetOffsets = [pos + delta for delta in deltaOffsets]

    # Zielstruktur erzeugen
    target = [pos[0], pos[1]]
    for p in relativeTargetOffsets:
        target.extend(p)  # fügt x, y direkt an

    return target, rotatedRelToTCP



def corner_smoothed_bezier_curve(points, radius=0.5, n_per_corner=20, degree=3):
    def unit(v):
        norm = np.linalg.norm(v)
        return v / norm if norm != 0 else v
    
    if len(points) == 2:
        # Bei nur 2 Punkten: einfache Gerade zurückgeben
        return np.linspace(points[0], points[1], n_per_corner)

    segments = []
    points = np.asarray(points)
    n = len(points)
    prev_out = None
    for i in range(1, n-1):
        p0, p1, p2 = points[i-1], points[i], points[i+1]
        v1 = unit(p0 - p1)
        v2 = unit(p2 - p1)
        d = min(radius, np.linalg.norm(p1 - p0) / 2, np.linalg.norm(p1 - p2) / 2)
        p1_in = p1 + v1 * d
        p1_out = p1 + v2 * d

        t = np.linspace(0, 1, n_per_corner)
        if degree == 2:
            cp = p1
            bezier = ((1 - t)**2)[:, None] * p1_in + 2 * ((1 - t)*t)[:, None] * cp + (t**2)[:, None] * p1_out
        elif degree == 3:
            cp1 = p1_in + (p1 - p1_in) * 0.5
            cp2 = p1_out + (p1 - p1_out) * 0.5
            bezier = ((1 - t)**3)[:, None] * p1_in + 3 * ((1 - t)**2 * t)[:, None] * cp1 + 3 * ((1 - t) * t**2)[:, None] * cp2 + (t**3)[:, None] * p1_out
        elif degree == 5:
            cp1 = p1_in + 0.2 * (p1 - p1_in)
            cp2 = p1_in + 0.5 * (p1 - p1_in)
            cp3 = p1_out + 0.5 * (p1 - p1_out)
            cp4 = p1_out + 0.2 * (p1 - p1_out)
            bezier = ((1 - t)**5)[:, None] * p1_in + 5*((1 - t)**4 * t)[:, None] * cp1 + 10*((1 - t)**3 * t**2)[:, None] * cp2 + 10*((1 - t)**2 * t**3)[:, None] * cp3 + 5*((1 - t) * t**4)[:, None] * cp4 + (t**5)[:, None] * p1_out
        else:
            raise ValueError("Grad nicht unterstützt")

        if i == 1:
            segments.append(np.linspace(points[0], p1_in, n_per_corner))
        else:
            segments.append(np.linspace(prev_out, p1_in, n_per_corner))
        segments.append(bezier)
        prev_out = p1_out

    segments.append(np.linspace(prev_out, points[-1], n_per_corner))
    return np.vstack(segments)



class CPTrajectory:
    def __init__(self, poseVec, resolution, ikObj, ikineNodeList, TCPOffsets, refLength, method="spline"): 
        self.lengths = []
        self.times = np.linspace(0, 1, resolution)
        self.t_total = resolution
        points = np.array([pos for pos, angle in poseVec])
        angles_rad = np.array([angle for pos, angle in poseVec])
        absPoints = relativeToAbsolutePoints(points)

        if method == "linear":
            t_vals = np.linspace(0, 1, resolution)
            segments = [
                np.linspace(absPoints[i], absPoints[i + 1], resolution // (len(absPoints) - 1), endpoint=False)
                for i in range(len(absPoints) - 1)
            ]
            traj = np.vstack(segments + [absPoints[-1][None]])
            
            # Winkelberechnung
            t_stuetz = np.linspace(0, 1, len(angles_rad))
            angle_spline = interp1d(t_stuetz, angles_rad, kind='linear')
            angles = angle_spline(t_vals)
            
        elif method == "spline":
            tck, _ = splprep(absPoints.T, s=1)
            t_vals = np.linspace(0, 1, resolution)
            traj = np.array(splev(t_vals, tck)).T
        
            # Winkelberechnung
            t_stuetz = np.linspace(0, 1, len(angles_rad))
            angle_spline = interp1d(t_stuetz, angles_rad, kind='linear')
            angles = angle_spline(t_vals)
            
            # angles = np.zeros(len(t_vals))
        elif method in ("bezier2", "bezier3", "bezier5"):
            t_vals = np.linspace(0, 1, resolution)
            degree = {"bezier2": 2, "bezier3": 3, "bezier5": 5}[method]
            traj_full = corner_smoothed_bezier_curve(absPoints, radius=0.8, degree=degree)
            indices = np.linspace(0, len(traj_full)-1, resolution).astype(int)
            traj = traj_full[indices]
            
            # Winkelberechnung
            t_stuetz = np.linspace(0, 1, len(angles_rad))
            angle_spline = interp1d(t_stuetz, angles_rad, kind='linear')
            angles = angle_spline(t_vals)
        else:
            raise ValueError(f"Unbekannte Methode: {method}")

        plotTrajAufgabenraum(traj, points=absPoints)

        traj_diffs = np.vstack((traj[0], np.diff(traj, axis=0)))
        angle_diffs = np.diff(angles, axis=0)
        angle_diffs = np.insert(angle_diffs, 0, angles[0])
        
        previousRelOffsets = TCPOffsets  # Initialisierung mit der Grundkonfiguration
        
        cumulativeAngle = 0
        
        for i in range(resolution):
            pos = traj_diffs[i]
            pos = absoluteToRelativePoints([pos])[0]
            angle = angle_diffs[i]
            cumulativeAngle += angle
        
            # posVec, prev = targetToNodeListCoords(pos, angle, TCPOffsets, previousRelOffsets)
            posVec, previousRelOffsets = targetToNodeListCoords(pos, cumulativeAngle, TCPOffsets, previousRelOffsets)
            
            ikObj.InverseKinematicsRelative(None, np.array(posVec))
            actuatorLengths = np.array(ikObj.GetActuatorLength())
            self.lengths.append(actuatorLengths)


        self.lengths = np.array(self.lengths)

    def evaluate(self, t):
        if t >= 1.0:
            return self.lengths[-1]
        
        t_clamped = np.clip(t, 0, 1)
        idx_float = t_clamped * (len(self.times) - 1)
        idx_low = int(np.floor(idx_float))
        idx_high = min(idx_low + 1, len(self.times) - 1)
        alpha = idx_float - idx_low
        return (1 - alpha) * self.lengths[idx_low] + alpha * self.lengths[idx_high]

        
        
        
    
    
    