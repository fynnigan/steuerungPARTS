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
    """
    Class for generating point-to-point trajectories for multiple actuators.
    """
    def __init__(self, startLengths, targetLengths, vMax, aMax, sync=True):
        """
        init function for point-to-point trajectory generation
        Parameters:
        - startLengths: array of starting actuator lengths
        - targetLengths: array of target actuator lengths
        - vMax: maximum velocity (scalar)
        - aMax: maximum acceleration (scalar)
        - sync: boolean indicating if the movement should be synchronized across all actuators 
        """
        self.startLengths = np.array(startLengths)
        self.targetLengths = np.array(targetLengths)
        self.vMax = vMax
        self.aMax = aMax
        self.sync = sync
        self.numActuators = len(startLengths)
        self.computeTiming()

    def computeTiming(self):
        """
        Computes the timing parameters for the trajectory based on the start and target lengths,
        """
        # Calculate deltas, distances, and directions
        self.deltas = self.targetLengths - self.startLengths
        self.distances = np.abs(self.deltas)
        self.directions = np.sign(self.deltas)

        self.triangleProfile = np.zeros(self.numActuators, dtype=bool)

        # Compute timing parameters
        if self.sync:
            # save max distance
            self.sMax = np.max(self.distances)

            # compute timing for max distance
            t_a = self.vMax / self.aMax
            s_a = 0.5 * self.aMax * t_a**2

            # check if triangle profile is needed and compute acceleration time and total time
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

            # compute individual aMax and vMax for each actuator
            for i in range(self.numActuators):
                if self.distances[i] == 0 or self.t_total == 0:
                    self.aMaxVec[i] = 0
                    self.vMaxVec[i] = 0
                else:
                    self.aMaxVec[i] = 4 * self.distances[i] / (self.t_total**2)
                    self.vMaxVec[i] = self.aMaxVec[i] * (self.t_total / 2)

        else:  # asynchronous
            self.t_accel = np.zeros(self.numActuators)
            self.t_single = np.zeros(self.numActuators)

            # compute timing for each actuator based on aMax and vMax
            for i in range(self.numActuators):
                # acceleration time and distance
                t_a = self.vMax / self.aMax
                s_a = 0.5 * self.aMax * t_a**2

                # check if triangle profile is needed and compute acceleration time and total time
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
            # total time is the maximum of individual times
            self.t_total = max(self.t_single)

    def evaluate(self, t):
        """
        Evaluates the trajectory at time t and returns the actuator lengths.
        Parameters:
        - t: time at which to evaluate the trajectory
        Returns:
        - lengths: array of actuator lengths at time t
        """
        lengths = np.zeros(self.numActuators)

        # compute lengths for each actuator
        for i in range(self.numActuators):
            # handle case where distance is zero
            if self.distances[i] == 0:
                lengths[i] = self.startLengths[i]
                continue
            # determine parameters based on sync or async
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

            # compute position based on profile type
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

            # update length based on direction
            lengths[i] = self.startLengths[i] + self.directions[i] * s

        return lengths
    
    
def relativeToAbsolutePoints(relativePoints):
    """
    transforms a list of relative points to absolute points.
    """
    absolutePoints = [relativePoints[0]]  # Startpunkt als erster absoluter Punkt
    for i in range(1, len(relativePoints)):
        newPoint = absolutePoints[-1] + relativePoints[i]
        absolutePoints.append(newPoint)
    return np.array(absolutePoints)


def absoluteToRelativePoints(absolutePoints):
    """
    transforms a list of absolute points to relative points.
    """
    relativePoints = [absolutePoints[0]]  # erster Punkt als Startpunkt
    for i in range(1, len(absolutePoints)):
        rel = absolutePoints[i] - absolutePoints[i - 1]
        relativePoints.append(rel)
    return np.array(relativePoints)


    
def plotTrajAufgabenraum(xy_array, points=None):
    """
    Plots a 2D trajectory in the task space.
    """
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
    rotates a list of 2D points around the TCP by a given angle.
    Parameters:
    - offsets: list of 2D-Vectors 
    - angle: angle in radians
    Returns:
    - rotatedOffsets: list of rotated 2D-Vectors
    """
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    
    return [R @ offset for offset in offsets]


def targetToNodeListCoords(pos, angle, relOffsetsToTCP, previousRelOffsets=None):
    """
    Calculates the target coordinates for the node list relative to their previous position.
    Parameters:
    - pos: current displacement of the TCP
    - angle: current rotation (in radians)
    - relOffsetsToTCP: original relative positions of the points to the TCP
    - previousRelOffsets: relative positions of the points from the previous step (optional)
    Returns:
    - target: list with [x0, y0, dx1, dy1, dx2, dy2, ...]
    - rotatedRelToTCP: current (rotated) positions, to be passed on for the next step
    """
    # rotate relative offsets around TCP
    rotatedRelToTCP = rotatePointsAroundTCP(relOffsetsToTCP, angle)

    # deltas = currentRel - originalRel
    if previousRelOffsets is not None:
        deltaOffsets = [rot - prev for rot, prev in zip(rotatedRelToTCP, previousRelOffsets)]
    else:
        deltaOffsets = [rot - orig for rot, orig in zip(rotatedRelToTCP, relOffsetsToTCP)]

    # target position of nodes = pos of TCP + delta
    relativeTargetOffsets = [pos + delta for delta in deltaOffsets]

    # create target list
    target = [pos[0], pos[1]]
    for p in relativeTargetOffsets:
        target.extend(p) # add dx, dy for each node

    return target, rotatedRelToTCP



def corner_smoothed_bezier_curve(points, radius=0.5, n_per_corner=20, degree=3):
    """
    Generates a corner-smoothed Bezier curve through a series of points.
    Parameters:
    - points: list of 2D points (numpy array or list of lists)
    - radius: radius for corner smoothing
    - n_per_corner: number of points to generate per corner
    - degree: degree of the Bezier curve (2, 3, or 5
    Returns:
    - curve: numpy array of points along the smoothed Bezier curve
    """
    def unit(v):
        norm = np.linalg.norm(v)
        return v / norm if norm != 0 else v
    
    if len(points) == 2:
        # Only two points, return straight line
        return np.linspace(points[0], points[1], n_per_corner)

    segments = []
    points = np.asarray(points)
    n = len(points)
    prev_out = None
    # Iterate through points, creating Bezier curves at corners
    for i in range(1, n-1):
        p0, p1, p2 = points[i-1], points[i], points[i+1]
        v1 = unit(p0 - p1)
        v2 = unit(p2 - p1)
        d = min(radius, np.linalg.norm(p1 - p0) / 2, np.linalg.norm(p1 - p2) / 2)
        p1_in = p1 + v1 * d
        p1_out = p1 + v2 * d

        t = np.linspace(0, 1, n_per_corner)
        # Bezier curve calculation
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
            raise ValueError("Degree must be 2, 3, or 5.")
        # Add line segment from previous point to p1_in
        if i == 1:
            segments.append(np.linspace(points[0], p1_in, n_per_corner))
        else:
            segments.append(np.linspace(prev_out, p1_in, n_per_corner))
        segments.append(bezier)
        prev_out = p1_out

    segments.append(np.linspace(prev_out, points[-1], n_per_corner))
    return np.vstack(segments)



class CPTrajectory:
    """
    Class for generating continuous point-to-point trajectories in task space and converting them to actuator lengths.
    """
    def __init__(self, poseVec, resolution, ikObj, ikineNodeList, TCPOffsets, refLength, method="spline"): 
        """
        init function for continuous point-to-point trajectory generation
        Parameters:
        - poseVec: list of tuples [(pos1, angle1), (pos2, angle2), ...] where pos is a 2D vector and angle is in radians
        - resolution: number of points in the trajectory
        - ikObj: inverse kinematics object with method InverseKinematicsRelative and GetActuatorLength
        - ikineNodeList: list of node indices for inverse kinematics (not used in this function)
        - TCPOffsets: list of 2D vectors representing the offsets of nodes relative to the TCP
        - refLength: reference length for the actuators (not used in this function)
        - method: interpolation method ("linear", "spline", "bezier2", "bezier3", "bezier5")
        """
        self.lengths = []
        self.times = np.linspace(0, 1, resolution)
        self.t_total = resolution
        points = np.array([pos for pos, angle in poseVec])
        angles_rad = np.array([angle for pos, angle in poseVec])
        absPoints = relativeToAbsolutePoints(points)

        if method == "linear":
            # lienar interpolation
            t_vals = np.linspace(0, 1, resolution)
            segments = [
                np.linspace(absPoints[i], absPoints[i + 1], resolution // (len(absPoints) - 1), endpoint=False)
                for i in range(len(absPoints) - 1)
            ]
            traj = np.vstack(segments + [absPoints[-1][None]])
            
            # angle calculation
            t_stuetz = np.linspace(0, 1, len(angles_rad))
            angle_spline = interp1d(t_stuetz, angles_rad, kind='linear')
            angles = angle_spline(t_vals)
            
        elif method == "spline":
            # spline interpolation
            tck, _ = splprep(absPoints.T, s=1)
            t_vals = np.linspace(0, 1, resolution)
            traj = np.array(splev(t_vals, tck)).T
        
            # angle calculation
            t_stuetz = np.linspace(0, 1, len(angles_rad))
            angle_spline = interp1d(t_stuetz, angles_rad, kind='linear')
            angles = angle_spline(t_vals)
            
        elif method in ("bezier2", "bezier3", "bezier5"):
            # Bezier curve with corner smoothing
            t_vals = np.linspace(0, 1, resolution)
            # Determine degree based on method
            degree = {"bezier2": 2, "bezier3": 3, "bezier5": 5}[method]
            # Generate Bezier curve
            traj_full = corner_smoothed_bezier_curve(absPoints, radius=0.8, degree=degree)
            # Resample to desired resolution
            indices = np.linspace(0, len(traj_full)-1, resolution).astype(int)
            traj = traj_full[indices]
            
            # angle calculation
            t_stuetz = np.linspace(0, 1, len(angles_rad))
            angle_spline = interp1d(t_stuetz, angles_rad, kind='linear')
            angles = angle_spline(t_vals)
        else:
            raise ValueError(f"Unbekannte Methode: {method}")

        # plotTrajAufgabenraum(traj, points=absPoints)

        traj_diffs = np.vstack((traj[0], np.diff(traj, axis=0)))
        angle_diffs = np.diff(angles, axis=0)
        angle_diffs = np.insert(angle_diffs, 0, angles[0])
        
        # initial previous offsets
        previousRelOffsets = TCPOffsets  
    
        cumulativeAngle = 0

        # compute actuator lengths for each point in the trajectory    
        for i in range(resolution):
            pos = traj_diffs[i]
            pos = absoluteToRelativePoints([pos])[0]
            angle = angle_diffs[i]
            cumulativeAngle += angle
        
            # get positions for all MCPs
            posVec, previousRelOffsets = targetToNodeListCoords(pos, cumulativeAngle, TCPOffsets, previousRelOffsets)
            
            # compute inverse kinematics
            ikObj.InverseKinematicsRelative(None, np.array(posVec))
            actuatorLengths = np.array(ikObj.GetActuatorLength())
            self.lengths.append(actuatorLengths)

        self.lengths = np.array(self.lengths)

    def evaluate(self, t):
        """
        Evaluates the trajectory at time t and returns the actuator lengths.
        Parameters:
        - t: time at which to evaluate the trajectory
        Returns:
        - lengths: array of actuator lengths at time t
        """
        t_clamped = np.clip(t/self.t_total, 0, 1)
        
        if t_clamped >= 1.0:
            return self.lengths[-1]
        # linear interpolation between points
        idx_float = t_clamped * (len(self.times) - 1)
        idx_low = int(np.floor(idx_float))
        idx_high = min(idx_low + 1, len(self.times) - 1)
        alpha = idx_float - idx_low
        return (1 - alpha) * self.lengths[idx_low] + alpha * self.lengths[idx_high]

        
        
        
    
    
    