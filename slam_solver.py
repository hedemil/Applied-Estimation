# slam_solver.py
import numpy as np
import gtsam
from gtsam import symbol

class SLAMSolver:
    def __init__(self):
        # Initialize the factor graph
        self.graph = gtsam.NonlinearFactorGraph()
        # Initialize the initial estimate
        self.initial_estimate = gtsam.Values()
        
        # Parameters for the car model
        self.CAR_A = 0.95  # m (distance from rear axis to laser scanner)
        self.CAR_B = 0.5   # m
        self.CAR_H = 0.76  # m
        self.CAR_L = 2.83  # m
        
        # Noise models
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1]))  # x, y, theta
        self.landmark_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1]))  # x, y
        self.gps_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1.0, 1.0]))  # Large uncertainty due to GPS issues
    
    def add_odometry_factor(self, pose1_id, pose2_id, dx, dy, dtheta):
        """Add an odometry factor between two poses"""
        delta = gtsam.Pose2(dx, dy, dtheta)
        self.graph.add(gtsam.BetweenFactorPose2(
            symbol('x', pose1_id), 
            symbol('x', pose2_id),
            delta,
            self.odometry_noise
        ))
    
    def add_landmark_factor(self, pose_id, landmark_id, range_meas, bearing_meas):
        """Add a landmark factor"""
        self.graph.add(gtsam.BearingRangeFactor2D(
            symbol('x', pose_id),
            symbol('l', landmark_id),
            gtsam.Rot2(bearing_meas),
            range_meas,
            self.landmark_noise
        ))
    
    def add_gps_factor(self, pose_id, x, y):
        """Add a GPS factor (weak prior)"""
        gps_factor = gtsam.PriorFactorPose2(
            symbol('x', pose_id),
            gtsam.Pose2(x, y, 0),  # We don't trust GPS orientation
            self.gps_noise
        )
        self.graph.add(gps_factor)
    
    def optimize(self):
        """Optimize the factor graph"""
        parameters = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(
            self.graph, 
            self.initial_estimate, 
            parameters
        )
        return optimizer.optimize()