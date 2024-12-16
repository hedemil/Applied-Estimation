# slam_solver.py
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X
from models import VehicleParams, MotionModel, ObservationModel
from dataclasses import dataclass
from typing import Dict, Tuple, List

@dataclass
class NoiseParams:
    """Noise parameters for SLAM"""
    PRIOR_NOISE: gtsam.noiseModel.Diagonal = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.1]))
    ODOMETRY_NOISE: gtsam.noiseModel.Diagonal = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
    MEASUREMENT_NOISE: gtsam.noiseModel.Diagonal = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.2]))  # bearing, range


class GraphSLAM:
    """Main SLAM implementation using GTSAM"""
    def __init__(self, vehicle_params: VehicleParams, noise_params: NoiseParams = NoiseParams()):
        """
        TODO: Initialize SLAM components
        
        Hint:
        - Set up factor graph
        - Initialize pose
        - Create motion and observation models
        """

        # Initialize graph and modela
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.motion_model = MotionModel(vehicle_params)
        self.observation_model = ObservationModel(vehicle_params)

        # Store parameters
        self.vehicle_params = vehicle_params
        self.noise_params = NoiseParams()

        # Initialize state tracker
        self.current_pose_idx = 0
        self.landmark_dict: Dict[int, Tuple[float, float, float]] = {}  # landmark_id -> (x, y)
        self.DIAMETER_THRES = 0.2
        self.MIN_OBSERVATIONS = 3
        self.POSITION_THRES = 1.0

        # Track landmark observations
        self.landmark_observations: Dict[int, List[Tuple[float, float, float]]] = {}  # landmark_id -> list of (x, y, diameter)
        

        # Add prior on first pose
        first_pose = gtsam.Pose2(0, 0, 0)
        first_key = X(0)
        self.graph.add(gtsam.PriorFactorPose2(first_key, first_pose, noise_params.PRIOR_NOISE))
        self.initial_estimate.insert(first_key, first_pose)

        self.current_pose = first_pose
    
    def get_current_pose_key(self) -> gtsam.Key:
        return X(self.current_pose_idx)
    
    def get_lamdmark_key(self, landmark_id: int) -> gtsam.Key:
        return L(landmark_id)
    
    def add_odometry(self, velocity: float, steering: float, dt: float):
        """
        TODO: Add odometry factors to graph
        
        Hint:
        - Use motion model to predict
        - Create appropriate factors
        - Update estimates
        """

        # Get keys and current pose
        self.current_pose_idx += 1
        prev_key = X(self.current_pose_idx - 1)
        curr_key = X(self.current_pose_idx)
        prev_pose = self.initial_estimate.atPose2(prev_key)

        pred_pose = self.motion_model.predict(prev_pose, velocity, steering, dt)
        noise_model = self.motion_model.get_noise_model(velocity, steering)

        self.graph.add(
            gtsam.BetweenFactorPose2(
                prev_key,
                curr_key,
                pred_pose,
                noise_model
            )
        )

        self.initial_estimate.insert(curr_key, pred_pose)
    
    def add_landmarks(self, landmarks, current_pose: gtsam.Pose2):
        """
        TODO: Add landmark observations to graph
        
        Hint:
        - Use observation model
        - Handle data association
        - Create appropriate factors
        """
        curr_pose_key = X(self.current_pose_idx)
        a = self.vehicle_params.a
        b = self.vehicle_params.b
        L = self.vehicle_params.L


        for dist, angle, diam in landmarks:
            sensor_x = dist * np.sin(angle)
            sensor_y = dist * np.cos(angle)
            # Correct?? 
            global_x = current_pose.x() + sensor_x + (L + a)*np.cos(current_pose.theta())
            global_y = current_pose.y() + sensor_y + b + (L + a)*np.sin(current_pose.theta())

            measurement = gtsam.Point2(dist, angle)

            matched = False
            best_score = -np.inf
            best_match = None

            # Check if prev seen landmark
            for lm_idx, (lm_x, lm_y, lm_diam) in self.landmark_dict.items():
                
                # Check diameter similarity
                diameter_diff = abs(diam - lm_diam)
                if diameter_diff > self.DIAMETER_THRES:
                    continue
                
                # Check position
                pos_diff = np.sqrt((global_x - lm_x)**2 + (global_y - lm_y)**2)
                if pos_diff > self.POSITION_THRES:
                    continue

                landmark_pnt = gtsam.Point2(lm_x, lm_y)
                predicted = self.observation_model(current_pose, landmark_pnt)

                noise_model = self.observation_model.get_noise_model(dist)

                measurement_score = self.observation_model.measurment_likelihood(
                    measurement, predicted, noise_model
                )

                diam_score = 1.0 - (diameter_diff/self.DIAMETER_THRES)
                comb_score = measurement_score + diam_score


                if comb_score > best_score:
                    best_score = comb_score
                    best_match = (lm_idx, noise_model)

            if matched:
                lm_idx, noise_model = best_match
                lm_key = L(lm_idx)

                if lm_idx not in self.landmark_observations:
                    self.landmark_observations[lm_idx] = []
                self.landmark_observations[lm_idx].append(global_x, global_y, diam)

                self.graph.add(
                    gtsam.BearingRangeFactor2D(
                        curr_pose_key,
                        lm_key,
                        gtsam.Rot2(angle),
                        dist,
                        noise_model
                    )
                )

                if len(self.landmark_observations[lm_idx]) >= self.MIN_OBSERVATIONS:
                    self.check_loop_closure(lm_idx)

            else:
                lm_idx = len(self.landmark_dict)
                lm_key = L(lm_idx)

                self.landmark_dict[lm_idx] = (global_x, global_y, diam)
                self.landmark_observations[lm_idx] = [(global_x, global_y, diam)]

                point = gtsam.Point2(global_x, global_y)
                self.initial_estimate.insert(lm_key, point)

                noise_model = self.observation_model.get_noise_model(dist)
                self.graph.add(
                    gtsam.BearingRangeFactor2D(
                        curr_pose_key,
                        lm_key,
                        gtsam.Rot2(angle),
                        dist,
                        noise_model
                    )
                )
    def check_loop_closure(self, landmark_idx: int):
        observations = self.landmark_observations[landmark_idx]
        if len(observations) < self.MIN_OBSERVATIONS:
            return
        
        positions = np.array([(x,y) for x, y, _ in observations])
        diameters = np.array([(d for _, _, d in observations)])

        avg_pos = np.mean(positions, axis=0)
        avg_diameter = np.mean(diameters)
        std_diameter = np.std(diameters)
        
        # Update landmark position with average
        self.landmark_dict[landmark_idx] = (avg_pos[0], avg_pos[1], avg_diameter)
        
        # If diameter is consistent (low std), increase confidence
        if std_diameter < self.DIAMETER_THRESHOLD / 2:
            # Could add additional loop closure factors here
            # or adjust noise models based on confidence
            pass

    
    def optimize(self):
        """
        TODO: Optimize the factor graph
        
        Hint:
        - Consider when to optimize
        - Choose appropriate optimizer parameters
        """
        pass