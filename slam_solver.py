# slam_solver.py
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

class SLAMSolver:
    def __init__(self):
        # Initialize the factor graph
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        
        # Define noise models using gtsam.Point3
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(gtsam.Point3(0.3, 0.3, 0.1))
        self.odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(gtsam.Point3(0.2, 0.2, 0.1))
        self.measurement_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.2]))  # This one stays as np.array since it's 2D

        # Add a prior on the first pose
        self.first_pose_added = False

    def add_first_pose(self, x, y, theta):
        """Add a prior factor for the first pose"""
        if not self.first_pose_added:
            self.graph.add(gtsam.PriorFactorPose2(X(0), 
                                                 gtsam.Pose2(x, y, theta), 
                                                 self.prior_noise))
            self.initial_estimate.insert(X(0), gtsam.Pose2(x, y, theta))
            self.first_pose_added = True

    def add_odometry_factor(self, pose1_id, pose2_id, dx, dy, dtheta):
        """Add an odometry factor between two poses"""
        # Add factor to graph
        self.graph.add(gtsam.BetweenFactorPose2(
            X(pose1_id), 
            X(pose2_id),
            gtsam.Pose2(dx, dy, dtheta),
            self.odometry_noise
        ))
        
        # Add initial estimate for new pose
        if not self.initial_estimate.exists(X(pose2_id)):
            # Always use previous pose for better initial estimate
            if self.initial_estimate.exists(X(pose1_id)):
                prev_pose = self.initial_estimate.atPose2(X(pose1_id))
                # Use compose for more accurate pose chain
                new_pose = prev_pose.compose(gtsam.Pose2(dx, dy, dtheta))
                # Add some noise to initial estimate to avoid deterministic problems
                noise_x = np.random.normal(0, 0.1)
                noise_y = np.random.normal(0, 0.1)
                noise_theta = np.random.normal(0, 0.05)
                new_pose = gtsam.Pose2(new_pose.x() + noise_x, 
                                    new_pose.y() + noise_y, 
                                    new_pose.theta() + noise_theta)
            else:
                new_pose = gtsam.Pose2(dx + 0.1, dy + 0.1, dtheta + 0.1)  # Add small offset
            
            self.initial_estimate.insert(X(pose2_id), new_pose)

    def add_landmark_factor(self, pose_id, landmark_id, range_meas, bearing_meas):
        """Add a landmark factor"""
        # Add factor to graph
        self.graph.add(gtsam.BearingRangeFactor2D(
            X(pose_id),
            L(landmark_id),
            gtsam.Rot2(bearing_meas),
            range_meas,
            self.measurement_noise
        ))
        
        # Add initial estimate for landmark if it doesn't exist
        if not self.initial_estimate.exists(L(landmark_id)):
            # Get pose estimate
            if self.initial_estimate.exists(X(pose_id)):
                pose = self.initial_estimate.atPose2(X(pose_id))
                # Calculate approximate landmark position
                landmark_x = pose.x() + range_meas * np.cos(bearing_meas + pose.theta())
                landmark_y = pose.y() + range_meas * np.sin(bearing_meas + pose.theta())
                self.initial_estimate.insert(L(landmark_id), gtsam.Point2(landmark_x, landmark_y))
                

    def optimize(self):
        """Optimize the factor graph"""
        # Print graph for debugging
        print("\nFactor Graph:\n{}".format(self.graph))
        print("\nInitial Estimate:\n{}".format(self.initial_estimate))

        # Create optimizer with more detailed parameters
        parameters = gtsam.GaussNewtonParams()
        parameters.setRelativeErrorTol(1e-5)
        parameters.setMaxIterations(100)
        parameters.setVerbosity('ERROR')  # Print error messages
        
        # Create the optimizer
        optimizer = gtsam.GaussNewtonOptimizer(self.graph, self.initial_estimate, parameters)
        
        try:
            # Optimize and return result
            result = optimizer.optimize()
            print("\nFinal Result:\n{}".format(result))
            
            # Calculate marginals (covariances)
            marginals = gtsam.Marginals(self.graph, result)
            
            return result, marginals
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            return None, None