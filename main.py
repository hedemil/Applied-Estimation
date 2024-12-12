from data_loader import DataLoader
from visualization import SLAMVisualizer
from slam_solver import SLAMSolver
from landmark_detector import LandmarkDetector
import pandas as pd
import gtsam.utils.plot as gtsam_plot
import matplotlib.pyplot as plt
from gtsam.symbol_shorthand import L  # For landmark symbols
import numpy as np  # For numerical operations

# main.py
def main():
    # Initialize classes
    data_loader = DataLoader()
    slam_solver = SLAMSolver()
    landmark_detector = LandmarkDetector()
    # visualizer = SLAMVisualizer()
    
    # Load data
    data_loader.load_laser_data('csv_output/laser_data.csv')
    data_loader.load_odometry_data('csv_output/dead_reckoning.csv')
    data_loader.load_gps_data('csv_output/gps_data.csv')
    
    # TODO: Process data and build factor graph
    # 1. Process odometry data to get relative pose changes
    # 2. Process laser scans to detect landmarks
    # 3. Add appropriate factors to the graph
    # 4. Optimize and visualize results

    # Convert timestamp data to milliseconds
    data_loader.laser_data['timestamp'] = data_loader.laser_data['timestamp'] / 1000
    data_loader.odometry_data['timestamp'] = data_loader.odometry_data['timestamp'] / 1000
    data_loader.gps_data['timestamp'] = data_loader.gps_data['timestamp'] / 1000

    # Add prior on first pose (assuming starting at origin)
    slam_solver.add_first_pose(0.0, 0.0, 0.0)
    
    # Initialize landmark tracking
    landmark_map = {}
    next_landmark_id = 0

    # Create a mapping from odometry timestamps to sequential pose IDs
    pose_id_map = {}
    next_pose_id = 0

    # Process odometry data with sequential IDs
    for i in range(1, len(data_loader.odometry_data)):
        row_curr = data_loader.odometry_data.iloc[i]
        row_prev = data_loader.odometry_data.iloc[i - 1]
        
        # Map timestamps to sequential pose IDs
        curr_timestamp = row_curr['timestamp']
        prev_timestamp = row_prev['timestamp']
        
        if prev_timestamp not in pose_id_map:
            pose_id_map[prev_timestamp] = next_pose_id
            next_pose_id += 1
        if curr_timestamp not in pose_id_map:
            pose_id_map[curr_timestamp] = next_pose_id
            next_pose_id += 1
        
        # Get sequential pose IDs
        prev_pose_id = pose_id_map[prev_timestamp]
        curr_pose_id = pose_id_map[curr_timestamp]
        
        dt = curr_timestamp - prev_timestamp
        dx = row_curr['speed'] * dt
        dtheta = row_curr['steering'] * dt
        dy = 0.0  # Assume no lateral movement
        
        slam_solver.add_odometry_factor(prev_pose_id, curr_pose_id, dx, dy, dtheta)
    
    # Process laser data
    for i in range(len(data_loader.laser_data)):
        row = data_loader.laser_data.iloc[i]
        laser_data = row[1:]  # Skip timestamp
        
        # Detect landmarks
        landmarks = landmark_detector.detect_trees(laser_data)
        
        # Process each detected landmark
        for landmark in landmarks.T:  # landmarks is 3xN array
            distance = landmark[0]
            bearing = landmark[1]
            diameter = landmark[2]
            
            # Create landmark signature
            landmark_signature = f"{distance:.2f}_{diameter:.2f}"
            
            # Get or create landmark ID
            if landmark_signature not in landmark_map:
                landmark_map[landmark_signature] = next_landmark_id
                next_landmark_id += 1
            
            landmark_id = landmark_map[landmark_signature]
            slam_solver.add_landmark_factor(i, landmark_id, distance, bearing)
    
    # Optimize
    result, marginals = slam_solver.optimize()

    if result is not None:
        # Plot trajectory
        plt.figure(figsize=(12, 8))
        
        # Plot poses
        poses_x = []
        poses_y = []
        for i in range(len(data_loader.odometry_data)):
            try:
                pose = result.atPose2(i)
                poses_x.append(pose.x())
                poses_y.append(pose.y())
                gtsam_plot.plot_pose2(0, pose, 0.5,
                                    marginals.marginalCovariance(i))
            except RuntimeError as e:
                print(f"Could not plot pose {i}: {str(e)}")
        
        # Connect poses with lines
        plt.plot(poses_x, poses_y, 'b-', label='Robot Trajectory', alpha=0.5)
        
        # Plot landmarks
        for landmark_id in range(next_landmark_id):
            try:
                point = result.atPoint2(L(landmark_id))
                plt.plot(point[0], point[1], 'r*', markersize=10)
                
                # Plot landmark covariance
                landmark_cov = marginals.marginalCovariance(L(landmark_id))
                eigenvals, eigenvecs = np.linalg.eigh(landmark_cov)
                angle = np.arctan2(eigenvecs[1,0], eigenvecs[0,0])
                
                plt.error_ellipse(landmark_cov, point[0], point[1],
                                color='r', alpha=0.3)
            except RuntimeError as e:
                print(f"Could not plot landmark {landmark_id}: {str(e)}")
        
        # Plot GPS data if available
        if data_loader.gps_data is not None:
            plt.plot(data_loader.gps_data['longitude_m'], 
                    data_loader.gps_data['latitude_m'],
                    'g.', label='GPS Data', alpha=0.5)
        
        plt.grid(True)
        plt.axis('equal')
        plt.title('SLAM Results')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()