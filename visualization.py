# visualization.py
import matplotlib.pyplot as plt

class SLAMVisualizer:
    def __init__(self):
        self.figure = plt.figure(figsize=(12, 8))
        
    def plot_trajectory(self, poses, landmarks=None, gps_data=None):
        """Plot the estimated trajectory and landmarks"""
        plt.clf()
        
        # Plot trajectory
        x = []
        y = []
        for pose_id in range(len(poses)):
            pose = poses[symbol('x', pose_id)]
            x.append(pose.x())
            y.append(pose.y())
        plt.plot(x, y, 'b-', label='Estimated Trajectory')
        
        # Plot landmarks if available
        if landmarks:
            landmark_x = []
            landmark_y = []
            for landmark_id in landmarks:
                landmark = poses[symbol('l', landmark_id)]
                landmark_x.append(landmark.x())
                landmark_y.append(landmark.y())
            plt.scatter(landmark_x, landmark_y, c='r', marker='*', 
                       label='Landmarks')
        
        # Plot GPS data if available
        if gps_data is not None:
            plt.plot(gps_data['longitude_m'], gps_data['latitude_m'], 
                    'g.', label='GPS Data')
        
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        plt.title('SLAM Results')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.show()