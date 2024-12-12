import matplotlib.pyplot as plt
import gtsam.utils.plot as gtsam_plot
from gtsam.symbol_shorthand import L, X
import numpy as np

class SLAMVisualizer:
    def __init__(self):
        """Initialize the visualizer with default plot settings"""
        plt.ion()  # Enable interactive plotting
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(1,1,1)
        
    def plot_trajectory_and_landmarks(self, result, marginals, gps_data=None):
        """
        Plot the complete SLAM result including:
        - Robot trajectory with pose covariances
        - Landmarks with position covariances
        - GPS data (if available)
        """
        plt.clf()
        
        # Plot robot poses and their covariances
        poses_x = []
        poses_y = []
        for i in range(result.size()):
            if result.exists(X(i)):
                pose = result.atPose2(X(i))
                poses_x.append(pose.x())
                poses_y.append(pose.y())
                
                # Plot pose
                gtsam_plot.plot_pose2(self.ax, pose, 0.5)
                
                # Plot covariance ellipse
                covariance = marginals.marginalCovariance(X(i))
                self._plot_covariance_ellipse(
                    pose.x(), pose.y(),
                    covariance[0:2, 0:2],  # Position covariance
                    color='b',
                    alpha=0.3
                )
        
        # Connect poses with lines
        plt.plot(poses_x, poses_y, 'b-', label='Robot Trajectory', alpha=0.5)
        
        # Plot landmarks and their covariances
        for i in range(result.size()):
            if result.exists(L(i)):
                landmark = result.atPoint2(L(i))
                plt.plot(landmark[0], landmark[1], 'r*', markersize=10, label='_landmark')
                
                # Plot landmark covariance
                landmark_cov = marginals.marginalCovariance(L(i))
                self._plot_covariance_ellipse(
                    landmark[0], landmark[1],
                    landmark_cov,
                    color='r',
                    alpha=0.3
                )
        
        # Plot GPS data if available
        if gps_data is not None:
            plt.plot(gps_data['longitude_m'], gps_data['latitude_m'], 
                    'g.', label='GPS Measurements', alpha=0.5)
        
        # Add legend (only once for each type)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.grid(True)
        plt.axis('equal')
        plt.title('SLAM Results')
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        
        # Draw and pause to update
        plt.draw()
        plt.pause(0.001)
    
    def _plot_covariance_ellipse(self, x, y, covariance, color='b', alpha=0.3):
        """
        Plot a covariance ellipse at the specified position
        
        Args:
            x, y: Center of the ellipse
            covariance: 2x2 covariance matrix
            color: Color of the ellipse
            alpha: Transparency of the ellipse
        """
        eigenvals, eigenvecs = np.linalg.eigh(covariance)
        
        # Get largest eigenvalue and eigenvector
        largest_eigval = np.sqrt(np.abs(eigenvals[-1]))
        largest_eigvec = eigenvecs[:, -1]
        
        # Get angle of largest eigenvector
        angle = np.arctan2(largest_eigvec[1], largest_eigvec[0])
        
        # Confidence interval = 95% -> scale = 2.448
        scale = 2.448
        
        # Generate points for ellipse
        theta = np.linspace(0, 2*np.pi, 100)
        a = largest_eigval * scale
        b = np.sqrt(np.abs(eigenvals[0])) * scale
        
        # Generate ellipse points
        ex = a * np.cos(theta)
        ey = b * np.sin(theta)
        
        # Rotate points
        R = np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])
        
        ex_r = []
        ey_r = []
        for i in range(len(ex)):
            point = np.dot(R, np.array([ex[i], ey[i]]))
            ex_r.append(point[0])
            ey_r.append(point[1])
        
        # Plot ellipse
        plt.fill(x + ex_r, y + ey_r, color=color, alpha=alpha)
    
    def save_plot(self, filename):
        """Save the current plot to a file"""
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    
    def show(self):
        """Display the plot (blocking)"""
        plt.ioff()
        plt.show()
    
    def close(self):
        """Close the plot"""
        plt.close(self.fig)