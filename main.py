from data_loader import DataLoader
from visualization import SLAMVisualizer
from slam_solver import SLAMSolver
from landmark_detector import LandmarkDetector

# main.py
def main():
    # Initialize classes
    data_loader = DataLoader()
    slam_solver = SLAMSolver()
    landmark_detector = LandmarkDetector()
    visualizer = SLAMVisualizer()
    
    # Load data
    data_loader.load_laser_data('laser_data.csv')
    data_loader.load_odometry_data('odometry_data.csv')
    data_loader.load_gps_data('gps_data.csv')
    
    # TODO: Process data and build factor graph
    # 1. Process odometry data to get relative pose changes
    # 2. Process laser scans to detect landmarks
    # 3. Add appropriate factors to the graph
    # 4. Optimize and visualize results
    
    # Example optimization loop
    result = slam_solver.optimize()
    
    # Visualize results
    visualizer.plot_trajectory(result)

if __name__ == "__main__":
    main()