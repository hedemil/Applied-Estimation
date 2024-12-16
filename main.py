from data_loader import DataLoader
from visualization import SLAMVisualizer
from slam_solver import GraphSLAM
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
    slam_solver = GraphSLAM()
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

    

if __name__ == "__main__":
    main()