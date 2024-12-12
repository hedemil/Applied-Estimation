# data_loader.py
import numpy as np
import pandas as pd
import gtsam



class DataLoader:
    def __init__(self):
        self.laser_data = None
        self.odometry_data = None
        self.gps_data = None
        
    def load_laser_data(self, filepath):
        """
        Load laser scan data
        Expected CSV format: timestamp, range1, range2, ..., range361
        """
        self.laser_data = pd.read_csv(filepath)
        
    def load_odometry_data(self, filepath):
        """
        Load dead reckoning data
        Expected CSV format: timestamp, speed, steering
        """
        self.odometry_data = pd.read_csv(filepath)
        
    def load_gps_data(self, filepath):
        """
        Load GPS data
        Expected CSV format: timestamp, longitude_m, latitude_m
        """
        self.gps_data = pd.read_csv(filepath)