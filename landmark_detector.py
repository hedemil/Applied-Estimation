import numpy as np
from typing import List, Tuple

class LandmarkDetector:
    """Handles landmark detection from laser scans"""
    def __init__(self):
        # Parameters from detectTreesI16.m
        self.M11 = 75.0  # Maximum range (meters)
        self.M10 = 1.0   # Minimum range (meters)
        self.M2 = 1.5    # Range difference threshold
        self.M2a = 10 * np.pi / 360  # Angle difference threshold
        self.M3 = 3.0    # Cluster size threshold
        self.M5 = 1.0    # Diameter threshold
        self.daa = 5 * np.pi / 306  # Minimum angle threshold
        self.daMin2 = 2 * np.pi / 360  # Minimum angular separation
        
    def detect_landmarks(self, ranges: np.ndarray, angles: np.ndarray) -> List[Tuple[float, float, float]]:
        """
        Detect tree-like landmarks from laser scan
        Returns: List of (distance, angle, diameter) tuples
        
        Implementation based on detectTrees.m from the provided code
        """
        landmarks = []
        
        # Filter valid ranges
        valid_idx = (ranges < self.M11) & (ranges > self.M10) & (angles > self.daa) & (angles < (np.pi - self.daa))
        valid_ranges = ranges[valid_idx]
        valid_angles = angles[valid_idx]
        
        if len(valid_ranges) < 3:
            return landmarks
            
        # Find potential segments
        range_diff = np.abs(np.diff(valid_ranges))
        angle_diff = np.diff(valid_angles)
        
        # Find discontinuities
        segment_breaks = np.where(
            (range_diff > self.M2) | (angle_diff > self.M2a)
        )[0]
        
        # Process each segment
        start_idx = 0
        for end_idx in np.append(segment_breaks, len(valid_ranges) - 1):
            if end_idx - start_idx < 3:
                start_idx = end_idx + 1
                continue
                
            # Get segment data
            segment_ranges = valid_ranges[start_idx:end_idx + 1]
            segment_angles = valid_angles[start_idx:end_idx + 1]
            
            # Calculate average range and angular width
            avg_range = np.mean(segment_ranges)
            ang_width = segment_angles[-1] - segment_angles[0]
            
            # Calculate diameter using arc length
            diameter = avg_range * ang_width
            
            if diameter < self.M5:
                # Calculate center point
                avg_angle = np.mean(segment_angles)
                landmarks.append((avg_range, avg_angle, diameter))
                
            start_idx = end_idx + 1
            
        return landmarks